import json
import hashlib
import hmac
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

import app as app_module
from app import app, engine, Session, User, Service, Booking, create_session_token, SESSION_COOKIE


client = TestClient(app)


def extract_csrf(html: str):
    # naive extraction of csrf_token value
    marker = 'name="csrf_token" value="'
    idx = html.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = html.find('"', start)
    return html[start:end]


@pytest.fixture(autouse=True)
def cleanup_db():
    # ensure tables exist and cleanup relevant rows before each test
    app_module.create_db_and_tables()
    with Session(engine) as s:
        s.exec(delete(Booking))
        s.exec(delete(Service))
        s.exec(delete(User))
        s.commit()
    yield
    with Session(engine) as s:
        s.exec(delete(Booking))
        s.exec(delete(Service))
        s.exec(delete(User))
        s.commit()


def test_csrf_requires_token_for_register():
    # GET register to obtain CSRF token
    r = client.get('/register')
    assert r.status_code == 200
    token = extract_csrf(r.text)
    assert token is not None

    # Missing CSRF should be forbidden
    resp = client.post('/register', data={'email': 'csrf1@example.com', 'password': 'pass'}, allow_redirects=False)
    assert resp.status_code == 403

    # With CSRF token should succeed (redirect)
    resp2 = client.post('/register', data={'email': 'csrf2@example.com', 'password': 'pass', 'csrf_token': token}, allow_redirects=False)
    assert resp2.status_code in (302, 303, 307)


def test_webhook_marks_booking_paid(monkeypatch):
    # create service and booking pending_payment
    with Session(engine) as s:
        mentor = User(email='mentor1@example.com', role='mentor', is_verified=True)
        learner = User(email='learner1@example.com', role='learner', is_verified=True)
        s.add(mentor)
        s.add(learner)
        s.commit()
        s.refresh(mentor)
        s.refresh(learner)
        svc = Service(mentor_id=mentor.id, name='Srv', price=500)
        s.add(svc)
        s.commit()
        s.refresh(svc)
        bk = Booking(service_id=svc.id, mentor_id=mentor.id, learner_id=learner.id, date='2025-12-31', time='10:00', status='pending_payment', razorpay_order_id='order_123')
        s.add(bk)
        s.commit()
        s.refresh(bk)
        booking_id = bk.id

    # monkeypatch razorpay_client utility verification to be a no-op
    monkeypatch.setattr(app_module, 'RAZORPAY_WEBHOOK_SECRET', 'testsecret')
    monkeypatch.setattr(app_module, 'razorpay_client', SimpleNamespace(utility=SimpleNamespace(verify_webhook_signature=lambda payload, signature, secret: None)))

    event = {'event': 'payment.captured', 'payload': {'payment': {'entity': {'order_id': 'order_123', 'id': 'pay_1'}}}}
    payload_raw = json.dumps(event).encode('utf-8')
    sig = hmac.new(b'testsecret', payload_raw, hashlib.sha256).hexdigest()

    headers = {'X-Razorpay-Signature': sig}
    res = client.post('/webhook/razorpay', content=payload_raw, headers=headers)
    assert res.status_code == 200
    assert res.json().get('status') == 'ok'

    # verify booking updated
    with Session(engine) as s:
        b = s.get(Booking, booking_id)
        assert b.status == 'paid'
        assert b.razorpay_payment_id == 'pay_1'


def test_admin_refund_process(monkeypatch):
    # create admin user and booking with paid status
    with Session(engine) as s:
        admin = User(email='admin@example.com', role='admin', is_verified=True)
        mentor = User(email='m2@example.com', role='mentor', is_verified=True)
        learner = User(email='l2@example.com', role='learner', is_verified=True)
        s.add(admin); s.add(mentor); s.add(learner); s.commit()
        s.refresh(admin); s.refresh(mentor); s.refresh(learner)
        admin_id = admin.id
        svc = Service(mentor_id=mentor.id, name='S2', price=300)
        s.add(svc); s.commit(); s.refresh(svc)
        bk = Booking(service_id=svc.id, mentor_id=mentor.id, learner_id=learner.id, date='2025-12-30', time='09:00', status='paid', razorpay_payment_id='pay_99')
        s.add(bk); s.commit(); s.refresh(bk)
        booking_id = bk.id

    # monkeypatch razorpay_client payment.refund
    def fake_refund(payment_id, payload=None):
        return {'id': 'refund_123'}

    monkeypatch.setattr(app_module, 'razorpay_client', SimpleNamespace(payment=SimpleNamespace(refund=fake_refund)))

    # login as admin by setting session cookie
    token = create_session_token(admin_id)
    client.cookies.set(SESSION_COOKIE, token)
    # GET admin bookings to fetch csrf token
    r = client.get('/admin/bookings')
    assert r.status_code == 200
    csrf = extract_csrf(r.text)
    assert csrf is not None

    # Perform refund
    resp = client.post(f'/admin/booking/{booking_id}/refund', data={'csrf_token': csrf}, allow_redirects=False)
    assert resp.status_code in (302, 303, 307)

    # verify booking updated
    with Session(engine) as s:
        b = s.get(Booking, booking_id)
        assert b.status == 'refunded'
        assert b.razorpay_refund_id == 'refund_123'