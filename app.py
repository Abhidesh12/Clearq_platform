import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import (Depends, FastAPI, Form, HTTPException, Request, Response,
                     UploadFile, File)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from sqlmodel import Field, Session, SQLModel, create_engine, select
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import smtplib, ssl
from email.message import EmailMessage
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
import logging
import secrets, json

# Load environment
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./dev.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me')
SMTP_HOST = os.getenv('SMTP_HOST')

# Use pbkdf2_sha256 primary scheme to avoid bcrypt backend issues in some test environments
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")
serializer = URLSafeTimedSerializer(SECRET_KEY)

app = FastAPI(title='ClearQ')
# Session middleware required by authlib (Google OAuth)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

# OAuth client (Google)
oauth = OAuth()
if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
    oauth.register(
        name='google',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

logging.basicConfig(level=logging.INFO)


# Database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {})


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    password_hash: Optional[str] = None
    role: str = Field(default='learner')  # learner, mentor, admin
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    full_name: Optional[str] = None
    profile_image: Optional[str] = None
    google_id: Optional[str] = None


class MentorProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key='user.id')
    bio: Optional[str] = None
    experience: Optional[str] = None
    industry: Optional[str] = None
    is_approved: bool = Field(default=False)


# Services, availability, bookings, and reviews
from sqlalchemy import UniqueConstraint

class Service(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    mentor_id: int = Field(foreign_key='user.id', index=True)
    name: str
    description: Optional[str] = None
    price: int = Field(default=0)  # in INR rupees
    duration_minutes: int = Field(default=60)
    digital_product_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Availability(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    mentor_id: int = Field(foreign_key='user.id', index=True)
    date: str = Field(nullable=False)  # YYYY-MM-DD
    start_time: str = Field(nullable=False)  # HH:MM
    end_time: str = Field(nullable=False)  # HH:MM


class Booking(SQLModel, table=True):
    __table_args__ = (UniqueConstraint('mentor_id', 'service_id', 'date', 'time', name='uq_booking_slot'),)
    id: Optional[int] = Field(default=None, primary_key=True)
    service_id: int = Field(foreign_key='service.id', index=True)
    mentor_id: int = Field(foreign_key='user.id', index=True)
    learner_id: int = Field(foreign_key='user.id', index=True)
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    status: str = Field(default='created')  # created, pending_payment, paid, cancelled, completed
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    razorpay_refund_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    booking_id: int = Field(foreign_key='booking.id')
    learner_id: int = Field(foreign_key='user.id')
    mentor_id: int = Field(foreign_key='user.id')
    rating: int = Field(default=5)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# Razorpay client (optional - run in test mode with test keys)
import razorpay
RAZORPAY_KEY_ID = os.getenv('RAZORPAY_KEY_ID')
RAZORPAY_KEY_SECRET = os.getenv('RAZORPAY_KEY_SECRET')
RAZORPAY_WEBHOOK_SECRET = os.getenv('RAZORPAY_WEBHOOK_SECRET')
razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


@app.on_event('startup')
def on_startup():
    create_db_and_tables()


# Authentication helpers (cookie-based session)
SESSION_COOKIE = 'session'


def hash_password(password: str) -> str:
    try:
        return pwd_context.hash(password)
    except ValueError as e:
        # bcrypt backend can raise when password > 72 bytes; truncate as a fallback
        msg = str(e)
        if 'longer than 72 bytes' in msg:
            truncated = password[:72]
            return pwd_context.hash(truncated)
        raise


def verify_password(password: str, hash: str) -> bool:
    return pwd_context.verify(password, hash)


def create_session_token(user_id: int) -> str:
    return serializer.dumps({'user_id': user_id})


def load_session_token(token: str, max_age: int = 60 * 60 * 24 * 7) -> Optional[int]:
    try:
        data = serializer.loads(token, max_age=max_age)
        return int(data.get('user_id'))
    except (BadSignature, SignatureExpired):
        return None


def send_verification_email(to_email: str, verify_url: str):
    """Send a simple verification email using SMTP configured in environment."""
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', 587))
    user = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASSWORD')
    from_email = os.getenv('FROM_EMAIL') or user

    if not host or not user or not password:
        # Fallback: print URL for development
        logging.warning('SMTP not configured, printing verification URL instead')
        print(f'Verification URL for {to_email}: {verify_url}')
        return

    msg = EmailMessage()
    msg['Subject'] = 'Verify your ClearQ email'
    msg['From'] = from_email
    msg['To'] = to_email
    msg.set_content(f'Please verify your email by visiting: {verify_url}')

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls(context=context)
            server.login(user, password)
            server.send_message(msg)
        logging.info('Sent verification email to %s', to_email)
    except Exception as e:
        logging.exception('Failed to send verification email: %s', e)


def get_current_user(request: Request) -> Optional[User]:
    session = request.cookies.get(SESSION_COOKIE)
    if not session:
        return None
    user_id = load_session_token(session)
    if not user_id:
        return None
    with Session(engine) as s:
        user = s.get(User, user_id)
        return user


# Template globals: ensure env globals available (compat across versions)
if hasattr(templates, 'env'):
    templates.env.globals.update({})
elif hasattr(templates, 'environment'):
    templates.environment.globals.update({})


def flash(request: Request, message: str, category: str = 'info'):
    """Store a flash message in session to be shown on next page load."""
    flashes = request.session.get('_flashes', [])
    flashes.append({'message': message, 'category': category})
    request.session['_flashes'] = flashes


def get_and_clear_flashes(request: Request):
    flashes = request.session.pop('_flashes', []) if request.session.get('_flashes') else []
    return flashes


def get_csrf_token(request: Request):
    token = request.session.get('_csrf')
    if not token:
        token = secrets.token_urlsafe(32)
        request.session['_csrf'] = token
    return token


async def validate_csrf(request: Request):
    # Only validate POSTs and skip webhook endpoints
    if request.method != 'POST':
        return
    path = request.url.path
    if path.startswith('/webhook/'):
        return
    token = request.session.get('_csrf')
    # Fallback to cookie if session store is not available (helps test client flows)
    if not token:
        token = request.cookies.get('csrf_token')
    if not token:
        raise HTTPException(status_code=403, detail='CSRF token missing')
    # For form posts, require the csrf_token form field (safer for non-AJAX forms).
    # For JSON/AJAX, accept X-CSRF-Token header.
    content_type = request.headers.get('content-type', '')
    submitted = None
    if content_type.startswith('application/x-www-form-urlencoded') or content_type.startswith('multipart/form-data'):
        # Prefer reading parsed form data (works even if FastAPI already parsed it)
        try:
            form = await request.form()
            submitted = form.get('csrf_token')
        except Exception:
            # fallback to raw body parsing (best effort)
            try:
                import urllib.parse
                body = await request.body()
                params = urllib.parse.parse_qs(body.decode())
                submitted = params.get('csrf_token', [None])[0]
            except Exception:
                submitted = None
        # fallback to header if provided
        if not submitted:
            submitted = request.headers.get('X-CSRF-Token')
    else:
        submitted = request.headers.get('X-CSRF-Token') or request.cookies.get('csrf_token')

    if not submitted or submitted != token:
        raise HTTPException(status_code=403, detail='Invalid CSRF token')


def render_template(request: Request, template_name: str, context: dict):
    # inject current_user if not present
    if 'current_user' not in context:
        context['current_user'] = getattr(request.state, 'user', None)
    context['flashes'] = get_and_clear_flashes(request)
    context['request'] = request
    csrf = get_csrf_token(request)
    context['csrf_token'] = csrf
    # Build response and set csrf cookie so that POSTs will carry it and validation can use it
    resp = templates.TemplateResponse(template_name, context)
    resp.set_cookie('csrf_token', csrf)
    return resp


# Dependencies
def inject_user(request: Request):
    user = get_current_user(request)
    request.state.user = user
    return user


@app.get('/', response_class=HTMLResponse)
def index(request: Request, current_user: User = Depends(inject_user)):
    return render_template(request, 'index.html', {'year': datetime.utcnow().year})


@app.get('/health')
def health():
    """Simple health check for deployment platforms (returns 200)."""
    return {'status': 'ok'}


@app.get('/explore', response_class=HTMLResponse)
def explore(request: Request, current_user: User = Depends(inject_user)):
    # List services as explore results
    with Session(engine) as s:
        services = s.exec(select(Service).order_by(Service.created_at.desc())).all()
    return render_template(request, 'explore.html', {'services': services, 'year': datetime.utcnow().year})


@app.get('/register', response_class=HTMLResponse, name='register_get')
def register_get(request: Request):
    return render_template(request, 'register.html', {'year': datetime.utcnow().year})


# Mentor endpoints: add service and availability
@app.get('/mentor/service/new', response_class=HTMLResponse)
def mentor_service_get(request: Request, current_user: User = Depends(inject_user)):
    if not current_user or current_user.role != 'mentor':
        return RedirectResponse(url='/login')
    return render_template(request, 'mentor_add_service.html', {})


@app.post('/mentor/service/create')
async def mentor_service_create(request: Request, name: str = Form(...), description: str = Form(None), price: int = Form(0), duration_minutes: int = Form(60), digital_product_url: str = Form(None), current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'mentor':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        service = Service(mentor_id=current_user.id, name=name, description=description, price=price, duration_minutes=duration_minutes, digital_product_url=digital_product_url)
        s.add(service)
        s.commit()
        s.refresh(service)
    flash(request, 'Service created', 'success')
    return RedirectResponse(url=f'/service/{service.id}', status_code=302)


@app.get('/mentor/availability/new', response_class=HTMLResponse)
def mentor_availability_get(request: Request, current_user: User = Depends(inject_user)):
    if not current_user or current_user.role != 'mentor':
        return RedirectResponse(url='/login')
    return render_template(request, 'mentor_add_availability.html', {})


@app.post('/mentor/availability/create')
async def mentor_availability_create(request: Request, date: str = Form(...), start_time: str = Form(...), end_time: str = Form(...), current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'mentor':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        av = Availability(mentor_id=current_user.id, date=date, start_time=start_time, end_time=end_time)
        s.add(av)
        s.commit()
    flash(request, 'Availability added', 'success')
    return RedirectResponse(url='/mentor/availability/new', status_code=302)


@app.post('/register')
async def register_post(request: Request, email: str = Form(...), password: str = Form(...), full_name: str = Form(None), role: str = Form('learner')):
    await validate_csrf(request)
    with Session(engine) as s:
        statement = select(User).where(User.email == email)
        existing = s.exec(statement).first()
        if existing:
            return render_template(request, 'register.html', {'error': 'Email already registered'})
        user = User(email=email, password_hash=hash_password(password), role=role, full_name=full_name)
        s.add(user)
        s.commit()
        # send verification (placeholder prints)
        token = serializer.dumps({'user_id': user.id})
        verify_url = str(request.url_for('verify_email')) + f'?token={token}'
        # Send verification email (will print if SMTP not configured)
        send_verification_email(user.email, verify_url)
        response = RedirectResponse(url='/', status_code=302)
        # auto-login for dev
        response.set_cookie(SESSION_COOKIE, create_session_token(user.id), httponly=True, max_age=60 * 60 * 24 * 7)
        return response


@app.get('/verify-email')
def verify_email(token: str = '', response: Response = None):
    user_id = load_session_token(token, max_age=60 * 60 * 24 * 7)
    if not user_id:
        raise HTTPException(status_code=400, detail='Invalid or expired token')
    with Session(engine) as s:
        user = s.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail='User not found')
        user.is_verified = True
        s.add(user)
        s.commit()
    return RedirectResponse(url='/', status_code=302)


@app.get('/login', response_class=HTMLResponse, name='login_get')
def login_get(request: Request):
    return render_template(request, 'login.html', {})


@app.post('/login')
async def login_post(request: Request, response: Response, email: str = Form(...), password: str = Form(...)):
    await validate_csrf(request)
    with Session(engine) as s:
        statement = select(User).where(User.email == email)
        user = s.exec(statement).first()
        if not user or not user.password_hash or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=400, detail='Invalid credentials')
        token = create_session_token(user.id)
        resp = RedirectResponse(url='/', status_code=302)
        resp.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=60 * 60 * 24 * 7)
        return resp


@app.get('/auth/google/login')
async def google_login(request: Request):
    # Redirect to Google's OAuth 2.0 authorization endpoint
    redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', str(request.url_for('google_callback')))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth/google/callback')
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        logging.exception('Google OAuth error: %s', e)
        return RedirectResponse(url='/login')

    # Try to get user info
    userinfo = None
    if 'userinfo' in token:
        userinfo = token.get('userinfo')
    else:
        resp = await oauth.google.get('userinfo', token=token)
        userinfo = resp.json()

    email = userinfo.get('email')
    google_id = userinfo.get('sub') or userinfo.get('id')
    full_name = userinfo.get('name')

    with Session(engine) as s:
        statement = select(User).where((User.google_id == google_id) | (User.email == email))
        existing = s.exec(statement).first()
        if existing:
            existing.google_id = google_id
            existing.is_verified = True
            s.add(existing)
            s.commit()
            user = existing
        else:
            user = User(email=email, role='learner', is_verified=True, google_id=google_id, full_name=full_name)
            s.add(user)
            s.commit()
            s.refresh(user)

    resp = RedirectResponse(url='/', status_code=302)
    resp.set_cookie(SESSION_COOKIE, create_session_token(user.id), httponly=True, max_age=60 * 60 * 24 * 7)
    return resp


@app.get('/logout')
def logout():
    resp = RedirectResponse(url='/', status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get('/profile', response_class=HTMLResponse)
def profile(request: Request, current_user: User = Depends(inject_user)):
    if not current_user:
        return RedirectResponse(url='/login')
    return render_template(request, 'profile.html', {})


@app.post('/profile')
async def profile_update(request: Request, full_name: str = Form(None), file: UploadFile = File(None), current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user:
        return RedirectResponse(url='/login')
    filename = None
    if file:
        ext = os.path.splitext(file.filename)[1]
        filename = f'user_{current_user.id}_profile{ext}'
        path = os.path.join('static', 'uploads', filename)
        # Use aiofiles to write content
        import aiofiles
        async with aiofiles.open(path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
    with Session(engine) as s:
        user = s.get(User, current_user.id)
        if full_name:
            user.full_name = full_name
        if filename:
            user.profile_image = f'/static/uploads/{filename}'
        s.add(user)
        s.commit()
    flash(request, 'Profile updated', 'success')
    return RedirectResponse(url='/profile', status_code=302)


# Simple API placeholder for time slots and booking
@app.post('/api/time-slots/{mentor_id}')
def time_slots(mentor_id: int, payload: dict):
    date = payload.get('date')
    # Return sample slots
    return {'success': True, 'slots': ['09:00', '10:30', '13:00', '15:30']}


@app.post('/api/create-booking')
async def create_booking(request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user:
        return {'success': False, 'message': 'Unauthorized'}
    payload = await request.json()
    service_id = int(payload.get('service_id')) if payload.get('service_id') else None
    mentor_id = int(payload.get('mentor_id')) if payload.get('mentor_id') else None
    date = payload.get('date')
    time = payload.get('time')
    if not (service_id and mentor_id and date and time):
        return {'success': False, 'message': 'Missing booking data'}
    with Session(engine) as s:
        # basic check: service exists and mentor matches
        service = s.get(Service, service_id)
        if not service or service.mentor_id != int(mentor_id):
            return {'success': False, 'message': 'Service not found'}
        # check existing booking
        existing = s.exec(select(Booking).where((Booking.service_id == service_id) & (Booking.date == date) & (Booking.time == time))).first()
        if existing:
            return {'success': False, 'message': 'Slot already booked'}
        booking = Booking(service_id=service_id, mentor_id=mentor_id, learner_id=current_user.id, date=date, time=time, status='created')
        s.add(booking)
        s.commit()
        s.refresh(booking)
        if razorpay_client:
            try:
                order = razorpay_client.order.create({'amount': service.price * 100, 'currency': 'INR', 'receipt': f'booking_{booking.id}', 'payment_capture': 1})
                booking.razorpay_order_id = order.get('id')
                booking.status = 'pending_payment'
                s.add(booking)
                s.commit()
                return {'success': True, 'booking_id': booking.id, 'order': order}
            except Exception as e:
                logging.exception('Failed to create Razorpay order: %s', e)
                # fallback: keep booking created but return error
                return {'success': False, 'message': 'Payment provider error'}
        else:
            booking.status = 'paid'
            s.add(booking)
            s.commit()
            return {'success': True, 'booking_id': booking.id}


@app.post('/payment/verify')
def payment_verify(payload: dict):
    """Verify Razorpay payment signature and mark booking as paid."""
    razorpay_payment_id = payload.get('razorpay_payment_id')
    razorpay_order_id = payload.get('razorpay_order_id')
    razorpay_signature = payload.get('razorpay_signature')

    if not (razorpay_payment_id and razorpay_order_id and razorpay_signature):
        return {'success': False, 'message': 'Missing signature data'}

    if not razorpay_client:
        return {'success': False, 'message': 'Razorpay not configured'}

    try:
        razorpay_client.utility.verify_payment_signature({'razorpay_order_id': razorpay_order_id, 'razorpay_payment_id': razorpay_payment_id}, razorpay_signature)
    except Exception as e:
        logging.exception('Razorpay signature verification failed: %s', e)
        return {'success': False, 'message': 'Invalid signature'}

    # Find booking with order id
    with Session(engine) as s:
        stmt = select(Booking).where(Booking.razorpay_order_id == razorpay_order_id)
        booking = s.exec(stmt).first()
        if not booking:
            return {'success': False, 'message': 'Booking not found for order'}
        booking.status = 'paid'
        booking.razorpay_payment_id = razorpay_payment_id
        s.add(booking)
        s.commit()
    return {'success': True, 'booking_id': booking.id}


@app.post('/webhook/razorpay')
async def razorpay_webhook(request: Request):
    # Raw body required for signature verification
    payload_raw = await request.body()
    signature = request.headers.get('X-Razorpay-Signature')
    if not RAZORPAY_WEBHOOK_SECRET or not signature:
        logging.warning('Webhook secret not configured or signature missing')
        return {'status': 'ignored'}
    try:
        # verify signature
        razorpay_client.utility.verify_webhook_signature(payload_raw, signature, RAZORPAY_WEBHOOK_SECRET)
    except Exception as e:
        logging.exception('Invalid webhook signature: %s', e)
        return {'status': 'invalid signature'}
    try:
        event = json.loads(payload_raw.decode('utf-8'))
        ev = event.get('event')
        # payment captured
        if ev == 'payment.captured' or ev == 'payment.authorized' or ev == 'order.paid':
            entity = event.get('payload', {}).get('payment', {}).get('entity') or {}
            order_id = entity.get('order_id')
            payment_id = entity.get('id')
            with Session(engine) as s:
                stmt = select(Booking).where(Booking.razorpay_order_id == order_id)
                booking = s.exec(stmt).first()
                if booking and booking.status != 'paid':
                    booking.status = 'paid'
                    booking.razorpay_payment_id = payment_id
                    s.add(booking)
                    s.commit()
        elif ev == 'payment.failed':
            entity = event.get('payload', {}).get('payment', {}).get('entity') or {}
            order_id = entity.get('order_id')
            with Session(engine) as s:
                stmt = select(Booking).where(Booking.razorpay_order_id == order_id)
                booking = s.exec(stmt).first()
                if booking:
                    booking.status = 'cancelled'
                    s.add(booking)
                    s.commit()
    except Exception as e:
        logging.exception('Error processing webhook: %s', e)
    return {'status': 'ok'}


@app.get('/service/{service_id}', response_class=HTMLResponse)
def service_detail(request: Request, service_id: int, current_user: User = Depends(inject_user)):
    with Session(engine) as s:
        service = s.get(Service, service_id)
        if not service:
            return RedirectResponse(url='/explore')
        # reviews for mentor
        reviews = s.exec(select(Review).where(Review.mentor_id == service.mentor_id).order_by(Review.created_at.desc())).all()
        # check if current user can review (has a completed booking for this service)
        can_review = False
        completed_booking_id = None
        if current_user and current_user.role == 'learner':
            stmt = select(Booking).where((Booking.service_id == service.id) & (Booking.learner_id == current_user.id) & (Booking.status == 'completed'))
            completed_booking = s.exec(stmt).first()
            if completed_booking:
                # ensure no existing review
                rstmt = select(Review).where(Review.booking_id == completed_booking.id)
                existing_review = s.exec(rstmt).first()
                can_review = not bool(existing_review)
                if can_review:
                    completed_booking_id = completed_booking.id
    return render_template(request, 'service_detail.html', {'service': service, 'reviews': reviews, 'can_review': can_review, 'completed_booking_id': completed_booking_id})


@app.get('/dashboard', response_class=HTMLResponse)
def dashboard(request: Request, current_user: User = Depends(inject_user)):
    if not current_user:
        return RedirectResponse(url='/login')

    with Session(engine) as s:
        if current_user.role == 'mentor':
            # Mentor dashboard: services, availability, bookings, earnings
            services = s.exec(select(Service).where(Service.mentor_id == current_user.id)).all()
            raw_bookings = s.exec(select(Booking).where(Booking.mentor_id == current_user.id).order_by(Booking.created_at.desc())).all()
            bookings = []
            earnings = 0
            for b in raw_bookings:
                svc = s.get(Service, b.service_id)
                learner = s.get(User, b.learner_id)
                bookings.append({'id': b.id, 'service_name': svc.name if svc else '', 'learner_email': learner.email if learner else '', 'date': b.date, 'time': b.time, 'status': b.status})
                if b.status == 'paid' and svc:
                    earnings += svc.price
            return render_template(request, 'dashboard_mentor.html', {'services': services, 'bookings': bookings, 'earnings': earnings})
        elif current_user.role == 'learner':
            # Learner dashboard: bookings
            raw_bookings = s.exec(select(Booking).where(Booking.learner_id == current_user.id).order_by(Booking.created_at.desc())).all()
            bookings = []
            for b in raw_bookings:
                svc = s.get(Service, b.service_id)
                mentor = s.get(User, b.mentor_id)
                bookings.append({'id': b.id, 'service_name': svc.name if svc else '', 'mentor_email': mentor.email if mentor else '', 'date': b.date, 'time': b.time, 'status': b.status, 'service_id': b.service_id})
            return render_template(request, 'dashboard_learner.html', {'bookings': bookings})
        elif current_user.role == 'admin':
            # Admin dashboard: quick stats
            pending_mentors = s.exec(select(MentorProfile).where(MentorProfile.is_approved == False)).all()
            total_bookings = s.exec(select(Booking)).all()
            return render_template(request, 'dashboard_admin.html', {'pending_mentors': pending_mentors, 'total_bookings': len(total_bookings)})


# Admin: list mentor profiles pending approval
@app.get('/admin/mentors', response_class=HTMLResponse)
def admin_list_mentors(request: Request, current_user: User = Depends(inject_user)):
    if not current_user or current_user.role != 'admin':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        pending = s.exec(select(MentorProfile).where(MentorProfile.is_approved == False)).all()
        # attach user emails
        mentor_data = []
        for m in pending:
            user = s.get(User, m.user_id)
            mentor_data.append({'profile': m, 'user': user})
    return render_template(request, 'admin_mentors.html', {'pending': mentor_data})


# Admin: list bookings for refunding
@app.get('/admin/bookings', response_class=HTMLResponse)
def admin_list_bookings(request: Request, current_user: User = Depends(inject_user)):
    if not current_user or current_user.role != 'admin':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        raw = s.exec(select(Booking).order_by(Booking.created_at.desc())).all()
        bookings = []
        for b in raw:
            svc = s.get(Service, b.service_id)
            mentor = s.get(User, b.mentor_id)
            learner = s.get(User, b.learner_id)
            bookings.append({'id': b.id, 'service_name': svc.name if svc else '', 'mentor_email': mentor.email if mentor else '', 'learner_email': learner.email if learner else '', 'date': b.date, 'time': b.time, 'status': b.status, 'razorpay_payment_id': b.razorpay_payment_id})
    return render_template(request, 'admin_bookings.html', {'bookings': bookings})


@app.post('/admin/booking/{booking_id}/refund')
async def admin_refund_booking(booking_id: int, request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'admin':
        return RedirectResponse(url='/login')
    if not razorpay_client:
        flash(request, 'Razorpay not configured', 'error')
        return RedirectResponse(url='/admin/bookings')
    with Session(engine) as s:
        booking = s.get(Booking, booking_id)
        if not booking or booking.status != 'paid' or not booking.razorpay_payment_id:
            flash(request, 'Cannot refund this booking', 'error')
            return RedirectResponse(url='/admin/bookings')
        svc = s.get(Service, booking.service_id)
        amount_paise = (svc.price * 100) if svc else None
        try:
            refund = razorpay_client.payment.refund(booking.razorpay_payment_id, {'amount': amount_paise})
            booking.status = 'refunded'
            booking.razorpay_refund_id = refund.get('id')
            s.add(booking)
            s.commit()
            flash(request, 'Refund initiated', 'success')
        except Exception as e:
            logging.exception('Refund failed: %s', e)
            flash(request, 'Refund failed', 'error')
    return RedirectResponse(url='/admin/bookings')


@app.post('/admin/mentor/{mentor_id}/approve')
async def admin_approve_mentor(mentor_id: int, request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'admin':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        mentor = s.get(MentorProfile, mentor_id)
        if not mentor:
            flash(request, 'Mentor profile not found', 'error')
            return RedirectResponse(url='/admin/mentors')
        mentor.is_approved = True
        s.add(mentor)
        # ensure user role is mentor
        user = s.get(User, mentor.user_id)
        if user:
            user.role = 'mentor'
            s.add(user)
        s.commit()
    flash(request, 'Mentor approved', 'success')
    return RedirectResponse(url='/admin/mentors')


@app.post('/admin/mentor/{mentor_id}/reject')
async def admin_reject_mentor(mentor_id: int, request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'admin':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        mentor = s.get(MentorProfile, mentor_id)
        if not mentor:
            flash(request, 'Mentor profile not found', 'error')
            return RedirectResponse(url='/admin/mentors')
        s.delete(mentor)
        s.commit()
    flash(request, 'Mentor rejected', 'success')
    return RedirectResponse(url='/admin/mentors')


# Mentor actions: complete or cancel a booking
@app.post('/mentor/booking/{booking_id}/complete')
async def mentor_complete_booking(booking_id: int, request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'mentor':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        booking = s.get(Booking, booking_id)
        if not booking or booking.mentor_id != current_user.id:
            flash(request, 'Booking not found or no permission', 'error')
            return RedirectResponse(url='/dashboard')
        booking.status = 'completed'
        s.add(booking)
        s.commit()
    flash(request, 'Booking marked completed', 'success')
    return RedirectResponse(url='/dashboard')


@app.post('/mentor/booking/{booking_id}/cancel')
async def mentor_cancel_booking(booking_id: int, request: Request, current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user:
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        booking = s.get(Booking, booking_id)
        if not booking:
            flash(request, 'Booking not found', 'error')
            return RedirectResponse(url='/dashboard')
        # allow mentor (owner) or learner (booker) to cancel
        if current_user.role == 'mentor' and booking.mentor_id == current_user.id:
            booking.status = 'cancelled'
        elif current_user.role == 'learner' and booking.learner_id == current_user.id:
            # allow learner cancellation if not completed
            if booking.status == 'completed':
                flash(request, 'Cannot cancel a completed booking', 'error')
                return RedirectResponse(url='/dashboard')
            booking.status = 'cancelled'
        else:
            flash(request, 'No permission to cancel this booking', 'error')
            return RedirectResponse(url='/dashboard')
        s.add(booking)
        s.commit()
    flash(request, 'Booking cancelled', 'success')
    return RedirectResponse(url='/dashboard')


# Learner review submission
@app.post('/booking/{booking_id}/review')
async def booking_review(booking_id: int, request: Request, rating: int = Form(...), comment: str = Form(None), current_user: User = Depends(inject_user)):
    await validate_csrf(request)
    if not current_user or current_user.role != 'learner':
        return RedirectResponse(url='/login')
    with Session(engine) as s:
        booking = s.get(Booking, booking_id)
        if not booking or booking.learner_id != current_user.id:
            flash(request, 'Booking not found or no permission', 'error')
            return RedirectResponse(url='/dashboard')
        if booking.status != 'completed':
            flash(request, 'Can only review completed bookings', 'error')
            return RedirectResponse(url='/dashboard')
        # ensure no existing review
        existing = s.exec(select(Review).where(Review.booking_id == booking.id)).first()
        if existing:
            flash(request, 'Review already submitted for this booking', 'error')
            return RedirectResponse(url=f'/service/{booking.service_id}')
        review = Review(booking_id=booking.id, learner_id=current_user.id, mentor_id=booking.mentor_id, rating=rating, comment=comment)
        s.add(review)
        s.commit()
    flash(request, 'Thank you for your review!', 'success')
    return RedirectResponse(url=f'/service/{booking.service_id}')


# Minimal templates for register/login/profile
register_html = '''{% extends "base.html" %}
{% block content %}
<h2>Register</h2>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<p><a href="/auth/google/login" style="display:inline-block;padding:0.5rem 1rem;border:1px solid #ddd;background:#fff;border-radius:6px;text-decoration:none;color:#111;margin-bottom:1rem;">Sign up with Google</a></p>
<form action="/register" method="post">
    <label>Email</label><br>
    <input type="email" name="email" required><br>
    <label>Password</label><br>
    <input type="password" name="password" required><br>
    <label>Full name</label><br>
    <input type="text" name="full_name"><br>
    <label>Role</label><br>
    <select name="role"><option value="learner">Learner</option><option value="mentor">Mentor</option></select><br><br>
    <button type="submit">Register</button>
</form>
{% endblock %}'''

login_html = '''{% extends "base.html" %}
{% block content %}
<h2>Login</h2>
<p><a href="/auth/google/login" style="display:inline-block;padding:0.5rem 1rem;border:1px solid #ddd;background:#fff;border-radius:6px;text-decoration:none;color:#111;margin-bottom:1rem;">Sign in with Google</a></p>
<form action="/login" method="post">
    <label>Email</label><br>
    <input type="email" name="email" required><br>
    <label>Password</label><br>
    <input type="password" name="password" required><br><br>
    <button type="submit">Login</button>
</form>
{% endblock %}'''

profile_html = '''{% extends "base.html" %}
{% block content %}
<h2>Profile</h2>
<p>Email: {{ current_user.email }}</p>
<p>Full name: {{ current_user.full_name }}</p>
{% if current_user.profile_image %}<img src="{{ current_user.profile_image }}" width="100">{% endif %}
<form action="/profile" method="post" enctype="multipart/form-data">
    <label>Full name</label><br>
    <input type="text" name="full_name"><br>
    <label>Profile image</label><br>
    <input type="file" name="file"><br><br>
    <button type="submit">Save</button>
</form>
{% endblock %}'''

# Write templates to disk if not exists (first-run convenience)
if not os.path.exists('templates/register.html'):
    with open('templates/register.html', 'w', encoding='utf-8') as f:
        f.write(register_html)
if not os.path.exists('templates/login.html'):
    with open('templates/login.html', 'w', encoding='utf-8') as f:
        f.write(login_html)
if not os.path.exists('templates/profile.html'):
    with open('templates/profile.html', 'w', encoding='utf-8') as f:
        f.write(profile_html)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
