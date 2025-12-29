# ClearQ - Mentorship Platform (FastAPI prototype)

This repository contains a FastAPI-based mentorship platform scaffold inspired by Topmate. The backend is a single `app.py` file (for the prototype) and templates are server-side rendered (Jinja2). The app is designed to be deployed on Render with a Postgres DB.

Key features scaffolded:
- FastAPI + Jinja2 templates
- SQLModel-based models and DB init
- Email/password signup + email verification placeholder
- Google OAuth placeholder
- Profile photo upload to `static/uploads/`
- Booking skeleton (models + API endpoint placeholder)
- Razorpay placeholders for order creation and verification

Environment variables: copy `.env.example` to `.env` and fill in secrets.

Run locally:
1. python -m venv .venv
2. .venv\Scripts\activate (Windows) or source .venv/bin/activate
3. pip install -r requirements.txt
4. uvicorn app:app --reload

Notes:
- This is an incremental scaffold. Next steps: implement Google OAuth, Razorpay wiring, admin flows, and ensure templates use `request.url_for`.

## Deploying to Render (recommended)

Follow these steps to deploy this app to Render with Managed Postgres:

1. Create a **Web Service** on Render and connect the repository.
2. In **Service settings** use:
   - **Build command**: pip install -r requirements.txt
   - **Start command**: gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --workers 2
3. Provision a **Managed Postgres** instance on Render and copy its connection URL into the Web Service environment variable `DATABASE_URL`.
4. Add the required environment variables under the Web Service settings (secrets):
   - SECRET_KEY
   - RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET
   - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, FROM_EMAIL
   - GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI
5. Configure the **health check** to `GET /health` (covered by `app.py`).
6. Run one-off DB init (once):
   - In Render console or via an SSH/one-off command: `python -c "from app import create_db_and_tables; create_db_and_tables()"`
7. Configure Razorpay webhook to: `https://<your-app>.onrender.com/webhook/razorpay` and set the `RAZORPAY_WEBHOOK_SECRET` accordingly.

### Optional: Docker
- A `Dockerfile` is included if you prefer deploying the app as a container on Render.

### CI
- GitHub Actions workflow `.github/workflows/ci.yml` runs tests on push to `main`.

If you'd like, I can also add a Render deployment GitHub Action to trigger deploys automatically (requires a Render API key).
