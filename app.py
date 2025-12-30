import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import razorpay
from dotenv import load_dotenv
import json
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="ClearQ Mentorship Platform")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/clearq")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Templates and static files
async def add_now_to_context(request: Request):
    return {"now": datetime.now()}
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.now 

from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Razorpay configuration
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# File upload configuration
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# ============ DATABASE MODELS ============

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    full_name = Column(String)
    profile_image = Column(String, default="default-avatar.png")
    role = Column(String, default="learner")  # learner, mentor, admin
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    google_id = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    mentor_profile = relationship("Mentor", back_populates="user", uselist=False)
    bookings_as_learner = relationship("Booking", foreign_keys="[Booking.learner_id]", back_populates="learner")
    #bookings_as_mentor = relationship("Booking", foreign_keys="[Booking.mentor_id]", back_populates="mentor")
    reviews = relationship("Review", back_populates="learner")

class Mentor(Base):
    __tablename__ = "mentors"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    experience_years = Column(Integer)
    industry = Column(String)
    job_title = Column(String)
    company = Column(String)
    bio = Column(Text)
    skills = Column(Text)  # Comma-separated skills
    linkedin_url = Column(String)
    github_url = Column(String)
    twitter_url = Column(String)
    website_url = Column(String)
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)
    is_verified_by_admin = Column(Boolean, default=False)
    verification_status = Column(String, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bookings_as_mentor = relationship("Booking", foreign_keys="[Booking.mentor_id]", back_populates="mentor")
    user = relationship("User", back_populates="mentor_profile")
    services = relationship("Service", back_populates="mentor")
    availabilities = relationship("Availability", back_populates="mentor")

class Service(Base):
    __tablename__ = "services"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    name = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)  # mock-interview, resume-review, career-guidance, etc.
    price = Column(Integer, nullable=False)  # in INR
    duration_minutes = Column(Integer, default=60)
    is_digital = Column(Boolean, default=False)
    digital_product_url = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    mentor = relationship("Mentor", back_populates="services")
    bookings = relationship("Booking", back_populates="service")

class Availability(Base):
    __tablename__ = "availabilities"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"), nullable=True)
    date = Column(DateTime, nullable=False)
    start_time = Column(String, nullable=False)  # "14:00"
    end_time = Column(String, nullable=False)    # "15:00"
    is_booked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    mentor = relationship("Mentor", back_populates="availabilities")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(Integer, ForeignKey("users.id"))
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"))
    booking_date = Column(DateTime, nullable=False)
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, confirmed, completed, cancelled
    payment_status = Column(String, default="pending")  # pending, paid, failed, refunded
    razorpay_order_id = Column(String)
    razorpay_payment_id = Column(String)
    amount_paid = Column(Integer)
    meeting_link = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    learner = relationship("User", foreign_keys=[learner_id], back_populates="bookings_as_learner")
    mentor = relationship("Mentor", foreign_keys=[mentor_id], back_populates="bookings_as_mentor")
    service = relationship("Service", back_populates="bookings")
    review = relationship("Review", back_populates="booking", uselist=False)

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"), unique=True)
    learner_id = Column(Integer, ForeignKey("users.id"))
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="review")
    learner = relationship("User", back_populates="reviews")

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"))
    razorpay_order_id = Column(String)
    razorpay_payment_id = Column(String)
    amount = Column(Integer)
    status = Column(String)  # created, attempted, paid, failed
    payment_method = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ============ PYDANTIC SCHEMAS ============

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str
    role: str = "learner"

class UserLogin(BaseModel):
    email: str
    password: str

class MentorProfileCreate(BaseModel):
    experience_years: int
    industry: str
    job_title: str
    company: str
    bio: str
    skills: str
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None

class ServiceCreate(BaseModel):
    name: str
    description: str
    category: str
    price: int
    duration_minutes: int = 60
    is_digital: bool = False
    digital_product_url: Optional[str] = None

class AvailabilityCreate(BaseModel):
    date: str
    start_time: str
    end_time: str
    service_id: Optional[int] = None

class BookingCreate(BaseModel):
    service_id: int
    date: str
    time_slot: str

# ============ DEPENDENCIES ============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    
    user = db.query(User).filter(User.id == user_id).first()
    return user if user and user.is_active else None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ============ HELPER FUNCTIONS ============

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_profile_image(file, user_id: int) -> str:
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"profile_{user_id}_{uuid.uuid4().hex[:8]}.{ext}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    
    return f"uploads/{filename}"

# ============ ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, current_user = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_user": current_user,
        "now": datetime.now()
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, current_user = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request})

from sqlalchemy import text

@app.post("/register")
async def register_user(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    role: str = Form("learner"),
    db: Session = Depends(get_db)
):
    # Check if user exists using RAW SQL
    result = db.execute(
        text("SELECT * FROM users WHERE email = :email OR username = :username"),
        {"email": email, "username": username}
    ).fetchone()
    
    if result:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    
    # Create user using RAW SQL
    hashed_password = pwd_context.hash(password)
    is_verified = role != "mentor"
    
    # Insert user directly
    db.execute(
        text("""
            INSERT INTO users (email, username, password_hash, full_name, role, is_verified, is_active, created_at)
            VALUES (:email, :username, :password_hash, :full_name, :role, :is_verified, :is_active, NOW())
        """),
        {
            "email": email,
            "username": username,
            "password_hash": hashed_password,
            "full_name": full_name,
            "role": role,
            "is_verified": is_verified,
            "is_active": True  
        }
    )
    db.commit()
    
    # Get the newly created user ID
    user_result = db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": email}
    ).fetchone()
    
    user_id = user_result[0] if user_result else None
    
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    # If mentor, create mentor profile using RAW SQL
    if role == "mentor":
        db.execute(
            text("INSERT INTO mentors (user_id, verification_status) VALUES (:user_id, 'pending')"),
            {"user_id": user_id}
        )
        db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user_id)})
    
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, current_user = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=303)
    
    # Generate Google OAuth URL
    google_auth_url = None
    if GOOGLE_CLIENT_ID:
        params = {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent"
        }
        google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "google_auth_url": google_auth_url
    })

@app.post("/login")
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not pwd_context.verify(password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account deactivated")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/auth/google/callback")
async def google_auth_callback(request: Request, code: str, db: Session = Depends(get_db)):
    # Implement Google OAuth callback
    # This would involve exchanging code for token, getting user info
    # For simplicity, we'll redirect to register with Google info
    return RedirectResponse(url="/register")

@app.get("/explore", response_class=HTMLResponse)
async def explore_mentors(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    mentors = db.query(Mentor).filter(
        Mentor.is_verified_by_admin == True
    ).all()
    
    return templates.TemplateResponse("explore.html", {
        "request": request,
        "current_user": current_user,
        "mentors": mentors
    })

@app.get("/mentor/{mentor_id}", response_class=HTMLResponse)
async def mentor_profile(
    request: Request,
    mentor_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    services = db.query(Service).filter(
        Service.mentor_id == mentor_id,
        Service.is_active == True
    ).all()
    
    # Generate sample availability dates (in real app, query from database)
    import random
    available_dates = []
    for i in range(7):
        date = datetime.now() + timedelta(days=i)
        if random.choice([True, False]):  # Simulate availability
            available_dates.append({
                "day_name": date.strftime("%a"),
                "day_num": date.day,
                "month": date.strftime("%b"),
                "full_date": date.strftime("%Y-%m-%d")
            })
    
    return templates.TemplateResponse("mentor_profile.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "services": services,
        "available_dates": available_dates
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    context = {"request": request, "current_user": current_user}
    
    if current_user.role == "learner":
        bookings = db.query(Booking).filter(
            Booking.learner_id == current_user.id
        ).order_by(Booking.booking_date.desc()).limit(10).all()
        context["bookings"] = bookings
    
    elif current_user.role == "mentor":
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if mentor:
            bookings = db.query(Booking).filter(
                Booking.mentor_id == mentor.id
            ).order_by(Booking.booking_date.desc()).limit(10).all()
            earnings = db.query(Booking).filter(
                Booking.mentor_id == mentor.id,
                Booking.payment_status == "paid"
            ).with_entities(db.func.sum(Booking.amount_paid)).scalar() or 0
            
            context.update({
                "mentor": mentor,
                "bookings": bookings,
                "earnings": earnings
            })
    
    elif current_user.role == "admin":
        pending_mentors = db.query(Mentor).filter(
            Mentor.verification_status == "pending"
        ).all()
        recent_bookings = db.query(Booking).order_by(
            Booking.created_at.desc()
        ).limit(20).all()
        
        context.update({
            "pending_mentors": pending_mentors,
            "recent_bookings": recent_bookings
        })
    
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/profile/edit", response_class=HTMLResponse)
async def edit_profile_page(
    request: Request,
    current_user = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    return templates.TemplateResponse("edit_profile.html", {
        "request": request,
        "current_user": current_user
    })

@app.post("/profile/edit")
async def update_profile(
    request: Request,
    full_name: str = Form(None),
    bio: str = Form(None),
    skills: str = Form(None),
    linkedin_url: str = Form(None),
    github_url: str = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.id == current_user.id).first()
    if full_name:
        user.full_name = full_name
    
    # Update mentor profile if exists
    if current_user.role == "mentor":
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if mentor:
            if bio: mentor.bio = bio
            if skills: mentor.skills = skills
            if linkedin_url: mentor.linkedin_url = linkedin_url
            if github_url: mentor.github_url = github_url
    
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/profile/upload-photo")
async def upload_profile_photo(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    form = await request.form()
    file = form.get("profile_photo")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        filename = save_profile_image(file, current_user.id)
        user = db.query(User).filter(User.id == current_user.id).first()
        user.profile_image = filename
        db.commit()
        
        return JSONResponse({"success": True, "filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ MENTOR ROUTES ============

@app.get("/mentor/dashboard/services", response_class=HTMLResponse)
async def mentor_services(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    services = db.query(Service).filter(Service.mentor_id == mentor.id).all()
    
    return templates.TemplateResponse("mentor_services.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "services": services
    })

@app.post("/mentor/services/create")
async def create_service(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    price: int = Form(...),
    duration_minutes: int = Form(60),
    is_digital: bool = Form(False),
    digital_product_url: str = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    service = Service(
        mentor_id=mentor.id,
        name=name,
        description=description,
        category=category,
        price=price,
        duration_minutes=duration_minutes,
        is_digital=is_digital,
        digital_product_url=digital_product_url
    )
    
    db.add(service)
    db.commit()
    
    return RedirectResponse(url="/mentor/dashboard/services", status_code=303)

@app.get("/mentor/availability", response_class=HTMLResponse)
async def mentor_availability(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    availabilities = db.query(Availability).filter(
        Availability.mentor_id == mentor.id,
        Availability.date >= datetime.now().date()
    ).order_by(Availability.date, Availability.start_time).all()
    
    services = db.query(Service).filter(Service.mentor_id == mentor.id).all()
    
    return templates.TemplateResponse("mentor_availability.html", {
        "request": request,
        "current_user": current_user,
        "availabilities": availabilities,
        "services": services
    })

@app.post("/mentor/availability/create")
async def create_availability(
    request: Request,
    date: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    service_id: int = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    
    # Check if time slot is available
    existing = db.query(Availability).filter(
        Availability.mentor_id == mentor.id,
        Availability.date == datetime.strptime(date, "%Y-%m-%d").date(),
        Availability.start_time == start_time
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Time slot already exists")
    
    availability = Availability(
        mentor_id=mentor.id,
        service_id=service_id,
        date=datetime.strptime(date, "%Y-%m-%d").date(),
        start_time=start_time,
        end_time=end_time
    )
    
    db.add(availability)
    db.commit()
    
    return RedirectResponse(url="/mentor/availability", status_code=303)

# ============ BOOKING & PAYMENT ROUTES ============

@app.post("/api/create-booking")
async def create_booking(
    request: Request,
    booking_data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "learner":
        raise HTTPException(status_code=403, detail="Only learners can book sessions")
    
    service_id = booking_data.get("service_id")
    date_str = booking_data.get("date")
    time_slot = booking_data.get("time")
    
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Create Razorpay order
    order_amount = service.price * 100  # Convert to paise
    order_currency = "INR"
    
    try:
        order_data = {
            "amount": order_amount,
            "currency": order_currency,
            "payment_capture": 1,
            "notes": {
                "service_id": service_id,
                "learner_id": current_user.id,
                "date": date_str,
                "time": time_slot
            }
        }
        
        razorpay_order = razorpay_client.order.create(order_data)
        
        # Create booking record
        booking = Booking(
            learner_id=current_user.id,
            mentor_id=service.mentor_id,
            service_id=service_id,
            booking_date=datetime.strptime(date_str, "%Y-%m-%d"),
            start_time=time_slot,
            end_time=(datetime.strptime(time_slot, "%H:%M") + timedelta(minutes=service.duration_minutes)).strftime("%H:%M"),
            razorpay_order_id=razorpay_order["id"],
            amount_paid=service.price
        )
        
        db.add(booking)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "booking_id": booking.id,
            "razorpay_order_id": razorpay_order["id"],
            "amount": service.price,
            "currency": order_currency,
            "key_id": RAZORPAY_KEY_ID
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payment/verify")
async def verify_payment(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    data = await request.json()
    
    try:
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': data['razorpay_order_id'],
            'razorpay_payment_id': data['razorpay_payment_id'],
            'razorpay_signature': data['razorpay_signature']
        })
        
        # Update booking status
        booking = db.query(Booking).filter(
            Booking.razorpay_order_id == data['razorpay_order_id']
        ).first()
        
        if booking:
            booking.payment_status = "paid"
            booking.razorpay_payment_id = data['razorpay_payment_id']
            booking.status = "confirmed"
            db.commit()
        
        return JSONResponse({"success": True})
    except:
        raise HTTPException(status_code=400, detail="Payment verification failed")

# ============ ADMIN ROUTES ============

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    pending_mentors = db.query(Mentor).filter(
        Mentor.verification_status == "pending"
    ).all()
    
    total_users = db.query(User).count()
    total_mentors = db.query(Mentor).count()
    total_bookings = db.query(Booking).count()
    revenue = db.query(Booking).filter(
        Booking.payment_status == "paid"
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "current_user": current_user,
        "pending_mentors": pending_mentors,
        "stats": {
            "total_users": total_users,
            "total_mentors": total_mentors,
            "total_bookings": total_bookings,
            "revenue": revenue
        }
    })

@app.post("/admin/mentor/{mentor_id}/verify")
async def verify_mentor(
    mentor_id: int,
    action: str = Form(...),  # approve or reject
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    if action == "approve":
        mentor.is_verified_by_admin = True
        mentor.verification_status = "approved"
        # Activate user account
        user = db.query(User).filter(User.id == mentor.user_id).first()
        if user:
            user.is_verified = True
    elif action == "reject":
        mentor.verification_status = "rejected"
    
    db.commit()
    return RedirectResponse(url="/admin/dashboard", status_code=303)

# ============ API ENDPOINTS ============

@app.post("/api/time-slots/{mentor_id}")
async def get_time_slots(
    mentor_id: int,
    data: dict,
    db: Session = Depends(get_db)
):
    date_str = data.get("date")
    if not date_str:
        return JSONResponse({"success": False, "message": "Date required"})
    
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    # Get booked time slots
    booked_slots = db.query(Availability).filter(
        Availability.mentor_id == mentor_id,
        Availability.date == target_date,
        Availability.is_booked == True
    ).all()
    
    # Get all available time slots
    available_slots = db.query(Availability).filter(
        Availability.mentor_id == mentor_id,
        Availability.date == target_date,
        Availability.is_booked == False
    ).all()
    
    booked_times = [slot.start_time for slot in booked_slots]
    available_times = [slot.start_time for slot in available_slots]
    
    # Generate time slots (simplified - in real app, use mentor's availability)
    all_slots = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00", "17:00"]
    free_slots = [slot for slot in all_slots if slot not in booked_times]
    
    return JSONResponse({
        "success": True,
        "slots": free_slots[:4]  # Return first 4 available slots
    })

# ============ STATIC PAGES ============

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def terms_of_service(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/mentorship-program", response_class=HTMLResponse)
async def mentorship_program(request: Request, current_user = Depends(get_current_user)):
    return templates.TemplateResponse("mentorship_program.html", {
        "request": request,
        "current_user": current_user
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("access_token")
    return response

# ============ ERROR HANDLERS ============

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
