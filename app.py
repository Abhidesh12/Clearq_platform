import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
from fastapi import Query
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
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
import shutil


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

# Add session middleware for flash messages


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
    
class EditProfileForm(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None
    skills: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    twitter_url: Optional[str] = None
    website_url: Optional[str] = None
    experience_years: Optional[int] = None
    industry: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    
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
# Add this function after your database models
def create_admin_user(db: Session):
    """Create an admin user if not exists"""
    admin_email = "admin@clearq.com"
    admin_user = db.query(User).filter(User.email == admin_email).first()
    
    if not admin_user:
        hashed_password = pwd_context.hash("admin123")  # Change this password
        admin_user = User(
            email=admin_email,
            username="admin",
            password_hash=hashed_password,
            full_name="Admin User",
            role="admin",
            is_verified=True,
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        print("Admin user created: admin@clearq.com / admin123")
    
    return admin_user

# Add this at the end of your imports/before routes
@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        create_admin_user(db)
    finally:
        db.close()
        
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
    
    if user and user.is_active:
        # Ensure profile_image has correct path
        if user.profile_image and not user.profile_image.startswith("uploads/"):
            user.profile_image = f"uploads/{user.profile_image}" if user.profile_image != "default-avatar.png" else "default-avatar.png"
    
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

def save_profile_image(file: UploadFile, user_id: int) -> str:
    """Save uploaded profile image and return filename"""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Create uploads/profile_images directory if it doesn't exist
    profile_images_dir = UPLOAD_DIR / "profile_images"
    profile_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"profile_{user_id}_{uuid.uuid4().hex[:8]}.{ext}"
    file_path = profile_images_dir / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Return relative path for database storage
    return f"uploads/profile_images/{filename}"

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
    
@app.get("/static/{path:path}")
async def serve_static(path: str):
    """Serve static files including uploaded images"""
    static_file = Path("static") / path
    if not static_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(static_file)
    
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
            ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
            
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
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    success: Optional[str] = None,
    error: Optional[str] = None
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    # Get mentor profile if user is mentor
    mentor_profile = None
    if current_user.role == "mentor":
        mentor_profile = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    
    # Get flash messages from query parameters
    flash_messages = []
    if success:
        flash_messages.append({
            "category": "success",
            "message": success
        })
    if error:
        flash_messages.append({
            "category": "error", 
            "message": error
        })
    
    # Also check for flash messages in response cookies (alternative approach)
    flash_msg = request.cookies.get("flash_message")
    flash_cat = request.cookies.get("flash_category")
    if flash_msg:
        flash_messages.append({
            "category": flash_cat or "info",
            "message": flash_msg
        })
    
    return templates.TemplateResponse("edit_profile.html", {
        "request": request,
        "current_user": current_user,
        "mentor_profile": mentor_profile,
        "now": datetime.now(),
        "flash_messages": flash_messages
    })

@app.get("/profile/change-password", response_class=HTMLResponse)
async def change_password_page(
    request: Request,
    current_user = Depends(get_current_user),
    success: Optional[str] = None,
    error: Optional[str] = None
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    flash_messages = []
    if success:
        flash_messages.append({
            "category": "success",
            "message": success
        })
    if error:
        flash_messages.append({
            "category": "error",
            "message": error
        })
    
    # Also check for flash messages in response cookies
    flash_msg = request.cookies.get("flash_message")
    flash_cat = request.cookies.get("flash_category")
    if flash_msg:
        flash_messages.append({
            "category": flash_cat or "info",
            "message": flash_msg
        })
    
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "current_user": current_user,
        "flash_messages": flash_messages
    })

@app.post("/profile/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Verify current password
    user = db.query(User).filter(User.id == current_user.id).first()
    if not pwd_context.verify(current_password, user.password_hash):
        # Redirect with error in query parameter
        return RedirectResponse(
            url="/profile/change-password?error=Current%20password%20is%20incorrect",
            status_code=303
        )
    
    # Check if new passwords match
    if new_password != confirm_password:
        # Redirect with error in query parameter
        return RedirectResponse(
            url="/profile/change-password?error=New%20passwords%20do%20not%20match",
            status_code=303
        )
    
    # Update password
    user.password_hash = pwd_context.hash(new_password)
    db.commit()
    
    # Redirect with success in query parameter
    return RedirectResponse(
        url="/profile/change-password?success=Password%20changed%20successfully!",
        status_code=303
    )

@app.post("/profile/edit")
async def update_profile(
    request: Request,
    full_name: str = Form(None),
    bio: str = Form(None),
    skills: str = Form(None),
    linkedin_url: str = Form(None),
    github_url: str = Form(None),
    twitter_url: str = Form(None),
    website_url: str = Form(None),
    experience_years: int = Form(None),
    industry: str = Form(None),
    job_title: str = Form(None),
    company: str = Form(None),
    profile_photo: UploadFile = File(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Update user info
        user = db.query(User).filter(User.id == current_user.id).first()
        
        if full_name:
            user.full_name = full_name
        
        # Handle profile photo upload
        if profile_photo and profile_photo.filename:
            # Delete old profile image if exists and not default
            if user.profile_image and user.profile_image != "default-avatar.png":
                old_image_path = UPLOAD_DIR / user.profile_image
                if old_image_path.exists():
                    old_image_path.unlink()
            
            # Save new profile image
            filename = save_profile_image(profile_photo, current_user.id)
            user.profile_image = filename
        
        # Update mentor profile if exists
        if current_user.role == "mentor":
            mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
            
            if not mentor:
                # Create mentor profile if doesn't exist
                mentor = Mentor(
                    user_id=current_user.id,
                    bio=bio if bio else "",
                    skills=skills if skills else "",
                    linkedin_url=linkedin_url,
                    github_url=github_url,
                    twitter_url=twitter_url,
                    website_url=website_url,
                    experience_years=experience_years if experience_years else 0,
                    industry=industry if industry else "",
                    job_title=job_title if job_title else "",
                    company=company if company else "",
                    created_at=datetime.utcnow()
                )
                db.add(mentor)
            else:
                # Update existing mentor profile
                if bio is not None:
                    mentor.bio = bio
                if skills is not None:
                    mentor.skills = skills
                if linkedin_url is not None:
                    mentor.linkedin_url = linkedin_url
                if github_url is not None:
                    mentor.github_url = github_url
                if twitter_url is not None:
                    mentor.twitter_url = twitter_url
                if website_url is not None:
                    mentor.website_url = website_url
                if experience_years is not None:
                    mentor.experience_years = experience_years
                if industry is not None:
                    mentor.industry = industry
                if job_title is not None:
                    mentor.job_title = job_title
                if company is not None:
                    mentor.company = company
        
        db.commit()
        
        # Redirect with success in query parameter
        response = RedirectResponse(
            url="/profile/edit?success=Profile%20updated%20successfully!",
            status_code=303
        )
        
        return response
        
    except Exception as e:
        db.rollback()
        # Redirect with error in query parameter
        error_message = f"Error updating profile: {str(e)}"
        encoded_error = error_message.replace(" ", "%20").replace(":", "%3A")
        return RedirectResponse(
            url=f"/profile/edit?error={encoded_error}",
            status_code=303
        )

@app.post("/profile/upload-photo")
async def upload_profile_photo_api(
    request: Request,
    profile_photo: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Not authenticated"}
        )
    
    try:
        if not profile_photo.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file uploaded"}
            )
        
        # Validate file type
        if not allowed_file(profile_photo.filename):
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Invalid file type. Allowed: png, jpg, jpeg, gif"}
            )
        
        # Check file size (max 5MB)
        content = await profile_photo.read()
        file_size = len(content)
        if file_size > 5 * 1024 * 1024:  # 5MB
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "File size must be less than 5MB"}
            )
        
        # Reset file pointer
        profile_photo.file.seek(0)
        
        # Delete old profile image if exists and not default
        user = db.query(User).filter(User.id == current_user.id).first()
        if user.profile_image and user.profile_image != "default-avatar.png":
            old_image_path = UPLOAD_DIR / user.profile_image
            if old_image_path.exists():
                old_image_path.unlink()
        
        # Save new profile image
        filename = save_profile_image(profile_photo, current_user.id)
        user.profile_image = filename
        db.commit()
        
        return JSONResponse({
            "success": True, 
            "filename": filename,
            "message": "Profile photo updated successfully!",
            "redirect_url": "/profile/edit?success=Profile%20photo%20updated%20successfully!"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Failed to upload photo: {str(e)}"}
        )

@app.post("/profile/remove-photo")
async def remove_profile_photo(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Not authenticated"}
        )
    
    try:
        user = db.query(User).filter(User.id == current_user.id).first()
        
        if user.profile_image and user.profile_image != "default-avatar.png":
            # Delete the file
            image_path = UPLOAD_DIR / user.profile_image
            if image_path.exists():
                image_path.unlink()
            
            # Reset to default avatar
            user.profile_image = "default-avatar.png"
            db.commit()
            
            return JSONResponse({
                "success": True, 
                "message": "Profile photo removed successfully!",
                "redirect_url": "/profile/edit?success=Profile%20photo%20removed%20successfully!"
            })
        else:
            return JSONResponse({
                "success": False, 
                "message": "No profile photo to remove"
            })
            
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Failed to remove photo: {str(e)}"}
        )
# ============ MENTOR ROUTES ============
@app.get("/payment/{booking_id}", response_class=HTMLResponse)
async def payment_page(
    request: Request,
    booking_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Payment page for a booking"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    booking = db.query(Booking).filter(
        Booking.id == booking_id,
        Booking.learner_id == current_user.id
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    if booking.payment_status == "paid":
        return RedirectResponse(url="/dashboard", status_code=303)
    
    service = db.query(Service).filter(Service.id == booking.service_id).first()
    mentor = db.query(Mentor).filter(Mentor.id == booking.mentor_id).first()
    
    return templates.TemplateResponse("payment.html", {
        "request": request,
        "current_user": current_user,
        "booking": booking,
        "service": service,
        "mentor": mentor,
        "razorpay_key_id": RAZORPAY_KEY_ID
    })
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
        
        # Return booking ID for redirection
        return JSONResponse({
            "success": True,
            "booking_id": booking.id,
            "redirect_url": f"/payment/{booking.id}"
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
