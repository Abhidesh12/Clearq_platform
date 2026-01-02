import os
import re
from sqlalchemy import or_
from sqlalchemy.pool import QueuePool
import uuid
from datetime import datetime, timedelta, date
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
from sqlalchemy import func, Date
from starlette.middleware.base import BaseHTTPMiddleware



# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="ClearQ Mentorship Platform")

# Updated middleware
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://checkout.razorpay.com 'unsafe-inline'; "
            "style-src 'self' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com/ajax/libs/font-awesome/ 'unsafe-inline'; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "connect-src 'self' https://api.razorpay.com; "
            "img-src 'self' data: https:; "
            "frame-src 'self' https://api.razorpay.com;"
        )
        return response

# Add to your app
app.add_middleware(CSPMiddleware)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/clearq")
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections every 30 minutes
)
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
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

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
    bookings_as_learner = relationship("Booking", foreign_keys="Booking.learner_id", back_populates="learner")
    #bookings_as_mentor = relationship("Booking", foreign_keys="Booking.mentor_id", back_populates="mentor")
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
    date = Column(Date, nullable=False)  # Changed from DateTime to Date
    start_time = Column(String, default="09:00")  # Default start time
    end_time = Column(String, default="24:00")
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"), nullable=True)   
    is_booked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    mentor = relationship("Mentor", back_populates="availabilities")

class TimeSlot(Base):
    __tablename__ = "time_slots"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"))
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="time_slots")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(Integer, ForeignKey("users.id"))
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"))
    booking_date = Column(Date, nullable=False)
    selected_time = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, confirmed, completed, cancelled
    payment_status = Column(String, default="pending")  # pending, paid, failed, refunded
    razorpay_order_id = Column(String)
    razorpay_payment_id = Column(String)
    amount_paid = Column(Integer)
    meeting_link = Column(String)
    meeting_id = Column(String)
    meeting_password = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    learner = relationship("User", foreign_keys=[learner_id], back_populates="bookings_as_learner")
    mentor = relationship("Mentor", foreign_keys=[mentor_id], back_populates="bookings_as_mentor")
    service = relationship("Service", back_populates="bookings")
    review = relationship("Review", back_populates="booking", uselist=False)
    time_slots = relationship("TimeSlot", back_populates="booking")

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



def cleanup_past_availabilities(db: Session):
    """Remove availability slots from past dates"""
    today = datetime.now().date()
    try:
        # Delete availabilities from past dates
        db.query(Availability).filter(
            Availability.date < today
        ).delete(synchronize_session=False)
        db.commit()
        print(f"Cleaned up past availabilities")
    except Exception as e:
        db.rollback()
        print(f"Error cleaning up past availabilities: {e}")


        
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
        
def load_availabilities_with_retry(mentor_id, db, retries=3):
    for i in range(retries):
        try:
            return db.query(Availability).filter(
                Availability.mentor_id == mentor_id,
                Availability.date >= datetime.now().date()
            ).order_by(Availability.date).all()
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(1)
            
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
        return user
    
    return None
        # Ensure profile_image has correct path
       # if user.profile_image and not user.profile_image.startswith("uploads/"):
            #user.profile_image = f"uploads/{user.profile_image}" if user.profile_image != "default-avatar.png" else "default-avatar.png"
    
    #return user if user and user.is_active else None

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
    # Check if user exists using ORM
    existing_user = db.query(User).filter(or_(User.email == email, User.username == username)).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    
    # Create user object
    hashed_password = pwd_context.hash(password)
    is_verified = role != "mentor"
    
    new_user = User(
        email=email,
        username=username,
        password_hash=hashed_password,
        full_name=full_name,
        role=role,
        is_verified=is_verified,
        is_active=True
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user) # get the ID
    
    # Create mentor profile if needed
    if role == "mentor":
        mentor = Mentor(user_id=new_user.id, verification_status='pending')
        db.add(mentor)
        db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": str(new_user.id)})
    
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
    db: Session = Depends(get_db),
    q: Optional[str] = None,  # Search query
    industry: Optional[str] = None,
    experience: Optional[str] = None,
    sort: Optional[str] = "recommended",
    page: int = 1,
    limit: int = 9
):
    # Build base query for verified mentors
    query = db.query(Mentor).join(User).filter(
        Mentor.is_verified_by_admin == True,
        User.is_active == True
    )
    
    # Apply search filter if provided
    if q and q.strip():
        search_term = f"%{q.strip()}%"
        query = query.filter(
            or_(
                User.full_name.ilike(search_term),
                User.username.ilike(search_term),
                Mentor.bio.ilike(search_term),
                Mentor.skills.ilike(search_term),
                Mentor.industry.ilike(search_term),
                Mentor.job_title.ilike(search_term),
                Mentor.company.ilike(search_term)
            )
        )
    
    # Apply industry filter if provided
    if industry and industry.strip():
        query = query.filter(Mentor.industry.ilike(f"%{industry.strip()}%"))
    
    # Apply experience filter if provided
    if experience and experience.strip():
        if experience == "1-3":
            query = query.filter(Mentor.experience_years.between(1, 3))
        elif experience == "3-5":
            query = query.filter(Mentor.experience_years.between(3, 5))
        elif experience == "5-10":
            query = query.filter(Mentor.experience_years.between(5, 10))
        elif experience == "10+":
            query = query.filter(Mentor.experience_years >= 10)
    
    # Apply sorting
    if sort == "rating":
        query = query.order_by(Mentor.rating.desc().nulls_last(), Mentor.review_count.desc())
    elif sort == "price_low":
        # Get mentors with their minimum service price
        from sqlalchemy import func
        subquery = db.query(
            Service.mentor_id,
            func.min(Service.price).label('min_price')
        ).filter(Service.is_active == True).group_by(Service.mentor_id).subquery()
        
        query = query.outerjoin(
            subquery, Mentor.id == subquery.c.mentor_id
        ).order_by(subquery.c.min_price.asc().nulls_last())
    elif sort == "price_high":
        # Get mentors with their maximum service price
        from sqlalchemy import func
        subquery = db.query(
            Service.mentor_id,
            func.max(Service.price).label('max_price')
        ).filter(Service.is_active == True).group_by(Service.mentor_id).subquery()
        
        query = query.outerjoin(
            subquery, Mentor.id == subquery.c.mentor_id
        ).order_by(subquery.c.max_price.desc().nulls_last())
    elif sort == "experience":
        query = query.order_by(Mentor.experience_years.desc().nulls_last())
    elif sort == "name":
        query = query.order_by(User.full_name.asc().nulls_last())
    else:  # recommended/default
        query = query.order_by(
            Mentor.rating.desc().nulls_last(), 
            Mentor.review_count.desc(),
            Mentor.experience_years.desc()
        )
    
    # Get total count for pagination
    total_mentors = query.count()
    total_pages = (total_mentors + limit - 1) // limit if limit > 0 else 1
    
    # Apply pagination
    offset = (page - 1) * limit
    mentors = query.offset(offset).limit(limit).all()
    
    # Load services for each mentor to get pricing info
    for mentor in mentors:
        mentor.services = db.query(Service).filter(
            Service.mentor_id == mentor.id,
            Service.is_active == True
        ).order_by(Service.price.asc()).all()
        
        # Also ensure user data is loaded
        if not mentor.user:
            mentor.user = db.query(User).filter(User.id == mentor.user_id).first()
    
    return templates.TemplateResponse("explore.html", {
        "request": request,
        "current_user": current_user,
        "mentors": mentors,
        "search_query": q,
        "selected_industry": industry,
        "selected_experience": experience,
        "selected_sort": sort,
        "page": page,
        "total_pages": total_pages,
        "total_mentors": total_mentors
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
        # Show ALL bookings for the learner, but separate by status
        all_bookings = db.query(Booking).filter(
            Booking.learner_id == current_user.id
        ).order_by(Booking.booking_date.desc()).limit(15).all()
        
        # Separate bookings by status
        confirmed_bookings = []
        pending_bookings = []
        upcoming_sessions = []
        
        today = datetime.now().date()
        
        for booking in all_bookings:
            if booking.payment_status == "paid" and booking.status == "confirmed" and booking.meeting_link:
                # This is a confirmed, paid booking with meeting link
                if booking.booking_date >= today:  # Only upcoming sessions
                    upcoming_sessions.append(booking)
                confirmed_bookings.append(booking)
            elif booking.payment_status == "pending" or booking.status == "pending":
                # This is a pending booking (awaiting payment)
                pending_bookings.append(booking)
            else:
                # Other statuses (cancelled, completed, etc.)
                confirmed_bookings.append(booking)
        
        # Get stats for the learner
        total_sessions = len([b for b in all_bookings if b.payment_status == "paid"])
        completed_sessions = len([b for b in all_bookings if b.status == "completed"])
        pending_payments = len(pending_bookings)
        
        context.update({
            "all_bookings": all_bookings,
            "upcoming_sessions": upcoming_sessions,  # Only confirmed, upcoming sessions with meeting links
            "confirmed_bookings": confirmed_bookings,
            "pending_bookings": pending_bookings,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "pending_payments": pending_payments,
            "stats": {
                "total": total_sessions,
                "completed": completed_sessions,
                "pending": pending_payments
            }
        })
    
    elif current_user.role == "mentor":
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if mentor:
            # Show ALL bookings for the mentor, but separate by status
            all_bookings = db.query(Booking).filter(
                Booking.mentor_id == mentor.id
            ).order_by(Booking.booking_date.desc()).limit(15).all()
            
            # Separate bookings by status
            confirmed_bookings = []
            pending_bookings = []
            upcoming_sessions = []
            
            today = datetime.now().date()
            
            for booking in all_bookings:
                if booking.payment_status == "paid" and booking.status == "confirmed" and booking.meeting_link:
                    # This is a confirmed, paid booking with meeting link
                    if booking.booking_date >= today:  # Only upcoming sessions
                        upcoming_sessions.append(booking)
                    confirmed_bookings.append(booking)
                elif booking.payment_status == "pending" or booking.status == "pending":
                    # This is a pending booking (awaiting payment)
                    pending_bookings.append(booking)
                else:
                    # Other statuses (cancelled, completed, etc.)
                    confirmed_bookings.append(booking)
            
            # Calculate earnings (only from paid bookings)
            earnings = db.query(Booking).filter(
                Booking.mentor_id == mentor.id,
                Booking.payment_status == "paid"
            ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
            
            # Get stats for the mentor
            total_bookings = len(all_bookings)
            pending_payments = len(pending_bookings)
            upcoming_count = len(upcoming_sessions)
            
            context.update({
                "mentor": mentor,
                "all_bookings": all_bookings,
                "upcoming_sessions": upcoming_sessions,  # Only confirmed, upcoming sessions with meeting links
                "confirmed_bookings": confirmed_bookings,
                "pending_bookings": pending_bookings,
                "earnings": earnings,
                "total_bookings": total_bookings,
                "pending_payments": pending_payments,
                "upcoming_count": upcoming_count,
                "stats": {
                    "total": total_bookings,
                    "pending": pending_payments,
                    "upcoming": upcoming_count,
                    "earnings": earnings
                }
            })
    
    elif current_user.role == "admin":
        pending_mentors = db.query(Mentor).filter(
            Mentor.verification_status == "pending"
        ).all()
        
        # Get all recent bookings for admin
        recent_bookings = db.query(Booking).order_by(
            Booking.created_at.desc()
        ).limit(20).all()
        
        # Calculate admin stats
        total_users = db.query(User).count()
        total_mentors = db.query(Mentor).filter(
            Mentor.is_verified_by_admin == True
        ).count()
        total_bookings = db.query(Booking).count()
        revenue = db.query(Booking).filter(
            Booking.payment_status == "paid"
        ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
        
        # Get pending bookings count
        pending_bookings_count = db.query(Booking).filter(
            Booking.payment_status == "pending"
        ).count()
        
        context.update({
            "pending_mentors": pending_mentors,
            "recent_bookings": recent_bookings,
            "stats": {
                "total_users": total_users,
                "total_mentors": total_mentors,
                "total_bookings": total_bookings,
                "revenue": revenue,
                "pending_bookings": pending_bookings_count
            }
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
import razorpay
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session


router = APIRouter()

# Initialize Razorpay


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
    
    # Debug print
    print(f"Payment page - Booking ID: {booking.id}, Razorpay Order ID: {booking.razorpay_order_id}")
    
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
async def mentor_availability_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mentor availability management page"""
    # Get current user from cookie manually since get_current_user might be failing
    token = request.cookies.get("access_token")
    current_user = None
    
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: int = payload.get("sub")
            if user_id:
                current_user = db.query(User).filter(User.id == user_id).first()
        except JWTError:
            pass
    
    # Check if user is authenticated and is a mentor
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    if current_user.role != "mentor":
        # If not mentor, redirect to appropriate dashboard
        if current_user.role == "admin":
            return RedirectResponse(url="/admin/dashboard", status_code=303)
        elif current_user.role == "learner":
            return RedirectResponse(url="/dashboard", status_code=303)
        else:
            return RedirectResponse(url="/dashboard", status_code=303)
    
    # Get or create mentor profile
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        # Create a mentor profile if it doesn't exist
        mentor = Mentor(
            user_id=current_user.id,
            verification_status="pending",
            created_at=datetime.utcnow()
        )
        db.add(mentor)
        db.commit()
        db.refresh(mentor)
    
    # Get today's date
    today = datetime.now().date()
    
    # Get availabilities (current and future dates only)
    try:
        # Try different date filtering approaches
        availabilities = db.query(Availability).filter(
            Availability.mentor_id == mentor.id
        ).order_by(Availability.date, Availability.start_time).all()
        
        # Filter for future dates in Python
        future_availabilities = []
        for avail in availabilities:
            # Extract date safely
            avail_date = None
            if isinstance(avail.date, datetime):
                avail_date = avail.date.date()
            elif isinstance(avail.date, date):
                avail_date = avail.date
            else:
                # Try to convert if it's a string or other type
                try:
                    if hasattr(avail.date, 'date'):
                        avail_date = avail.date.date()
                    elif isinstance(avail.date, str):
                        avail_date = datetime.strptime(avail.date, "%Y-%m-%d").date()
                except:
                    continue
            
            # Only include future dates
            if avail_date and avail_date >= today:
                future_availabilities.append(avail)
        
        # Sort by date and time
        future_availabilities.sort(key=lambda x: (
            x.date if isinstance(x.date, (date, datetime)) else datetime.min,
            x.start_time
        ))
        
    except Exception as e:
        print(f"Error loading availabilities: {e}")
        future_availabilities = []
    
    # Get services for dropdown
    services = db.query(Service).filter(Service.mentor_id == mentor.id).all()
    
    # Get success/error messages from query params
    flash_messages = []
    success = request.query_params.get("success")
    error = request.query_params.get("error")
    
    if success:
        flash_messages.append({"category": "success", "message": success})
    if error:
        flash_messages.append({"category": "error", "message": error})
    
    # Format dates for display
    for avail in future_availabilities:
        try:
            if isinstance(avail.date, datetime):
                avail.display_date = avail.date.strftime("%Y-%m-%d")
                avail.display_day = avail.date.strftime("%A")
                avail.display_time = f"{avail.start_time} - {avail.end_time}"
            elif isinstance(avail.date, date):
                avail.display_date = avail.date.strftime("%Y-%m-%d")
                avail.display_day = avail.date.strftime("%A")
                avail.display_time = f"{avail.start_time} - {avail.end_time}"
        except:
            avail.display_date = "N/A"
            avail.display_day = "N/A"
            avail.display_time = f"{avail.start_time} - {avail.end_time}"
    
    return templates.TemplateResponse("mentor_availability.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "availabilities": future_availabilities,
        "services": services,
        "today": today.strftime("%Y-%m-%d"),
        "flash_messages": flash_messages
    })
@app.post("/mentor/availability/create")
async def create_availability(
    request: Request,
    date: str = Form(...),
    service_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new availability date (default 9am-9pm)"""
    # Get current user from cookie manually
    token = request.cookies.get("access_token")
    current_user = None
    
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: int = payload.get("sub")
            if user_id:
                current_user = db.query(User).filter(User.id == user_id).first()
        except JWTError:
            pass
    
    # Check if user is authenticated and is a mentor
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    try:
        # Get or create mentor profile
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if not mentor:
            mentor = Mentor(
                user_id=current_user.id,
                verification_status="pending",
                created_at=datetime.utcnow()
            )
            db.add(mentor)
            db.commit()
            db.refresh(mentor)
        
        # Parse and validate service_id
        parsed_service_id = None
        if service_id and service_id.strip() and service_id != "None":
            try:
                parsed_service_id = int(service_id)
            except ValueError:
                pass
        
        # Validate date format
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return RedirectResponse(
                url="/mentor/availability?error=Invalid%20date%20format.%20Use%20YYYY-MM-DD",
                status_code=303
            )
        
        # Validate date is not in the past
        today = datetime.now().date()
        if parsed_date < today:
            return RedirectResponse(
                url="/mentor/availability?error=Cannot%20add%20availability%20for%20past%20dates",
                status_code=303
            )
        
        # Check if availability already exists for this date
        existing = db.query(Availability).filter(
            Availability.mentor_id == mentor.id,
            Availability.date == parsed_date
        ).first()
        
        if existing:
            return RedirectResponse(
                url="/mentor/availability?error=Availability%20already%20exists%20for%20this%20date",
                status_code=303
            )
        
        # Validate service_id if provided
        if parsed_service_id:
            service = db.query(Service).filter(
                Service.id == parsed_service_id,
                Service.mentor_id == mentor.id
            ).first()
            if not service:
                return RedirectResponse(
                    url="/mentor/availability?error=Invalid%20service%20selected",
                    status_code=303
                )
        
        # Create new availability with default 9am-9pm
        availability = Availability(
            mentor_id=mentor.id,
            service_id=parsed_service_id,
            date=parsed_date,
            start_time="09:00",  # Default start time
            end_time="21:00",    # Default end time (9pm)
            is_booked=False,
            created_at=datetime.utcnow()
        )
        
        db.add(availability)
        db.commit()
        
        return RedirectResponse(
            url="/mentor/availability?success=Availability%20added%20successfully%20(9:00%20AM%20-%209:00%20PM)",
            status_code=303
        )
        
    except Exception as e:
        db.rollback()
        error_msg = str(e).replace(" ", "%20").replace(":", "%3A")
        return RedirectResponse(
            url=f"/mentor/availability?error={error_msg}",
            status_code=303
        )

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
    time_slot = booking_data.get("time")  # This is the start time selected by learner
    
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Calculate end time based on service duration
    start_time = datetime.strptime(time_slot, "%H:%M")
    end_time = start_time + timedelta(minutes=service.duration_minutes)
    end_time_str = end_time.strftime("%H:%M")
    
    # DO NOT generate Google Meet link yet - will generate after payment
    meeting_id = None
    meeting_link = None
    
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
                "time": time_slot,
                "duration": service.duration_minutes
            }
        }
        
        print(f"Creating Razorpay order with amount: {order_amount}")
        razorpay_order = razorpay_client.order.create(order_data)
        print(f"Razorpay order created: {razorpay_order['id']}")
        
        # Create booking record with selected_time instead of start_time/end_time
        booking = Booking(
            learner_id=current_user.id,
            mentor_id=service.mentor_id,
            service_id=service_id,
            booking_date=datetime.strptime(date_str, "%Y-%m-%d"),
            selected_time=time_slot,  # Store the selected start time
            razorpay_order_id=razorpay_order["id"],
            amount_paid=service.price,
            status="pending",  # Will change to "confirmed" after payment
            payment_status="pending",  # Will change to "paid" after payment
            meeting_link=meeting_link,  # Will be set after payment
            meeting_id=meeting_id,  # Will be set after payment
            notes=f"Session scheduled for {date_str} at {time_slot} (Pending Payment)"
        )
        
        db.add(booking)
        db.commit()
        db.refresh(booking)
        
        print(f"Booking created with ID: {booking.id}, Payment pending")
        
        # Create a TimeSlot record but DO NOT mark availability as booked yet
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Find the specific time slot in availability
        availability = db.query(Availability).filter(
            Availability.mentor_id == service.mentor_id,
            Availability.date == target_date,
            Availability.is_booked == False
        ).first()
        
        if availability:
            # Create a time slot record to reserve this slot temporarily
            time_slot_record = TimeSlot(
                booking_id=booking.id,
                start_time=time_slot,
                end_time=end_time_str,
                date=target_date,
                created_at=datetime.utcnow()
            )
            db.add(time_slot_record)
            db.commit()
            
            # Note: We're NOT marking availability as booked yet
            # This will happen after payment confirmation
        
        return JSONResponse({
            "success": True,
            "booking_id": booking.id,
            "redirect_url": f"/payment/{booking.id}",
            "message": "Booking created. Please complete payment to confirm your session."
        })
        
    except Exception as e:
        print(f"Error creating booking: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Add this new API endpoint for generating time slots
def generate_meeting_link(booking_id: int, db: Session):
    """Generate Google Meet link for a confirmed booking"""
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Generate unique meeting ID
    meeting_id = f"clearq-{uuid.uuid4().hex[:12]}"
    
    # Generate Google Meet link
    # Note: This is a placeholder. In production, you'd use Google Calendar API
    meeting_link = f"https://meet.google.com/new?hs=197&authuser=0"
    
    # Update booking with meeting details
    booking.meeting_link = meeting_link
    booking.meeting_id = meeting_id
    booking.status = "confirmed"
    
    # Also mark the time slot/availability as booked
    target_date = booking.booking_date
    
    # Find and mark the availability as booked
    availability = db.query(Availability).filter(
        Availability.mentor_id == booking.mentor_id,
        Availability.date == target_date
    ).first()
    
    if availability:
        availability.is_booked = True
    
    db.commit()
    
    return meeting_link, meeting_id
    
@app.post("/api/generate-time-slots")
async def generate_time_slots(
    data: dict,
    db: Session = Depends(get_db)
):
    """Generate available time slots based on service duration and date"""
    date_str = data.get("date")
    service_id = data.get("service_id")
    mentor_id = data.get("mentor_id")
    
    if not all([date_str, service_id, mentor_id]):
        return JSONResponse({"success": False, "message": "Missing parameters"})
    
    try:
        # Get service to know duration
        service = db.query(Service).filter(Service.id == service_id).first()
        if not service:
            return JSONResponse({"success": False, "message": "Service not found"})
        
        duration = service.duration_minutes
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Get mentor's availability for this date
        availability = db.query(Availability).filter(
            Availability.mentor_id == mentor_id,
            Availability.date == date,
            Availability.is_booked == False
        ).first()
        
        if not availability:
            return JSONResponse({"success": False, "message": "No availability for this date"})
        
        # Parse default time range (9am to 9pm)
        start_time = datetime.strptime(availability.start_time, "%H:%M")
        end_time = datetime.strptime(availability.end_time, "%H:%M")
        
        # Generate time slots
        slots = []
        current_time = start_time
        slot_duration = timedelta(minutes=duration)
        buffer_duration = timedelta(minutes=15)  # 15 min buffer between sessions
        
        while current_time + slot_duration <= end_time:
            slot_start = current_time.strftime("%H:%M")
            slot_end = (current_time + slot_duration).strftime("%H:%M")
            
            # Check if this slot is already booked
            existing_booking = db.query(Booking).filter(
                Booking.mentor_id == mentor_id,
                Booking.booking_date == date,
                Booking.selected_time == slot_start,
                Booking.status.in_(["pending", "confirmed"])
            ).first()
            
            if not existing_booking:
                # Format for display
                display_start = current_time.strftime("%I:%M %p").lstrip("0")
                display_end = (current_time + slot_duration).strftime("%I:%M %p").lstrip("0")
                
                slots.append({
                    "value": slot_start,
                    "display": f"{display_start} - {display_end}",
                    "end_time": slot_end
                })
            
            # Move to next slot with buffer
            current_time += slot_duration + buffer_duration
        
        if slots:
            return JSONResponse({
                "success": True,
                "slots": slots,
                "duration": duration,
                "availability_hours": f"{availability.start_time} - {availability.end_time}"
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "No available time slots for this date"
            })
            
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})
@app.post("/payment/verify")
async def verify_payment(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    data = await request.json()
    
    try:
        # Verify payment signature
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': data['razorpay_order_id'],
            'razorpay_payment_id': data['razorpay_payment_id'],
            'razorpay_signature': data['razorpay_signature']
        })
        
        # Update booking status
        booking_id = data.get('booking_id')
        booking = db.query(Booking).filter(
            Booking.id == booking_id,
            Booking.learner_id == current_user.id
        ).first()
        
        if booking:
            booking.payment_status = "paid"
            booking.razorpay_payment_id = data['razorpay_payment_id']
            
            # NOW generate meeting link after payment is confirmed
            meeting_link, meeting_id = generate_meeting_link(booking.id, db)
            
            # Create payment record
            payment = Payment(
                booking_id=booking.id,
                razorpay_order_id=data['razorpay_order_id'],
                razorpay_payment_id=data['razorpay_payment_id'],
                amount=booking.amount_paid,
                status="paid",
                payment_method="razorpay",
                created_at=datetime.utcnow()
            )
            db.add(payment)
            db.commit()
            
            # Optionally: Send confirmation email here
            
            return JSONResponse({
                "success": True, 
                "message": "Payment verified successfully! Meeting scheduled.",
                "meeting_link": meeting_link,
                "meeting_id": meeting_id,
                "redirect_url": "/dashboard"
            })
        else:
            return JSONResponse({"success": False, "message": "Booking not found"})
        
    except Exception as e:
        print(f"Payment verification error: {e}")
        db.rollback()
        return JSONResponse({"success": False, "message": "Payment verification failed"})

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

# Update the time-slots API to support both ID and username
@app.post("/api/time-slots/user/{identifier}")
async def get_time_slots(
    identifier: str,
    data: dict,
    db: Session = Depends(get_db)
):
    date_str = data.get("date")
    if not date_str:
        return JSONResponse({"success": False, "message": "Date required"})
    
    try:
        # Check if identifier is a number (ID) or string (username)
        mentor_id = None
        if identifier.isdigit():
            # It's an ID
            mentor_id = int(identifier)
            mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
        else:
            # It's a username
            user = db.query(User).filter(User.username == identifier).first()
            if not user:
                return JSONResponse({"success": False, "message": "Mentor not found"})
            mentor = db.query(Mentor).filter(Mentor.user_id == user.id).first()
            if mentor:
                mentor_id = mentor.id
        
        if not mentor or not mentor_id:
            return JSONResponse({"success": False, "message": "Mentor not found"})
        
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JSONResponse({"success": False, "message": "Invalid date format"})
    
    # Get available time slots for this date
    available_slots = db.query(Availability).filter(
        Availability.mentor_id == mentor_id,
        Availability.date == target_date,
        Availability.is_booked == False
    ).order_by(Availability.start_time).all()
    
    # Format time slots for display
    formatted_slots = []
    for slot in available_slots:
        try:
            start_time_obj = datetime.strptime(slot.start_time, "%H:%M")
            end_time_obj = datetime.strptime(slot.end_time, "%H:%M")
            
            # Format time in 12-hour format with AM/PM
            formatted_start = start_time_obj.strftime("%I:%M %p").lstrip("0")
            formatted_end = end_time_obj.strftime("%I:%M %p").lstrip("0")
            
            formatted_slots.append({
                "value": slot.start_time,
                "display": f"{formatted_start} - {formatted_end}",
                "duration": slot.end_time
            })
        except:
            formatted_slots.append({
                "value": slot.start_time,
                "display": f"{slot.start_time} - {slot.end_time}",
                "duration": slot.end_time
            })
    
    if formatted_slots:
        return JSONResponse({
            "success": True,
            "slots": formatted_slots
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "No available time slots for this date"
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
    
    # Get actual availabilities from database (future dates only)
    today = datetime.now().date()
    availabilities = db.query(Availability).filter(
        Availability.mentor_id == mentor_id,
        Availability.is_booked == False,
        Availability.date >= today
    ).order_by(Availability.date).all()
    
    # Group availabilities by date for the frontend
    from collections import defaultdict
    date_slots = defaultdict(list)
    
    for avail in availabilities:
        if isinstance(avail.date, datetime):
            date_str = avail.date.strftime("%Y-%m-%d")
        elif isinstance(avail.date, date):
            date_str = avail.date.strftime("%Y-%m-%d")
        else:
            try:
                date_str = datetime.strptime(str(avail.date), "%Y-%m-%d").strftime("%Y-%m-%d")
            except:
                continue
        
        date_slots[date_str].append({
            "start_time": avail.start_time,
            "end_time": avail.end_time
        })
    
    # Prepare available dates for display (next 7 days with slots)
    available_dates = []
    for i in range(7):
        target_date = today + timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        
        if date_str in date_slots:
            time_slots = date_slots[date_str]
            # Only show dates that have available time slots
            if time_slots:
                available_dates.append({
                    "day_name": target_date.strftime("%a"),
                    "day_num": target_date.day,
                    "month": target_date.strftime("%b"),
                    "full_date": date_str,
                    "time_slots": time_slots  # Include time slots for reference
                })
    
    return templates.TemplateResponse("mentor_profile.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "services": services,
        "available_dates": available_dates[:7]  # Show up to 7 dates
    })
# ============ ERROR HANDLERS ============
@app.get("/mentor/username/{username}")
async def redirect_old_mentor_url(username: str):
    """Redirect old /mentor/username/{username} URLs to new /{username} format"""
    return RedirectResponse(url=f"/{username}", status_code=301)

@app.get("/mentor/username/{username}/service/{service_id}")
async def redirect_old_service_url(username: str, service_id: int):
    """Redirect old service URLs to new format"""
    return RedirectResponse(url=f"/{username}/service/{service_id}", status_code=301)

# Then your API endpoint

    
@app.get("/{username}", response_class=HTMLResponse)
async def user_profile(
    request: Request,
    username: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """User profile page - works for both mentors and learners"""
    # Find user by username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # If user is a mentor, show mentor profile
    if user.role == "mentor":
        mentor = db.query(Mentor).filter(Mentor.user_id == user.id).first()
        if not mentor:
            raise HTTPException(status_code=404, detail="Mentor profile not found")
        
        # Clean up past availabilities for this mentor
        today = datetime.now().date()
        try:
            db.query(Availability).filter(
                Availability.mentor_id == mentor.id,
                Availability.date < today
            ).delete(synchronize_session=False)
            db.commit()
        except Exception as e:
            print(f"Error cleaning up past availabilities for mentor {mentor.id}: {e}")
            db.rollback()
        
        services = db.query(Service).filter(
            Service.mentor_id == mentor.id,
            Service.is_active == True
        ).all()
        
        # Get actual availabilities from database (future dates only)
        availabilities = db.query(Availability).filter(
            Availability.mentor_id == mentor.id,
            Availability.is_booked == False,
            Availability.date >= today
        ).order_by(Availability.date).all()
        
        # Group availabilities by date for the frontend
        from collections import defaultdict
        date_slots = defaultdict(list)
        
        for avail in availabilities:
            if isinstance(avail.date, datetime):
                date_str = avail.date.strftime("%Y-%m-%d")
                date_obj = avail.date
            elif isinstance(avail.date, date):
                date_str = avail.date.strftime("%Y-%m-%d")
                date_obj = avail.date
            else:
                try:
                    date_str = datetime.strptime(str(avail.date), "%Y-%m-%d").strftime("%Y-%m-%d")
                    date_obj = datetime.strptime(str(avail.date), "%Y-%m-%d").date()
                except:
                    continue
            
            # Only include if date is not in the past
            if date_obj >= today:
                date_slots[date_str].append({
                    "start_time": avail.start_time,
                    "end_time": avail.end_time
                })
        
        # Prepare available dates for display (only dates with availability)
        available_dates = []
        for date_str in sorted(date_slots.keys()):
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                if date_obj >= today:  # Ensure date is not in the past
                    time_slots = date_slots[date_str]
                    # Only show dates that have available time slots
                    if time_slots:
                        available_dates.append({
                            "day_name": date_obj.strftime("%a"),
                            "day_num": date_obj.day,
                            "month": date_obj.strftime("%b"),
                            "full_date": date_str,
                            "time_slots": time_slots
                        })
            except ValueError:
                continue
        
        # Limit to 7 dates for display
        return templates.TemplateResponse("mentor_profile.html", {
            "request": request,
            "current_user": current_user,
            "mentor": mentor,
            "services": services,
            "available_dates": available_dates[:7]  # Only show up to 7 available dates
        })
    
    # If user is a learner, show simple learner profile
    elif user.role == "learner":
        return templates.TemplateResponse("learner_profile.html", {
            "request": request,
            "current_user": current_user,
            "profile_user": user
        })
    
    else:
        raise HTTPException(status_code=404, detail="Profile not found")
# Service page by username and service ID
@app.get("/{username}/service/{service_id}", response_class=HTMLResponse)
async def user_service_page(
    request: Request,
    username: str,
    service_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Service page for a mentor's service"""
    # Find user by username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user is a mentor
    if user.role != "mentor":
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Get mentor profile
    mentor = db.query(Mentor).filter(Mentor.user_id == user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Get the specific service
    service = db.query(Service).filter(
        Service.id == service_id,
        Service.mentor_id == mentor.id,
        Service.is_active == True
    ).first()
    
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Get other services by the same mentor (for related services)
    other_services = db.query(Service).filter(
        Service.mentor_id == mentor.id,
        Service.id != service_id,
        Service.is_active == True
    ).limit(4).all()
    
    # Get availabilities for this service (only future dates)
    today = datetime.now().date()
    availabilities = db.query(Availability).filter(
        Availability.mentor_id == mentor.id,
        Availability.is_booked == False,
        Availability.date >= today,
        or_(
            Availability.service_id == service_id,
            Availability.service_id == None
        )
    ).order_by(Availability.date).all()
    
    # Group availabilities by date
    from collections import defaultdict
    date_slots = defaultdict(list)
    
    for avail in availabilities:
        if isinstance(avail.date, datetime):
            date_str = avail.date.strftime("%Y-%m-%d")
            date_obj = avail.date
        elif isinstance(avail.date, date):
            date_str = avail.date.strftime("%Y-%m-%d")
            date_obj = avail.date
        else:
            try:
                date_str = datetime.strptime(str(avail.date), "%Y-%m-%d").strftime("%Y-%m-%d")
                date_obj = datetime.strptime(str(avail.date), "%Y-%m-%d").date()
            except:
                continue
        
        # Only include if date is not in the past
        if date_obj >= today:
            date_slots[date_str].append({
                "start_time": avail.start_time,
                "end_time": avail.end_time
            })
    
    # Prepare available dates for display (only dates with availability)
    available_dates = []
    for date_str in sorted(date_slots.keys()):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            if date_obj >= today:  # Ensure date is not in the past
                time_slots = date_slots[date_str]
                if time_slots:
                    available_dates.append({
                        "day_name": date_obj.strftime("%a"),
                        "day_num": date_obj.day,
                        "month": date_obj.strftime("%b"),
                        "full_date": date_str,
                        "time_slots": time_slots
                    })
        except ValueError:
            continue
    
    # Limit to 7 dates for display
    return templates.TemplateResponse("service_page.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "service": service,
        "other_services": other_services,
        "available_dates": available_dates[:7]  # Only show up to 7 available dates
    })
# ============ API ENDPOINTS ============
# In your FastAPI app
@app.post("/api/create-razorpay-order")
async def create_razorpay_order(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    booking_id = data.get("booking_id")
    amount = data.get("amount")
    
    # Create Razorpay order
    order_data = {
        "amount": amount,
        "currency": "INR",
        "receipt": f"booking_{booking_id}",
        "notes": {
            "booking_id": str(booking_id)
        }
    }
    
    order = razorpay_client.order.create(data=order_data)
    
    # Update booking with order ID
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if booking:
        booking.razorpay_order_id = order["id"]
        db.commit()
    
    return {"order_id": order["id"]}
    
@app.get("/meeting/{booking_id}", response_class=HTMLResponse)
async def meeting_page(
    request: Request,
    booking_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Meeting page with Google Meet integration"""
    booking = db.query(Booking).filter(
        Booking.id == booking_id,
        or_(
            Booking.learner_id == current_user.id,
            Booking.mentor_id == current_user.id  # If user is mentor
        )
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Generate Google Meet link if not exists
    if not booking.meeting_link or "meet.google.com" not in booking.meeting_link:
        # Generate new meeting link
        meeting_id = f"clearq-{uuid.uuid4().hex[:8]}"
        booking.meeting_link = f"https://meet.google.com/{meeting_id}"
        booking.meeting_id = meeting_id
        db.commit()
    
    return templates.TemplateResponse("meeting.html", {
        "request": request,
        "current_user": current_user,
        "booking": booking,
        "meeting_link": booking.meeting_link
    })
@app.post("/api/verify-payment")
async def verify_payment(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    
    # Verify payment signature
    params_dict = {
        'razorpay_order_id': data['razorpay_order_id'],
        'razorpay_payment_id': data['razorpay_payment_id'],
        'razorpay_signature': data['razorpay_signature']
    }
    
    try:
        # Verify signature
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        # Update booking status
        booking = db.query(Booking).filter(Booking.id == data['booking_id']).first()
        if booking:
            booking.payment_status = "paid"
            db.commit()
        
        return {"success": True, "message": "Payment verified"}
    except Exception as e:
        return {"success": False, "error": str(e)}
# Update the time-slots API to support both ID and username
@app.post("/api/time-slots/{identifier}")
async def get_time_slots(
    identifier: str,
    data: dict,
    db: Session = Depends(get_db)
):
    date_str = data.get("date")
    if not date_str:
        return JSONResponse({"success": False, "message": "Date required"})
    
    try:
        # Check if identifier is a number (ID) or string (username)
        mentor_id = None
        if identifier.isdigit():
            # It's an ID
            mentor_id = int(identifier)
            mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
        else:
            # It's a username
            user = db.query(User).filter(User.username == identifier).first()
            if not user:
                return JSONResponse({"success": False, "message": "Mentor not found"})
            mentor = db.query(Mentor).filter(Mentor.user_id == user.id).first()
            if mentor:
                mentor_id = mentor.id
        
        if not mentor or not mentor_id:
            return JSONResponse({"success": False, "message": "Mentor not found"})
        
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JSONResponse({"success": False, "message": "Invalid date format"})
    
    # Get available time slots for this date
    available_slots = db.query(Availability).filter(
        Availability.mentor_id == mentor_id,
        Availability.date == target_date,
        Availability.is_booked == False
    ).order_by(Availability.start_time).all()
    
    # Format time slots for display
    formatted_slots = []
    for slot in available_slots:
        try:
            start_time_obj = datetime.strptime(slot.start_time, "%H:%M")
            end_time_obj = datetime.strptime(slot.end_time, "%H:%M")
            
            # Format time in 12-hour format with AM/PM
            formatted_start = start_time_obj.strftime("%I:%M %p").lstrip("0")
            formatted_end = end_time_obj.strftime("%I:%M %p").lstrip("0")
            
            formatted_slots.append({
                "value": slot.start_time,
                "display": f"{formatted_start} - {formatted_end}",
                "duration": slot.end_time
            })
        except:
            formatted_slots.append({
                "value": slot.start_time,
                "display": f"{slot.start_time} - {slot.end_time}",
                "duration": slot.end_time
            })
    
    if formatted_slots:
        return JSONResponse({
            "success": True,
            "slots": formatted_slots
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "No available time slots for this date"
        })
        
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
