import os
import re
import uuid
from sqlalchemy import DECIMAL
from decimal import Decimal
import requests
import traceback
from datetime import datetime, timedelta
from passlib.context import CryptContext
import time  # Add this import
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
templates.env.globals["now"] = datetime.now()

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Security
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
    digital_product_url = Column(String, nullable=True)
    digital_product_file = Column(String, nullable=True)
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
    is_booked = Column(Boolean, default=False)  # Add this field
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    booking = relationship("Booking", back_populates="time_slots")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(Integer, ForeignKey("users.id"))
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"))
    booking_type = Column(String, default="session")
    booking_date = Column(Date, nullable=True)
    selected_time = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, confirmed, completed, cancelled
    payment_status = Column(String, default="pending")  # pending, paid, failed, refunded, free
    razorpay_order_id = Column(String)
    razorpay_payment_id = Column(String)
    amount_paid = Column(Integer)
    meeting_link = Column(String)
    meeting_id = Column(String)
    meeting_password = Column(String)
    notes = Column(Text)
    download_count = Column(Integer, default=0)
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


class MentorBalance(Base):
    __tablename__ = "mentor_balances"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"), unique=True)
    total_earnings = Column(DECIMAL(10, 2), default=0.00)  # Total earnings (paid + pending)
    available_balance = Column(DECIMAL(10, 2), default=0.00)  # Available for withdrawal
    pending_withdrawal = Column(DECIMAL(10, 2), default=0.00)  # Amount in pending withdrawals
    total_withdrawn = Column(DECIMAL(10, 2), default=0.00)  # Total withdrawn amount
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    mentor = relationship("Mentor")

class MentorWithdrawal(Base):
    __tablename__ = "mentor_withdrawals"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    amount = Column(DECIMAL(10, 2), nullable=False)  # in INR
    status = Column(String, default="pending")  # pending, processing, approved, completed, rejected
    payment_method = Column(String)  # bank_transfer, upi, etc.
    account_details = Column(Text)  # Bank details or UPI ID
    requested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(Integer, ForeignKey("users.id"), nullable=True)  # Admin who processed
    notes = Column(Text, nullable=True)
    
    # Relationships
    mentor = relationship("Mentor")
    admin_user = relationship("User", foreign_keys=[processed_by])

# Add Pydantic schemas after existing schemas


class WithdrawalUpdate(BaseModel):
    status: str
    notes: Optional[str] = None

# Add this function to calculate mentor's withdrawable balance
def get_mentor_withdrawable_balance(mentor_id: int, db: Session) -> Decimal:
    """Calculate mentor's available balance for withdrawal"""
    # Get total paid earnings from bookings
    total_paid = db.query(func.sum(Booking.amount_paid)).filter(
        Booking.mentor_id == mentor_id,
        Booking.payment_status == "paid"
    ).scalar() or 0
    
    # Get total withdrawals (completed + pending)
    total_withdrawn_completed = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.mentor_id == mentor_id,
        MentorWithdrawal.status.in_(["completed", "approved", "processing"])
    ).scalar() or 0
    
    # Get pending withdrawals
    pending_withdrawals = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.mentor_id == mentor_id,
        MentorWithdrawal.status == "pending"
    ).scalar() or 0
    
    # Calculate available balance
    total_paid_decimal = Decimal(str(total_paid))
    total_withdrawn_decimal = Decimal(str(total_withdrawn_completed))
    pending_withdrawals_decimal = Decimal(str(pending_withdrawals))
    
    # Available = total paid - (completed withdrawals + pending withdrawals)
    available = total_paid_decimal - (total_withdrawn_decimal + pending_withdrawals_decimal)
    
    # Ensure non-negative
    return max(available, Decimal('0'))
    
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

class WithdrawalRequest(BaseModel):
    amount: Decimal
    payment_method: str
    account_details: Optional[str] = None


# ============ DEPENDENCIES ============
# Add this function after your database models


# Add this to your imports
import uuid
import requests
from datetime import datetime, timedelta

# Add this function after your imports
def generate_meeting_link(booking_id: int, db: Session):
    """Generate or retrieve meeting link for a confirmed booking"""
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # If meeting link already exists, return it
    if booking.meeting_link and booking.meeting_id:
        return booking.meeting_link, booking.meeting_id
    
    # Get user details
    mentor = db.query(Mentor).filter(Mentor.id == booking.mentor_id).first()
    learner = db.query(User).filter(User.id == booking.learner_id).first()
    
    if not mentor or not learner:
        raise HTTPException(status_code=404, detail="User details not found")
    
    # Generate simple Jitsi meeting link
    meeting_id = f"clearq-{booking_id}-{uuid.uuid4().hex[:8]}"
    meeting_link = f"https://meet.jit.si/{meeting_id}"
    
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
    
    # Also update all TimeSlot records for this booking
    time_slots = db.query(TimeSlot).filter(TimeSlot.booking_id == booking.id).all()
    for time_slot in time_slots:
        time_slot.is_booked = True
    
    db.commit()
    
    return meeting_link, meeting_id
    
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
    """Remove availability slots from past dates, but keep today's"""
    today = date.today()
    try:
        # Delete availabilities from past dates (before today)
        past_availabilities = db.query(Availability).filter(
            Availability.date < today
        ).all()
        
        # Check each past availability for bookings before deleting
        for avail in past_availabilities:
            # Check if there are any bookings for this availability
            booking_exists = db.query(Booking).filter(
                Booking.mentor_id == avail.mentor_id,
                Booking.booking_date == avail.date,
                Booking.status.in_(["confirmed", "pending"])
            ).first()
            
            # Only delete if no bookings exist
            if not booking_exists:
                db.delete(avail)
        
        db.commit()
        print(f"Cleaned up {len(past_availabilities)} past availabilities")
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
    if not filename or '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def save_profile_image(file: UploadFile, user_id: int) -> str:
    """Save uploaded profile image and return filename"""
    print(f"Saving profile image for user {user_id}: {file.filename}")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Create uploads/profile_images directory if it doesn't exist
    profile_images_dir = UPLOAD_DIR / "profile_images"
    profile_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Upload directory: {profile_images_dir}")
    
    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"profile_{user_id}_{uuid.uuid4().hex[:8]}.{ext}"
    file_path = profile_images_dir / filename
    
    print(f"Saving file to: {file_path}")
    
    # Save file
    try:
        # Reset file pointer (important!)
        file.file.seek(0)
        
        # Copy file content
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise
    
    # Return relative path for database storage
    relative_path = f"uploads/profile_images/{filename}"
    print(f"Returning relative path: {relative_path}")
    return relative_path

# ============ ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, current_user = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_user": current_user,
        "now": datetime.now()
    })

@app.get("/debug/mentor/{mentor_id}/social")
async def debug_social_links(mentor_id: int, db: Session = Depends(get_db)):
    mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
    if not mentor:
        return {"error": "Mentor not found"}
    
    return {
        "linkedin": mentor.linkedin_url,
        "github": mentor.github_url,
        "twitter": mentor.twitter_url,
        "website": mentor.website_url
    }

@app.post("/api/purchase-digital-product")
async def purchase_digital_product(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simplified digital product purchase (no dates/times)"""
    if not current_user or current_user.role != "learner":
        raise HTTPException(status_code=403, detail="Only learners can purchase")
    
    data = await request.json()
    service_id = data.get("service_id")
    
    service = db.query(Service).filter(
        Service.id == service_id,
        Service.is_digital == True,
        Service.is_active == True
    ).first()
    
    if not service:
        raise HTTPException(status_code=404, detail="Digital product not found")
    
    # Check if already purchased - FIXED: Remove status check
    existing_booking = db.query(Booking).filter(
        Booking.service_id == service_id,
        Booking.learner_id == current_user.id,
        Booking.payment_status.in_(["paid", "free"])  # No status check here
    ).first()
    
    if existing_booking:
        print(f"DEBUG: User already owns product {service_id}")
        return JSONResponse({
            "success": False,
            "message": "You already own this product",
            "redirect_url": f"/digital-product/{service_id}"  # Still redirect to product page
        })
    
    # FREE digital product
    if service.price == 0:
        booking = Booking(
            learner_id=current_user.id,
            mentor_id=service.mentor_id,
            service_id=service_id,
            booking_type="digital",
            booking_date=datetime.now().date(),
            selected_time=datetime.now().strftime("%H:%M"),
            razorpay_order_id=None,
            amount_paid=0,
            status="completed",
            payment_status="free",
            meeting_link=None,
            meeting_id=None,
            notes=f"Free digital product: {service.name}",
            download_count=0
        )
        
        db.add(booking)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "redirect_url": f"/digital-product/{service_id}",  # Use service_id
            "message": "Product added to your library!"
        })
    
    # PAID digital product
    else:
        try:
            # Create Razorpay order
            order_amount = service.price * 100
            order_data = {
                "amount": order_amount,
                "currency": "INR",
                "payment_capture": 1,
                "notes": {
                    "service_id": service_id,
                    "learner_id": current_user.id,
                    "type": "digital_product"
                }
            }
            
            razorpay_order = razorpay_client.order.create(order_data)
            
            # Create pending booking
            booking = Booking(
                learner_id=current_user.id,
                mentor_id=service.mentor_id,
                service_id=service_id,
                booking_type="digital",
                booking_date=datetime.now().date(),
                selected_time=datetime.now().strftime("%H:%M"),
                razorpay_order_id=razorpay_order["id"],
                amount_paid=service.price,
                status="pending",
                payment_status="pending",
                meeting_link=None,
                meeting_id=None,
                notes=f"Digital product: {service.name}",
                download_count=0
            )
            
            db.add(booking)
            db.commit()
            
            return JSONResponse({
                "success": True,
                "booking_id": booking.id,
                "redirect_url": f"/payment/{booking.id}",
                "razorpay_order_id": razorpay_order["id"],
                "message": "Please complete payment"
            })
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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


@app.get("/mentor/withdraw", response_class=HTMLResponse)
async def mentor_withdrawal_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mentor withdrawal request page"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Calculate available balance
    available_balance = get_mentor_withdrawable_balance(mentor.id, db)
    
    # Get withdrawal history
    withdrawals = db.query(MentorWithdrawal).filter(
        MentorWithdrawal.mentor_id == mentor.id
    ).order_by(MentorWithdrawal.requested_at.desc()).limit(10).all()
    
    # Get recent earnings (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_earnings = db.query(Booking).filter(
        Booking.mentor_id == mentor.id,
        Booking.payment_status == "paid",
        Booking.created_at >= thirty_days_ago
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    return templates.TemplateResponse("mentor_withdraw.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "available_balance": available_balance,
        "recent_earnings": recent_earnings,
        "withdrawals": withdrawals,
        "min_withdrawal": Decimal('500.00'),  # Minimum withdrawal amount
        "max_withdrawal": available_balance,
        "now": datetime.now()
    })

@app.post("/api/admin/withdrawals/{withdrawal_id}/update")
async def update_withdrawal_status(
    withdrawal_id: int,
    status: str = Form(...),
    notes: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update withdrawal status (admin only)"""
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    withdrawal = db.query(MentorWithdrawal).filter(MentorWithdrawal.id == withdrawal_id).first()
    if not withdrawal:
        raise HTTPException(status_code=404, detail="Withdrawal request not found")
    
    try:
        old_status = withdrawal.status
        withdrawal.status = status
        withdrawal.notes = notes if notes else withdrawal.notes
        
        if status in ["approved", "processing", "completed"]:
            withdrawal.processed_at = datetime.utcnow()
            withdrawal.processed_by = current_user.id
        
        if status == "rejected":
            withdrawal.processed_at = datetime.utcnow()
            withdrawal.processed_by = current_user.id
        
        db.commit()
        
        # Optional: Send notification to mentor
        # You can implement notification system here
        
        return JSONResponse({
            "success": True,
            "message": f"Withdrawal status updated to {status}"
        })
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating withdrawal: {str(e)}")

@app.post("/api/mentor/withdraw/request")
async def request_withdrawal(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Handle withdrawal request from mentor"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        data = await request.json()
        amount = Decimal(str(data.get("amount")))
        payment_method = data.get("payment_method")
        account_details = data.get("account_details")
        notes = data.get("notes", "")
        
        # Validate amount
        if amount <= 0:
            return JSONResponse({
                "success": False,
                "message": "Invalid withdrawal amount"
            }, status_code=400)
        
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if not mentor:
            return JSONResponse({
                "success": False,
                "message": "Mentor profile not found"
            }, status_code=404)
        
        # Calculate available balance
        available_balance = get_mentor_withdrawable_balance(mentor.id, db)
        
        # Check minimum withdrawal amount
        if amount < Decimal('500.00'):
            return JSONResponse({
                "success": False,
                "message": "Minimum withdrawal amount is ₹500"
            }, status_code=400)
        
        # Check if sufficient balance
        if amount > available_balance:
            return JSONResponse({
                "success": False,
                "message": f"Insufficient balance. Available: ₹{available_balance}"
            }, status_code=400)
        
        # Create withdrawal request
        withdrawal = MentorWithdrawal(
            mentor_id=mentor.id,
            amount=amount,
            payment_method=payment_method,
            account_details=account_details,
            notes=notes,
            status="pending",
            requested_at=datetime.utcnow()
        )
        
        db.add(withdrawal)
        db.commit()
        db.refresh(withdrawal)
        
        # Optional: Send notification to admin
        # You can implement notification system here
        
        return JSONResponse({
            "success": True,
            "message": "Withdrawal request submitted successfully!",
            "withdrawal_id": withdrawal.id,
            "redirect_url": "/dashboard"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error processing withdrawal: {str(e)}"
        }, status_code=500)

@app.get("/admin/withdrawals", response_class=HTMLResponse)
async def admin_withdrawals_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[str] = "pending",
    page: int = 1,
    limit: int = 20
):
    """Admin withdrawal management page"""
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Build query based on status filter
    query = db.query(MentorWithdrawal).join(Mentor).join(User)
    
    if status and status != "all":
        query = query.filter(MentorWithdrawal.status == status)
    
    # Get total count for pagination
    total_withdrawals = query.count()
    total_pages = (total_withdrawals + limit - 1) // limit if limit > 0 else 1
    
    # Apply pagination
    offset = (page - 1) * limit
    withdrawals = query.order_by(MentorWithdrawal.requested_at.desc()).offset(offset).limit(limit).all()
    
    # Calculate statistics
    total_pending = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.status == "pending"
    ).scalar() or 0
    
    total_approved = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.status == "approved"
    ).scalar() or 0
    
    total_completed = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.status == "completed"
    ).scalar() or 0
    
    # Calculate total for current month
    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_this_month = db.query(func.sum(MentorWithdrawal.amount)).filter(
        MentorWithdrawal.status == "completed",
        MentorWithdrawal.processed_at >= start_of_month
    ).scalar() or 0
    
    return templates.TemplateResponse("admin_withdrawals.html", {
        "request": request,
        "current_user": current_user,
        "withdrawals": withdrawals,
        "status": status,
        "page": page,
        "total_pages": total_pages,
        "total_withdrawals": total_withdrawals,
        "stats": {
            "total_pending": total_pending,
            "total_approved": total_approved,
            "total_completed": total_completed,
            "total_this_month": total_this_month,
            "pending_count": db.query(MentorWithdrawal).filter(MentorWithdrawal.status == "pending").count(),
            "approved_count": db.query(MentorWithdrawal).filter(MentorWithdrawal.status == "approved").count(),
            "completed_count": db.query(MentorWithdrawal).filter(MentorWithdrawal.status == "completed").count(),
        }
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
        # Get ALL bookings for the learner
        all_bookings = db.query(Booking).filter(
            Booking.learner_id == current_user.id
        ).order_by(Booking.created_at.desc()).all()
        
        # Get digital product purchases - FIXED: using correct variable name
        digital_product_purchases = db.query(Booking).filter(
            Booking.learner_id == current_user.id,
            Booking.service.has(is_digital=True),
            Booking.payment_status.in_(["paid", "free"]),
            Booking.status == "completed"
        ).order_by(Booking.created_at.desc()).limit(6).all()
        
        # Separate bookings by status and type
        confirmed_bookings = []
        pending_bookings = []
        upcoming_sessions = []
        free_sessions = []
        digital_product_list = []  # Renamed for clarity
        session_bookings = []  # Non-digital bookings
        
        today = datetime.now().date()
        
        for booking in all_bookings:
            # Safely check if service exists and is digital
            is_digital = False
            if booking.service:
                is_digital = booking.service.is_digital
            
            if is_digital:
                # Digital product booking
                if booking.payment_status in ["paid", "free"] and booking.status == "completed":
                    digital_product_list.append(booking)
                    # Add to confirmed bookings for consistency
                    confirmed_bookings.append(booking)
                elif booking.payment_status == "pending":
                    pending_bookings.append(booking)
            else:
                # Session booking
                if booking.payment_status in ["paid", "free"] and booking.status == "confirmed" and booking.meeting_link:
                    # Confirmed session with meeting link
                    if booking.booking_date >= today:
                        upcoming_sessions.append(booking)
                    confirmed_bookings.append(booking)
                    
                    if booking.payment_status == "free":
                        free_sessions.append(booking)
                elif booking.payment_status == "pending" or booking.status == "pending":
                    # Pending session (awaiting payment)
                    pending_bookings.append(booking)
                else:
                    # Other statuses (cancelled, completed, etc.)
                    confirmed_bookings.append(booking)
                
                # Add to session bookings list
                session_bookings.append(booking)
        
        # Get stats for the learner
        total_sessions = len([b for b in session_bookings if b.payment_status in ["paid", "free"] and b.status == "confirmed"])
        completed_sessions = len([b for b in session_bookings if b.status == "completed"])
        pending_payments = len([b for b in pending_bookings if b.payment_status == "pending"])
        total_digital_products = len(digital_product_list)
        
        # Get recent sessions (last 5)
        recent_sessions = session_bookings[:5] if session_bookings else []
        
        # Get upcoming sessions (next 7 days)
        next_week = today + timedelta(days=7)
        upcoming_next_week = [b for b in upcoming_sessions if hasattr(b, 'booking_date') and b.booking_date and b.booking_date <= next_week][:3]
        
        # FIXED: Use correct variable names
        context.update({
            "all_bookings": all_bookings,
            "session_bookings": session_bookings,
            "digital_products": digital_product_list,  # FIXED: Changed from digital_bookings to digital_product_list
            "digital_product_purchases": digital_product_purchases,  # Add this if needed
            "upcoming_sessions": upcoming_sessions,
            "upcoming_next_week": upcoming_next_week,
            "confirmed_bookings": confirmed_bookings,
            "pending_bookings": pending_bookings,
            "free_sessions": free_sessions,
            "recent_sessions": recent_sessions,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "pending_payments": pending_payments,
            "total_digital_products": total_digital_products,
            "stats": {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "pending_payments": pending_payments,
                "digital_products": total_digital_products,
                "total_bookings": len(all_bookings)
            }
        })
    
    elif current_user.role == "mentor":
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        if mentor:
            # Show ALL bookings for the mentor
            all_bookings = db.query(Booking).filter(
                Booking.mentor_id == mentor.id
            ).order_by(Booking.created_at.desc()).all()
            
            # Separate bookings by type and status
            confirmed_bookings = []
            pending_bookings = []
            upcoming_sessions = []
            free_sessions = []
            digital_product_sales = []
            session_bookings = []  # Non-digital bookings
            
            today = datetime.now().date()
            
            for booking in all_bookings:
                # Check if this is a digital product
                is_digital = False
                if booking.service:
                    is_digital = booking.service.is_digital
                
                if is_digital:
                    # Digital product sale
                    if booking.payment_status in ["paid", "free"] and booking.status == "completed":
                        digital_product_sales.append(booking)
                        confirmed_bookings.append(booking)
                    elif booking.payment_status == "pending":
                        pending_bookings.append(booking)
                else:
                    # Session booking
                    if booking.payment_status in ["paid", "free"] and booking.status == "confirmed" and booking.meeting_link:
                        if hasattr(booking, 'booking_date') and booking.booking_date and booking.booking_date >= today:
                            upcoming_sessions.append(booking)
                        confirmed_bookings.append(booking)
                        
                        if booking.payment_status == "free":
                            free_sessions.append(booking)
                    elif booking.payment_status == "pending" or booking.status == "pending":
                        pending_bookings.append(booking)
                    else:
                        confirmed_bookings.append(booking)
                    
                    session_bookings.append(booking)
            
            # Calculate earnings (only from paid bookings, both sessions and digital products)
            earnings_query = db.query(Booking).filter(
                Booking.mentor_id == mentor.id,
                Booking.payment_status == "paid"
            ).with_entities(func.sum(Booking.amount_paid))
            
            earnings = earnings_query.scalar() or 0
            
            # Calculate digital product earnings separately
            digital_earnings_query = db.query(Booking).filter(
                Booking.mentor_id == mentor.id,
                Booking.payment_status == "paid",
                Booking.service.has(is_digital=True)
            ).with_entities(func.sum(Booking.amount_paid))
            
            digital_earnings = digital_earnings_query.scalar() or 0
            
            # Get stats for the mentor
            total_bookings = len(all_bookings)
            session_bookings_count = len(session_bookings)
            digital_sales_count = len(digital_product_sales)
            pending_payments = len(pending_bookings)
            upcoming_count = len(upcoming_sessions)
            
            # Get recent sales (last 5)
            recent_sales = all_bookings[:5] if all_bookings else []
            
            # Get top services
            top_services = db.query(
                Service.name,
                func.count(Booking.id).label('sales_count'),
                func.sum(Booking.amount_paid).label('revenue')
            ).join(Booking, Service.id == Booking.service_id).filter(
                Booking.mentor_id == mentor.id,
                Booking.payment_status.in_(["paid", "free"])
            ).group_by(Service.id).order_by(func.count(Booking.id).desc()).limit(5).all()
            
            context.update({
                "mentor": mentor,
                "all_bookings": all_bookings,
                "session_bookings": session_bookings,
                "digital_product_sales": digital_product_sales,
                "upcoming_sessions": upcoming_sessions,
                "confirmed_bookings": confirmed_bookings,
                "pending_bookings": pending_bookings,
                "free_sessions": free_sessions,
                "recent_sales": recent_sales,
                "top_services": top_services,
                "earnings": earnings,
                "digital_earnings": digital_earnings,
                "session_earnings": earnings - digital_earnings,
                "total_bookings": total_bookings,
                "session_bookings_count": session_bookings_count,
                "digital_sales_count": digital_sales_count,
                "pending_payments": pending_payments,
                "upcoming_count": upcoming_count,
                "stats": {
                    "total": total_bookings,
                    "sessions": session_bookings_count,
                    "digital_sales": digital_sales_count,
                    "pending": pending_payments,
                    "upcoming": upcoming_count,
                    "earnings": earnings,
                    "digital_earnings": digital_earnings,
                    "session_earnings": earnings - digital_earnings
                }
            })
        else:
            # Mentor profile doesn't exist, create one
            mentor = Mentor(user_id=current_user.id, verification_status="pending")
            db.add(mentor)
            db.commit()
            db.refresh(mentor)
            
            context.update({
                "mentor": mentor,
                "all_bookings": [],
                "digital_product_sales": [],
                "upcoming_sessions": [],
                "confirmed_bookings": [],
                "pending_bookings": [],
                "free_sessions": [],
                "earnings": 0,
                "total_bookings": 0,
                "pending_payments": 0,
                "upcoming_count": 0,
                "stats": {
                    "total": 0,
                    "pending": 0,
                    "upcoming": 0,
                    "earnings": 0
                }
            })
    
    elif current_user.role == "admin":
        # Pending mentor verifications
        pending_mentors = db.query(Mentor).filter(
            Mentor.verification_status == "pending"
        ).join(User).all()
        
        # Get all recent bookings (last 20)
        recent_bookings = db.query(Booking).order_by(
            Booking.created_at.desc()
        ).limit(20).all()
        
        # Get recent digital product sales
        recent_digital_sales = db.query(Booking).filter(
            Booking.service.has(is_digital=True),
            Booking.payment_status == "paid"
        ).order_by(Booking.created_at.desc()).limit(10).all()
        
        # Calculate admin stats
        total_users = db.query(User).count()
        total_mentors = db.query(Mentor).filter(
            Mentor.is_verified_by_admin == True
        ).count()
        total_bookings = db.query(Booking).count()
        total_digital_sales = db.query(Booking).filter(
            Booking.service.has(is_digital=True),
            Booking.payment_status == "paid"
        ).count()
        
        # Calculate revenue
        total_revenue = db.query(Booking).filter(
            Booking.payment_status == "paid"
        ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
        
        digital_revenue = db.query(Booking).filter(
            Booking.service.has(is_digital=True),
            Booking.payment_status == "paid"
        ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
        
        session_revenue = total_revenue - digital_revenue
        
        # Get pending bookings count
        pending_bookings_count = db.query(Booking).filter(
            Booking.payment_status == "pending"
        ).count()
        
        # Get today's stats
        today = datetime.now().date()
        today_bookings = db.query(Booking).filter(
            func.date(Booking.created_at) == today
        ).count()
        
        today_revenue = db.query(Booking).filter(
            func.date(Booking.created_at) == today,
            Booking.payment_status == "paid"
        ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0

        withdrawal_stats = {
        "pending_withdrawals": db.query(MentorWithdrawal).filter(
            MentorWithdrawal.status == "pending"
        ).count(),
        "pending_withdrawal_amount": db.query(func.sum(MentorWithdrawal.amount)).filter(
            MentorWithdrawal.status == "pending"
        ).scalar() or 0,
        "recent_withdrawals": db.query(MentorWithdrawal).order_by(
            MentorWithdrawal.requested_at.desc()
        ).limit(5).all()
        }
        
        context.update({
            "pending_mentors": pending_mentors,
            "recent_bookings": recent_bookings,
            "recent_digital_sales": recent_digital_sales,
            "stats": {
                "total_users": total_users,
                "total_mentors": total_mentors,
                "total_bookings": total_bookings,
                "total_digital_sales": total_digital_sales,
                "total_revenue": total_revenue,
                "digital_revenue": digital_revenue,
                "session_revenue": session_revenue,
                "pending_bookings": pending_bookings_count,
                "today_bookings": today_bookings,
                "today_revenue": today_revenue,
                "withdrawal_stats": withdrawal_stats
            }
        })
    
    # Add current datetime for templates
    context["now"] = datetime.now()
    
    return templates.TemplateResponse("dashboard.html", context)

def generate_jitsi_meeting_link(booking_id: int, mentor_name: str, learner_name: str):
    """Generate Jitsi meeting link"""
    # Simple Jitsi meeting link with booking ID and random string
    import uuid
    meeting_id = f"clearq-{booking_id}-{uuid.uuid4().hex[:8]}"
    return f"https://meet.jit.si/{meeting_id}", meeting_id
    
@app.post("/mentor/services/{service_id}/update-digital-url")
async def update_digital_product_url(
    service_id: int,
    digital_product_url: str = Form(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update digital product URL"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    service = db.query(Service).filter(
        Service.id == service_id,
        Service.mentor_id == mentor.id
    ).first()
    
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    if not service.is_digital:
        raise HTTPException(status_code=400, detail="This is not a digital product")
    
    # Validate URL
    import re
    url_pattern = re.compile(
        r'^(https?://)'  # http:// or https://
        r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
        r'(:\d+)?'  # optional port
        r'(/.*)?$'  # optional path
    )
    
    if not url_pattern.match(digital_product_url):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    # Update URL
    service.digital_product_url = digital_product_url
    db.commit()
    
    return JSONResponse({
        "success": True,
        "message": "Digital product URL updated successfully"
    })
    
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
        print(f"=== UPDATING PROFILE FOR USER ID: {current_user.id} ===")
        print(f"Profile photo received: {profile_photo.filename if profile_photo else 'None'}")
        
        # Get the user from database
        user = db.query(User).filter(User.id == current_user.id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update full name if provided
        if full_name:
            user.full_name = full_name
            print(f"Updated full_name: {full_name}")
        
        # Handle profile photo upload
        if profile_photo and profile_photo.filename:
            print(f"Processing profile photo: {profile_photo.filename}")
            
            # Validate file type
            if not allowed_file(profile_photo.filename):
                error_msg = "Invalid file type. Allowed: PNG, JPG, JPEG, GIF"
                print(error_msg)
                return RedirectResponse(
                    url=f"/profile/edit?error={error_msg.replace(' ', '%20')}",
                    status_code=303
                )
            
            # Check file size (max 5MB) without consuming the file
            try:
                # Read a small chunk to check size
                chunk = await profile_photo.read(5 * 1024 * 1024 + 1)  # Read 5MB + 1 byte
                file_size = len(chunk)
                print(f"File size: {file_size} bytes")
                
                if file_size > 5 * 1024 * 1024:  # 5MB
                    error_msg = "File size must be less than 5MB"
                    print(error_msg)
                    return RedirectResponse(
                        url=f"/profile/edit?error={error_msg.replace(' ', '%20')}",
                        status_code=303
                    )
                
                # Reset file pointer for save_profile_image
                await profile_photo.seek(0)
                
                # Delete old profile image if exists and not default
                if user.profile_image and user.profile_image != "default-avatar.png":
                    old_image_path = UPLOAD_DIR / user.profile_image
                    if old_image_path.exists():
                        print(f"Deleting old image: {old_image_path}")
                        old_image_path.unlink()
                
                # Save new profile image (use await if function is async, remove await if sync)
                filename = save_profile_image(profile_photo, current_user.id)  # No await needed now
                user.profile_image = filename
                print(f"✅ New profile image saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing profile photo: {str(e)}")
                return RedirectResponse(
                    url=f"/profile/edit?error=Error%20processing%20profile%20photo%20-%20{str(e).replace(' ', '%20')}",
                    status_code=303
                )
        
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
                print("Created new mentor profile")
            else:
                # Update existing mentor profile
                updates = []
                if bio is not None: 
                    mentor.bio = bio
                    updates.append("bio")
                if skills is not None: 
                    mentor.skills = skills
                    updates.append("skills")
                if linkedin_url is not None: 
                    mentor.linkedin_url = linkedin_url
                    updates.append("linkedin_url")
                if github_url is not None: 
                    mentor.github_url = github_url
                    updates.append("github_url")
                if twitter_url is not None: 
                    mentor.twitter_url = twitter_url
                    updates.append("twitter_url")
                if website_url is not None: 
                    mentor.website_url = website_url
                    updates.append("website_url")
                if experience_years is not None: 
                    mentor.experience_years = experience_years
                    updates.append("experience_years")
                if industry is not None: 
                    mentor.industry = industry
                    updates.append("industry")
                if job_title is not None: 
                    mentor.job_title = job_title
                    updates.append("job_title")
                if company is not None: 
                    mentor.company = company
                    updates.append("company")
                
                if updates:
                    print(f"Updated mentor fields: {updates}")
        
        # COMMIT TO DATABASE
        db.commit()
        print(f"✅ Database commit successful!")
        print(f"Final profile_image value: {user.profile_image}")
        
        # Verify the file exists
        if user.profile_image and user.profile_image != "default-avatar.png":
            file_path = UPLOAD_DIR / user.profile_image
            print(f"Checking if file exists: {file_path}")
            print(f"File exists: {file_path.exists()}")
        
        # Redirect with success
        return RedirectResponse(
            url="/profile/edit?success=Profile%20updated%20successfully!",
            status_code=303
        )
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error updating profile: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Error updating profile: {str(e)}"
        return RedirectResponse(
            url=f"/profile/edit?error={error_message.replace(' ', '%20').replace(':', '%3A')}",
            status_code=303
        )
@app.get("/debug/upload-dir")
async def debug_upload_dir():
    """Debug endpoint to check upload directory"""
    try:
        profile_images_dir = UPLOAD_DIR / "profile_images"
        exists = profile_images_dir.exists()
        files = []
        
        if exists:
            files = [f.name for f in profile_images_dir.iterdir() if f.is_file()]
        
        return {
            "upload_dir": str(UPLOAD_DIR),
            "profile_images_dir": str(profile_images_dir),
            "exists": exists,
            "files": files,
            "permissions": oct(os.stat(profile_images_dir).st_mode) if exists else "N/A"
        }
    except Exception as e:
        return {"error": str(e)}

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
from fastapi.responses import FileResponse

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")
    
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
    
    # Validate digital product
    if is_digital:
        duration_minutes = 0
        if not digital_product_url or not digital_product_url.strip():
            # Redirect back with error
            return RedirectResponse(
                url="/mentor/dashboard/services?error=Digital%20product%20requires%20a%20URL",
                status_code=303
            )
        
        # Validate URL format
        import re
        url_pattern = re.compile(
            r'^(https?://)'  # http:// or https://
            r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
            r'(:\d+)?'  # optional port
            r'(/.*)?$'  # optional path
        )
        
        if not url_pattern.match(digital_product_url):
            return RedirectResponse(
                url="/mentor/dashboard/services?error=Please%20enter%20a%20valid%20URL%20starting%20with%20http://%20or%20https://",
                status_code=303
            )
    
    service = Service(
        mentor_id=mentor.id,
        name=name,
        description=description,
        category=category,
        price=price,
        duration_minutes=duration_minutes,
        is_digital=is_digital,
        digital_product_url=digital_product_url if is_digital else None
    )
    
    db.add(service)
    db.commit()
    
    success_msg = "Service created successfully!"
    if is_digital:
        success_msg = "Digital product created successfully! The download link will be available to buyers."
    
    return RedirectResponse(
        url=f"/mentor/dashboard/services?success={success_msg.replace(' ', '%20')}",
        status_code=303
    )

from fastapi import Request, HTTPException, Depends
from fastapi.responses import RedirectResponse

from fastapi.responses import FileResponse, StreamingResponse
import os

@app.get("/digital-product/{service_id}")
async def digital_product_page(
    service_id: int,
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Digital product delivery page - This is the UNIQUE URL for all digital products"""
    # Check authentication
    if not current_user:
        return RedirectResponse(
            url=f"/login?next=/digital-product/{service_id}",
            status_code=303
        )
    
    # Find the service (must be digital and active)
    service = db.query(Service).filter(
        Service.id == service_id,
        Service.is_active == True
    ).first()
    
    if not service:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Get mentor info
    mentor = db.query(Mentor).filter(Mentor.id == service.mentor_id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Creator not found")
    
    mentor_user = db.query(User).filter(User.id == mentor.user_id).first()
    
    # FIXED: Check if user has purchased this product (more flexible query)
    booking = db.query(Booking).filter(
        Booking.service_id == service_id,
        Booking.learner_id == current_user.id,
        Booking.payment_status.in_(["paid", "free"])
        # Removed: Booking.status == "completed"
    ).first()
    
    # Debug logging
    print(f"DEBUG: User {current_user.id} checking service {service_id}")
    print(f"DEBUG: Found booking: {booking}")
    if booking:
        print(f"DEBUG: Booking status: {booking.status}, payment: {booking.payment_status}")
    
    # If user has NOT purchased, show purchase page
    if not booking:
        print(f"DEBUG: No booking found for user {current_user.id}")
        return templates.TemplateResponse("digital_product_purchase.html", {
            "request": request,
            "current_user": current_user,
            "service": service,
            "mentor": mentor_user,
            "is_owner": False
        })
    
    # If user HAS purchased, show delivery page
    print(f"DEBUG: Booking found! Showing delivery page")
    return templates.TemplateResponse("digital_product_delivery.html", {
        "request": request,
        "current_user": current_user,
        "service": service,
        "booking": booking,
        "mentor": mentor_user,
        "is_owner": True
    })

@app.get("/debug/my-digital-products")
async def debug_my_digital_products(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check user's digital product purchases"""
    if not current_user:
        return {"error": "Not logged in"}
    
    # Get all digital product purchases
    purchases = db.query(Booking).join(Service).filter(
        Booking.learner_id == current_user.id,
        Service.is_digital == True
    ).all()
    
    result = []
    for booking in purchases:
        service = db.query(Service).filter(Service.id == booking.service_id).first()
        result.append({
            "booking_id": booking.id,
            "service_id": booking.service_id,
            "service_name": service.name if service else "Unknown",
            "service_is_digital": service.is_digital if service else False,
            "service_url": service.digital_product_url if service else None,
            "payment_status": booking.payment_status,
            "status": booking.status,
            "amount_paid": booking.amount_paid,
            "created_at": str(booking.created_at)
        })
    
    return {"user_id": current_user.id, "purchases": result}
    
    
@app.get("/mentor/availability", response_class=HTMLResponse)
async def mentor_availability_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mentor availability management page"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    # Get or create mentor profile
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        mentor = Mentor(user_id=current_user.id, verification_status="pending")
        db.add(mentor)
        db.commit()
        db.refresh(mentor)
    
    # Get today's date - FIXED: removed extra today()
    today = datetime.now().date()
    
    # Clean up past availabilities first
    try:
        cleanup_past_availabilities(db)
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    # Get all future availabilities
    try:
        # Get availabilities from today onwards
        availabilities = db.query(Availability).filter(
            Availability.mentor_id == mentor.id,
            Availability.date >= today
        ).order_by(Availability.date, Availability.start_time).all()
        
        # Process each availability for display
        for avail in availabilities:
            # Ensure date is a date object
            if isinstance(avail.date, datetime):
                avail.date = avail.date.date()
            
            # Format display fields
            avail.display_date = avail.date.strftime("%Y-%m-%d")
            avail.display_day = avail.date.strftime("%A")
            avail.display_time = f"{avail.start_time} - {avail.end_time}"
            
            # Check if this date/time is actually booked
            booking_exists = db.query(Booking).filter(
                Booking.mentor_id == mentor.id,
                Booking.booking_date == avail.date,
                Booking.status.in_(["confirmed", "pending"]),
                Booking.payment_status.in_(["paid", "free", "pending"])
            ).first() is not None  # FIXED: replaced .exists() with .first() is not None
            
            if booking_exists:
                # Double-check with TimeSlot
                time_slot_exists = db.query(TimeSlot).filter(
                    TimeSlot.date == avail.date,
                    TimeSlot.is_booked == True
                ).first() is not None  # FIXED: replaced .exists() with .first() is not None
                if time_slot_exists:
                    avail.is_booked = True
        
    except Exception as e:
        print(f"Error loading availabilities: {e}")
        availabilities = []
    
    # Get services
    services = db.query(Service).filter(Service.mentor_id == mentor.id).all()
    
    # Get flash messages
    flash_messages = []
    if request.query_params.get("success"):
        flash_messages.append({"category": "success", "message": request.query_params.get("success")})
    if request.query_params.get("error"):
        flash_messages.append({"category": "error", "message": request.query_params.get("error")})
    
    # Get current datetime for template
    current_datetime = datetime.now()
    
    return templates.TemplateResponse("mentor_availability.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "availabilities": availabilities,
        "services": services,
        "today": today.strftime("%Y-%m-%d"),
        "now": current_datetime,  # Pass datetime object
        "flash_messages": flash_messages
    })

@app.get("/debug/availability/{mentor_id}")
async def debug_availability(
    mentor_id: int,
    db: Session = Depends(get_db)
):
    """Debug endpoint to check availability data"""
    mentor = db.query(Mentor).filter(Mentor.id == mentor_id).first()
    if not mentor:
        return {"error": "Mentor not found"}
    
    availabilities = db.query(Availability).filter(
        Availability.mentor_id == mentor_id
    ).all()
    
    bookings = db.query(Booking).filter(
        Booking.mentor_id == mentor_id
    ).all()
    
    time_slots = db.query(TimeSlot).filter(
        TimeSlot.booking.has(mentor_id=mentor_id)
    ).all()
    
    return {
        "mentor": {
            "id": mentor.id,
            "name": mentor.user.full_name if mentor.user else "No user"
        },
        "availabilities": [
            {
                "id": a.id,
                "date": str(a.date),
                "start_time": a.start_time,
                "end_time": a.end_time,
                "is_booked": a.is_booked,
                "service_id": a.service_id
            } for a in availabilities
        ],
        "bookings": [
            {
                "id": b.id,
                "date": str(b.booking_date),
                "time": b.selected_time,
                "status": b.status,
                "payment_status": b.payment_status
            } for b in bookings
        ],
        "time_slots": [
            {
                "id": t.id,
                "date": str(t.date),
                "start_time": t.start_time,
                "end_time": t.end_time,
                "is_booked": t.is_booked
            } for t in time_slots
        ]
    }

@app.post("/api/track-download/{product_id}")
async def track_download(
    product_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Track when a user downloads a digital product"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user has purchased this product
    booking = db.query(Booking).filter(
        Booking.service_id == product_id,
        Booking.learner_id == current_user.id,
        Booking.payment_status.in_(["paid", "free"])
    ).first()
    
    if not booking:
        raise HTTPException(status_code=403, detail="You haven't purchased this product")
    
    # Update download count or track download
    # You could add a download_logs table here
    
    return JSONResponse({
        "success": True,
        "message": "Download tracked"
    })
    
@app.post("/mentor/availability/create")
async def create_availability(
    request: Request,
    date: str = Form(...),
    service_id: Optional[str] = Form(None),
    start_time: str = Form("09:00"),
    end_time: str = Form("21:00"),
    db: Session = Depends(get_db)
):
    """Create a new availability with custom time range"""
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
        return RedirectResponse(
            url="/login?error=Please%20login%20as%20a%20mentor%20to%20add%20availability",
            status_code=303
        )
    
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
        
        # Clean input
        date = date.strip()
        start_time = start_time.strip()
        end_time = end_time.strip()
        
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
        
        # Validate time formats
        try:
            # Parse times to validate format
            start_dt = datetime.strptime(start_time, "%H:%M")
            end_dt = datetime.strptime(end_time, "%H:%M")
            
            # Validate start_time is before end_time
            if start_dt >= end_dt:
                return RedirectResponse(
                    url="/mentor/availability?error=Start%20time%20must%20be%20before%20end%20time",
                    status_code=303
                )
            
            # Validate times are within reasonable bounds (0-23:59)
            if not (0 <= start_dt.hour <= 23 and 0 <= start_dt.minute <= 59):
                return RedirectResponse(
                    url="/mentor/availability?error=Start%20time%20must%20be%20between%2000%3A00%20and%2023%3A59",
                    status_code=303
                )
            
            if not (0 <= end_dt.hour <= 23 and 0 <= end_dt.minute <= 59):
                return RedirectResponse(
                    url="/mentor/availability?error=End%20time%20must%20be%20between%2000%3A00%20and%2023%3A59",
                    status_code=303
                )
            
            # Minimum slot duration (15 minutes)
            min_duration = timedelta(minutes=15)
            if (end_dt - start_dt) < min_duration:
                return RedirectResponse(
                    url="/mentor/availability?error=Time%20slot%20must%20be%20at%20least%2015%20minutes",
                    status_code=303
                )
            
        except ValueError:
            return RedirectResponse(
                url="/mentor/availability?error=Invalid%20time%20format.%20Use%20HH%3AMM%20(24-hour%20format)",
                status_code=303
            )
        
        # Parse and validate service_id if provided
        parsed_service_id = None
        if service_id and service_id.strip() and service_id.lower() != "none" and service_id != "null":
            try:
                parsed_service_id = int(service_id)
                
                # Validate the service belongs to this mentor
                service = db.query(Service).filter(
                    Service.id == parsed_service_id,
                    Service.mentor_id == mentor.id
                ).first()
                
                if not service:
                    return RedirectResponse(
                        url="/mentor/availability?error=Invalid%20service%20selected",
                        status_code=303
                    )
                    
                # Also check if service is active
                if not service.is_active:
                    return RedirectResponse(
                        url="/mentor/availability?error=Selected%20service%20is%20not%20active",
                        status_code=303
                    )
                    
            except ValueError:
                return RedirectResponse(
                    url="/mentor/availability?error=Invalid%20service%20ID",
                    status_code=303
                )
        
        # Check if availability already exists for this date
        existing = db.query(Availability).filter(
            Availability.mentor_id == mentor.id,
            Availability.date == parsed_date
        ).first()
        
        if existing:
            # If availability exists but is booked, we shouldn't allow overriding
            if existing.is_booked:
                return RedirectResponse(
                    url="/mentor/availability?error=Cannot%20modify%20availability%20that%20has%20bookings",
                    status_code=303
                )
            
            # If it exists but is not booked, update it with new times
            existing.start_time = start_time
            existing.end_time = end_time
            existing.service_id = parsed_service_id
            existing.is_booked = False
            
            db.commit()
            
            return RedirectResponse(
                url="/mentor/availability?success=Availability%20updated%20successfully",
                status_code=303
            )
        
        # Check if there are any bookings for this date
        existing_bookings = db.query(Booking).filter(
            Booking.mentor_id == mentor.id,
            Booking.booking_date == parsed_date,
            Booking.status.in_(["pending", "confirmed"])
        ).first()
        
        if existing_bookings:
            return RedirectResponse(
                url="/mentor/availability?error=Cannot%20add%20availability%20for%20date%20with%20existing%20bookings",
                status_code=303
            )
        
        # Create new availability
        availability = Availability(
            mentor_id=mentor.id,
            service_id=parsed_service_id,
            date=parsed_date,
            start_time=start_time,
            end_time=end_time,
            is_booked=False,
            created_at=datetime.utcnow()
        )
        
        db.add(availability)
        db.commit()
        
        # Format success message with the times
        start_display = datetime.strptime(start_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
        end_display = datetime.strptime(end_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
        
        success_msg = f"Availability%20added%20successfully%20({start_display}%20-%20{end_display})"
        
        return RedirectResponse(
            url=f"/mentor/availability?success={success_msg}",
            status_code=303
        )
        
    except Exception as e:
        db.rollback()
        error_msg = f"Error creating availability: {str(e)}"
        # Clean up error message for URL
        clean_error = error_msg.replace(" ", "%20").replace(":", "%3A")
        return RedirectResponse(
            url=f"/mentor/availability?error={clean_error}",
            status_code=303
        )
# ============ BOOKING & PAYMENT ROUTES ============

def generate_meeting_link(booking_id: int, db: Session):
    """Generate or retrieve meeting link for a confirmed booking"""
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # If meeting link already exists, return it
    if booking.meeting_link and booking.meeting_id:
        return booking.meeting_link, booking.meeting_id
    
    # Get user details for the meeting
    mentor = db.query(User).filter(User.id == booking.mentor.user_id).first()
    learner = db.query(User).filter(User.id == booking.learner_id).first()
    
    mentor_name = mentor.full_name or mentor.username
    learner_name = learner.full_name or learner.username
    
    # Generate Jitsi meeting link
    meeting_link, meeting_id = generate_jitsi_meeting_link(
        booking_id=booking.id,
        mentor_name=mentor_name,
        learner_name=learner_name
    )
    
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
    
    # Also update all TimeSlot records for this booking
    time_slots = db.query(TimeSlot).filter(TimeSlot.booking_id == booking.id).all()
    for time_slot in time_slots:
        time_slot.is_booked = True
    
    db.commit()
    
    # Send notification (optional)
    send_meeting_notification(booking, meeting_link, db)
    
    return meeting_link, meeting_id

def send_meeting_notification(booking: Booking, meeting_link: str, db: Session):
    """Send meeting details to both users"""
    # In production, you'd send emails or push notifications
    # For now, we'll just print/log
    print(f"Meeting created for Booking #{booking.id}")
    print(f"Meeting Link: {meeting_link}")
    print(f"Date: {booking.booking_date}")
    print(f"Time: {booking.selected_time}")
    print(f"Participants: Learner {booking.learner_id}, Mentor {booking.mentor_id}")

# ============ MENTOR PAYOUT ROUTES ============

@app.get("/mentor/payout", response_class=HTMLResponse)
async def mentor_payout_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mentor payout dashboard"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Get or create mentor balance
    balance = db.query(MentorBalance).filter(MentorBalance.mentor_id == mentor.id).first()
    if not balance:
        balance = MentorBalance(
            mentor_id=mentor.id,
            total_earnings=0,
            available_balance=0,
            pending_withdrawal=0,
            total_withdrawn=0
        )
        db.add(balance)
        db.commit()
        db.refresh(balance)
    
    # Calculate actual earnings from paid bookings
    earnings_query = db.query(Booking).filter(
        Booking.mentor_id == mentor.id,
        Booking.payment_status == "paid"
    ).with_entities(func.sum(Booking.amount_paid))
    
    total_earnings = earnings_query.scalar() or 0
    
    # Update balance with actual earnings
    if Decimal(str(total_earnings)) != balance.total_earnings:
        balance.total_earnings = Decimal(str(total_earnings))
        
        # Calculate available balance (total earnings - pending withdrawals - already withdrawn)
        total_withdrawn_query = db.query(func.sum(MentorPayout.amount)).filter(
            MentorPayout.mentor_id == mentor.id,
            MentorPayout.status == "completed"
        ).scalar() or 0
        
        pending_withdrawal_query = db.query(func.sum(MentorPayout.amount)).filter(
            MentorPayout.mentor_id == mentor.id,
            MentorPayout.status.in_(["pending", "processing"])
        ).scalar() or 0
        
        balance.available_balance = Decimal(str(total_earnings)) - Decimal(str(total_withdrawn_query)) - Decimal(str(pending_withdrawal_query))
        balance.total_withdrawn = Decimal(str(total_withdrawn_query))
        balance.pending_withdrawal = Decimal(str(pending_withdrawal_query))
        db.commit()
    
    # Get withdrawal history
    withdrawal_history = db.query(MentorPayout).filter(
        MentorPayout.mentor_id == mentor.id
    ).order_by(MentorPayout.request_date.desc()).limit(20).all()
    
    # Get recent earnings (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_earnings = db.query(Booking).filter(
        Booking.mentor_id == mentor.id,
        Booking.payment_status == "paid",
        Booking.created_at >= thirty_days_ago
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    # Get upcoming payouts (bookings that will be eligible for withdrawal)
    # Assuming 7-day clearing period
    seven_days_ago = datetime.now() - timedelta(days=7)
    upcoming_payouts = db.query(Booking).filter(
        Booking.mentor_id == mentor.id,
        Booking.payment_status == "paid",
        Booking.created_at >= seven_days_ago
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    return templates.TemplateResponse("mentor_payout.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "balance": balance,
        "withdrawal_history": withdrawal_history,
        "recent_earnings": recent_earnings,
        "upcoming_payouts": upcoming_payouts,
        "min_withdrawal_amount": 500,  # Minimum withdrawal amount in INR
        "now": datetime.now()
    })

@app.post("/api/mentor/request-withdrawal")
async def request_withdrawal(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Handle withdrawal request"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    data = await request.json()
    amount = Decimal(str(data.get("amount")))
    payment_method = data.get("payment_method")
    account_details = data.get("account_details")
    
    # Validate amount
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid withdrawal amount")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Get mentor balance
    balance = db.query(MentorBalance).filter(MentorBalance.mentor_id == mentor.id).first()
    if not balance:
        raise HTTPException(status_code=400, detail="Balance not found")
    
    # Check minimum withdrawal amount
    if amount < 500:  # Minimum 500 INR
        raise HTTPException(status_code=400, detail="Minimum withdrawal amount is ₹500")
    
    # Check if sufficient balance
    if amount > balance.available_balance:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    try:
        # Create withdrawal request
        withdrawal = MentorPayout(
            mentor_id=mentor.id,
            amount=amount,
            status="pending",
            payment_method=payment_method,
            account_details=account_details,
            request_date=datetime.utcnow()
        )
        
        db.add(withdrawal)
        
        # Update mentor balance
        balance.available_balance -= amount
        balance.pending_withdrawal += amount
        balance.last_updated = datetime.utcnow()
        
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Withdrawal request submitted successfully!",
            "withdrawal_id": withdrawal.id
        })
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing withdrawal: {str(e)}")

# ============ ADMIN PAYOUT ROUTES ============

@app.get("/admin/payouts", response_class=HTMLResponse)
async def admin_payouts_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[str] = "pending",
    page: int = 1,
    limit: int = 20
):
    """Admin payout management page"""
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Build query based on status filter
    query = db.query(MentorPayout).join(Mentor).join(User)
    
    if status and status != "all":
        query = query.filter(MentorPayout.status == status)
    
    # Get total count for pagination
    total_payouts = query.count()
    total_pages = (total_payouts + limit - 1) // limit if limit > 0 else 1
    
    # Apply pagination
    offset = (page - 1) * limit
    payouts = query.order_by(MentorPayout.request_date.desc()).offset(offset).limit(limit).all()
    
    # Calculate total pending amount
    total_pending = db.query(func.sum(MentorPayout.amount)).filter(
        MentorPayout.status == "pending"
    ).scalar() or 0
    
    # Calculate total processed this month
    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_this_month = db.query(func.sum(MentorPayout.amount)).filter(
        MentorPayout.status == "completed",
        MentorPayout.processed_date >= start_of_month
    ).scalar() or 0
    
    return templates.TemplateResponse("admin_payouts.html", {
        "request": request,
        "current_user": current_user,
        "payouts": payouts,
        "status": status,
        "page": page,
        "total_pages": total_pages,
        "total_payouts": total_payouts,
        "total_pending": total_pending,
        "total_this_month": total_this_month,
        "stats": {
            "pending_count": db.query(MentorPayout).filter(MentorPayout.status == "pending").count(),
            "processing_count": db.query(MentorPayout).filter(MentorPayout.status == "processing").count(),
            "completed_count": db.query(MentorPayout).filter(MentorPayout.status == "completed").count(),
            "failed_count": db.query(MentorPayout).filter(MentorPayout.status == "failed").count(),
        }
    })

@app.post("/admin/payouts/{payout_id}/update")
async def update_payout_status(
    payout_id: int,
    status: str = Form(...),
    notes: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update payout status (admin only)"""
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    payout = db.query(MentorPayout).filter(MentorPayout.id == payout_id).first()
    if not payout:
        raise HTTPException(status_code=404, detail="Payout not found")
    
    # Get mentor balance
    balance = db.query(MentorBalance).filter(MentorBalance.mentor_id == payout.mentor_id).first()
    if not balance:
        raise HTTPException(status_code=404, detail="Mentor balance not found")
    
    try:
        old_status = payout.status
        payout.status = status
        payout.notes = notes
        
        if status == "completed":
            payout.processed_date = datetime.utcnow()
            # Update mentor balance
            balance.pending_withdrawal -= payout.amount
            balance.total_withdrawn += payout.amount
            
        elif status == "failed" and old_status in ["pending", "processing"]:
            # Refund the amount to available balance
            balance.available_balance += payout.amount
            balance.pending_withdrawal -= payout.amount
            
        elif status == "processing":
            payout.processed_date = None
        
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": f"Payout status updated to {status}"
        })
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating payout: {str(e)}")

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
    
    # Get mentor
    mentor = db.query(Mentor).filter(Mentor.id == service.mentor_id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    # Check if service is digital
    is_digital_service = service.is_digital
    
    # Skip date/time validation for digital products
    if not is_digital_service:
        # For live sessions, validate date and time
        if not all([date_str, time_slot]):
            raise HTTPException(status_code=400, detail="Missing required booking data")
        
        # Validate date and time for live sessions
        try:
            selected_datetime = datetime.strptime(f"{date_str} {time_slot}", "%Y-%m-%d %H:%M")
            if selected_datetime < datetime.now():
                raise HTTPException(status_code=400, detail="Cannot book sessions in the past")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date or time format")
    else:
        # For digital products, set date and time to current
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_slot = datetime.now().strftime("%H:%M")
    
    # Check if service is free (price = 0)
    is_free_service = service.price == 0
    
    if is_free_service:
        # For free services
        if is_digital_service:
            # Free digital product
            booking = Booking(
                learner_id=current_user.id,
                mentor_id=service.mentor_id,
                service_id=service_id,
                booking_date=datetime.now().date(),
                selected_time=datetime.now().strftime("%H:%M"),
                razorpay_order_id=None,
                amount_paid=0,
                status="completed",
                payment_status="free",
                meeting_link=None,
                meeting_id=None,
                notes=f"Free digital product: {service.name}",
                download_count=0
            )
            
            db.add(booking)
            db.commit()
            db.refresh(booking)
            
            return JSONResponse({
                "success": True,
                "booking_id": booking.id,
                "redirect_url": f"/digital-product/{service_id}",
                "is_digital": True,
                "is_free": True,
                "message": "Free digital product added to your account!"
            })
        else:
            # Free session - generate meeting link
            meeting_id = f"clearq-{uuid.uuid4().hex[:12]}"
            meeting_link = f"https://meet.jit.si/{meeting_id}"
            
            # Calculate end time based on service duration
            start_time = datetime.strptime(time_slot, "%H:%M")
            end_time = start_time + timedelta(minutes=service.duration_minutes)
            end_time_str = end_time.strftime("%H:%M")
            
            # Create booking record
            booking = Booking(
                learner_id=current_user.id,
                mentor_id=service.mentor_id,
                service_id=service_id,
                booking_date=datetime.strptime(date_str, "%Y-%m-%d"),
                selected_time=time_slot,
                razorpay_order_id=None,
                amount_paid=0,
                status="confirmed",
                payment_status="free",
                meeting_link=meeting_link,
                meeting_id=meeting_id,
                notes=f"Free session scheduled for {date_str} at {time_slot}"
            )
            
            db.add(booking)
            db.commit()
            db.refresh(booking)
            
            # Mark the time slot as booked (only for live sessions)
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Find and mark the availability as booked
            availability = db.query(Availability).filter(
                Availability.mentor_id == service.mentor_id,
                Availability.date == target_date
            ).first()
            
            if availability:
                availability.is_booked = True
            
            # Create a time slot record (only for live sessions)
            time_slot_record = TimeSlot(
                booking_id=booking.id,
                start_time=time_slot,
                end_time=end_time_str,
                date=target_date,
                is_booked=True,
                created_at=datetime.utcnow()
            )
            db.add(time_slot_record)
            db.commit()
            
            return JSONResponse({
                "success": True,
                "booking_id": booking.id,
                "redirect_url": f"/dashboard",
                "meeting_link": meeting_link,
                "meeting_id": meeting_id,
                "is_digital": False,
                "is_free": True,
                "message": "Free session booked successfully!"
            })
    else:
        # Paid services
        if is_digital_service:
            # Paid digital product - create Razorpay order
            order_amount = service.price * 100
            order_currency = "INR"
            
            try:
                order_data = {
                    "amount": order_amount,
                    "currency": order_currency,
                    "payment_capture": 1,
                    "notes": {
                        "service_id": service_id,
                        "learner_id": current_user.id,
                        "product_type": "digital"
                    }
                }
                
                razorpay_order = razorpay_client.order.create(order_data)
                
                # Create booking record for digital product
                booking = Booking(
                    learner_id=current_user.id,
                    mentor_id=service.mentor_id,
                    service_id=service_id,
                    booking_date=datetime.now().date(),
                    selected_time=datetime.now().strftime("%H:%M"),
                    razorpay_order_id=razorpay_order["id"],
                    amount_paid=service.price,
                    status="pending",
                    payment_status="pending",
                    meeting_link=None,
                    meeting_id=None,
                    notes=f"Digital product purchase: {service.name}",
                    download_count=0
                )
                
                db.add(booking)
                db.commit()
                db.refresh(booking)
                
                return JSONResponse({
                    "success": True,
                    "booking_id": booking.id,
                    "redirect_url": f"/payment/{booking.id}",
                    "is_digital": True,
                    "is_free": False,
                    "razorpay_order_id": razorpay_order["id"],
                    "message": "Please complete payment to access your digital product."
                })
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Paid session - create Razorpay order
            order_amount = service.price * 100
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
                
                razorpay_order = razorpay_client.order.create(order_data)
                
                # Calculate end time based on service duration
                start_time = datetime.strptime(time_slot, "%H:%M")
                end_time = start_time + timedelta(minutes=service.duration_minutes)
                end_time_str = end_time.strftime("%H:%M")
                
                # Create booking record for session
                booking = Booking(
                    learner_id=current_user.id,
                    mentor_id=service.mentor_id,
                    service_id=service_id,
                    booking_date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    selected_time=time_slot,
                    razorpay_order_id=razorpay_order["id"],
                    amount_paid=service.price,
                    status="pending",
                    payment_status="pending",
                    meeting_link=None,
                    meeting_id=None,
                    notes=f"Session scheduled for {date_str} at {time_slot} (Pending Payment)",
                    download_count=0
                )
                
                db.add(booking)
                db.commit()
                db.refresh(booking)
                
                # Create a TimeSlot record to temporarily reserve the slot (only for live sessions)
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                time_slot_record = TimeSlot(
                    booking_id=booking.id,
                    start_time=time_slot,
                    end_time=end_time_str,
                    date=target_date,
                    is_booked=False,
                    created_at=datetime.utcnow()
                )
                db.add(time_slot_record)
                db.commit()
                
                return JSONResponse({
                    "success": True,
                    "booking_id": booking.id,
                    "redirect_url": f"/payment/{booking.id}",
                    "is_digital": False,
                    "is_free": False,
                    "razorpay_order_id": razorpay_order["id"],
                    "message": "Booking created. Please complete payment to confirm your session."
                })
                
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
# Add this new API endpoint for generating time slots
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
        # Get service
        service = db.query(Service).filter(Service.id == service_id).first()
        if not service:
            return JSONResponse({"success": False, "message": "Service not found"})
        
        duration = service.duration_minutes
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        
        # Validate date
        if target_date < today:
            return JSONResponse({"success": False, "message": "Cannot book sessions for past dates"})
        
        # Get mentor's availability for this date
        availability = db.query(Availability).filter(
            Availability.mentor_id == mentor_id,
            Availability.date == target_date,
            Availability.is_booked == False
        ).first()
        
        if not availability:
            return JSONResponse({"success": False, "message": "No availability for this date"})
        
        # Parse time range
        start_time = datetime.strptime(availability.start_time, "%H:%M")
        end_time = datetime.strptime(availability.end_time, "%H:%M")
        
        # Adjust for today's current time
        if target_date == today:
            now = datetime.now()
            current_time = now.time()
            
            # If start_time is in the past, adjust it
            if start_time.time() < current_time:
                # Set start_time to current time + 30 minutes buffer
                buffer_time = datetime.combine(today, current_time) + timedelta(minutes=30)
                # Round to next 15-minute interval
                minutes_to_add = (15 - buffer_time.minute % 15) % 15
                start_time = buffer_time + timedelta(minutes=minutes_to_add)
        
        # Generate slots
        slots = []
        current_slot = start_time
        slot_duration = timedelta(minutes=duration)
        
        while current_slot + slot_duration <= end_time:
            slot_start_str = current_slot.strftime("%H:%M")
            slot_end_str = (current_slot + slot_duration).strftime("%H:%M")
            
            # Check if this slot is booked
            is_booked = False
            
            # Check Booking table
            booking = db.query(Booking).filter(
                Booking.mentor_id == mentor_id,
                Booking.booking_date == target_date,
                Booking.selected_time == slot_start_str,
                Booking.status.in_(["confirmed", "pending"])
            ).first()
            
            if booking:
                is_booked = True
            
            # Check TimeSlot table
            if not is_booked:
                time_slot = db.query(TimeSlot).filter(
                    TimeSlot.date == target_date,
                    TimeSlot.start_time == slot_start_str,
                    TimeSlot.is_booked == True
                ).first()
                if time_slot:
                    is_booked = True
            
            if not is_booked:
                # Format for display
                display_start = current_slot.strftime("%I:%M %p").lstrip("0")
                display_end = (current_slot + slot_duration).strftime("%I:%M %p").lstrip("0")
                
                slots.append({
                    "value": slot_start_str,
                    "display": f"{display_start} - {display_end}",
                    "end_time": slot_end_str
                })
            
            # Move to next slot (add 15-minute buffer)
            current_slot += slot_duration + timedelta(minutes=15)
        
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
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "message": f"Error: {str(e)}"})
def ensure_date_object(input_date):
    """Convert any date input to a date object"""
    if isinstance(input_date, date):
        return input_date
    elif isinstance(input_date, datetime):
        return input_date.date()
    elif isinstance(input_date, str):
        try:
            return datetime.strptime(input_date, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None
@app.post("/payment/verify")
async def verify_payment(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Legacy payment verification endpoint - redirects to new API"""
    try:
        data = await request.json()
        
        # Call the new API endpoint
        response = await verify_payment_api(request, db)
        
        # If it's a JSONResponse, return it as is
        if isinstance(response, JSONResponse):
            return response
        
        # Otherwise, return success
        return JSONResponse({
            "success": True,
            "message": "Payment verified"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False, 
            "message": f"Error: {str(e)}"
        })
# ============ ADMIN ROUTES ============

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    user_type: Optional[str] = None,
    page: int = 1,
    limit: int = 20
):
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Build user query
    user_query = db.query(User)
    
    if user_type and user_type != "all":
        user_query = user_query.filter(User.role == user_type)
    
    # Get total users count
    total_users = user_query.count()
    total_pages = (total_users + limit - 1) // limit if limit > 0 else 1
    
    # Apply pagination
    offset = (page - 1) * limit
    all_users = user_query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    
    # Get all mentors with user info
    all_mentors = db.query(Mentor).join(User).order_by(Mentor.created_at.desc()).all()
    
    # Existing pending mentors for approval
    pending_mentors = db.query(Mentor).filter(
        Mentor.verification_status == "pending"
    ).join(User).all()
    
    # Recent bookings
    recent_bookings = db.query(Booking).order_by(
        Booking.created_at.desc()
    ).limit(20).all()
    
    # Recent digital sales
    recent_digital_sales = db.query(Booking).filter(
        Booking.service.has(is_digital=True),
        Booking.payment_status == "paid"
    ).order_by(Booking.created_at.desc()).limit(10).all()
    
    # Calculate statistics
    total_users_count = db.query(User).count()
    total_mentors_count = db.query(Mentor).filter(
        Mentor.is_verified_by_admin == True
    ).count()
    total_bookings_count = db.query(Booking).count()
    total_digital_sales_count = db.query(Booking).filter(
        Booking.service.has(is_digital=True),
        Booking.payment_status == "paid"
    ).count()
    
    # Revenue calculations
    total_revenue = db.query(Booking).filter(
        Booking.payment_status == "paid"
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    digital_revenue = db.query(Booking).filter(
        Booking.service.has(is_digital=True),
        Booking.payment_status == "paid"
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    session_revenue = total_revenue - digital_revenue
    
    # Pending bookings
    pending_bookings_count = db.query(Booking).filter(
        Booking.payment_status == "pending"
    ).count()
    
    # Today's stats
    today = datetime.now().date()
    today_bookings = db.query(Booking).filter(
        func.date(Booking.created_at) == today
    ).count()
    
    today_revenue = db.query(Booking).filter(
        func.date(Booking.created_at) == today,
        Booking.payment_status == "paid"
    ).with_entities(func.sum(Booking.amount_paid)).scalar() or 0
    
    # User growth (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    new_users_30_days = db.query(User).filter(
        User.created_at >= thirty_days_ago
    ).count()
    
    new_mentors_30_days = db.query(Mentor).filter(
        Mentor.created_at >= thirty_days_ago
    ).count()
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "current_user": current_user,
        "all_users": all_users,
        "all_mentors": all_mentors,
        "pending_mentors": pending_mentors,
        "recent_bookings": recent_bookings,
        "recent_digital_sales": recent_digital_sales,
        "user_type": user_type,
        "page": page,
        "total_pages": total_pages,
        "total_users": total_users_count,
        "stats": {
            "total_users": total_users_count,
            "total_mentors": total_mentors_count,
            "total_bookings": total_bookings_count,
            "total_digital_sales": total_digital_sales_count,
            "total_revenue": total_revenue,
            "digital_revenue": digital_revenue,
            "session_revenue": session_revenue,
            "pending_bookings": pending_bookings_count,
            "today_bookings": today_bookings,
            "today_revenue": today_revenue,
            "new_users_30_days": new_users_30_days,
            "new_mentors_30_days": new_mentors_30_days
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
    booking_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Meeting page with Jitsi Meet integration"""
    booking = db.query(Booking).filter(
        Booking.id == booking_id,
        or_(
            Booking.learner_id == current_user.id,
            Booking.mentor_id == current_user.id
        )
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Check if meeting is confirmed (paid or free)
    if booking.payment_status not in ["paid", "free"] or booking.status != "confirmed":
        raise HTTPException(
            status_code=400, 
            detail="Meeting not confirmed yet. Please wait for payment confirmation."
        )
    
    # If meeting link doesn't exist, generate one
    if not booking.meeting_link:
        meeting_link, meeting_id = generate_meeting_link(booking.id, db)
    else:
        meeting_link = booking.meeting_link
        meeting_id = booking.meeting_id
    
    return templates.TemplateResponse("jitsi_meeting.html", {
        "request": request,
        "current_user": current_user,
        "booking": booking,
        "meeting_link": meeting_link,
        "meeting_id": meeting_id
    })
@app.post("/api/verify-payment")
async def verify_payment_api(request: Request, db: Session = Depends(get_db)):
    """Verify Razorpay payment - called from frontend"""
    try:
        data = await request.json()
        print(f"=== Payment Verification Started ===")
        print(f"Data received: {data}")
        
        razorpay_payment_id = data.get('razorpay_payment_id')
        razorpay_order_id = data.get('razorpay_order_id')
        razorpay_signature = data.get('razorpay_signature')
        booking_id = data.get('booking_id')
        
        if not all([razorpay_payment_id, razorpay_order_id, razorpay_signature, booking_id]):
            print("Missing required payment data")
            return JSONResponse({
                "success": False, 
                "message": "Missing required payment data"
            })
        
        # Verify payment signature
        try:
            params_dict = {
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            }
            print(f"Verifying signature for order: {razorpay_order_id}")
            
            razorpay_client.utility.verify_payment_signature(params_dict)
            print("✅ Payment signature verified successfully")
            
        except Exception as e:
            print(f"❌ Signature verification failed: {str(e)}")
            return JSONResponse({
                "success": False, 
                "message": f"Payment verification failed: {str(e)}"
            })
        
        # Find the booking
        booking = db.query(Booking).filter(Booking.id == booking_id).first()
        if not booking:
            print(f"❌ Booking not found: {booking_id}")
            return JSONResponse({
                "success": False, 
                "message": "Booking not found"
            })
        
        print(f"✅ Booking found: ID={booking.id}, Current Status={booking.status}, Payment Status={booking.payment_status}")
        
        # Check if already paid
        if booking.payment_status == "paid":
            print("⚠️ Payment already marked as paid")
            return JSONResponse({
                "success": True, 
                "message": "Payment already verified",
                "redirect_url": "/dashboard"
            })
        
        # Update booking with payment details
        booking.payment_status = "paid"
        booking.razorpay_payment_id = razorpay_payment_id
        
        # Check if this is a digital product
        service = db.query(Service).filter(Service.id == booking.service_id).first()
        is_digital = service.is_digital if service else False
        
        if is_digital:
            # For digital products, mark as completed immediately
            booking.status = "completed"
            print("✅ Digital product purchase completed")
        else:
            # For live sessions, generate meeting link
            try:
                meeting_link, meeting_id = generate_meeting_link(booking.id, db)
                print(f"✅ Meeting link generated: {meeting_link}")
            except Exception as e:
                print(f"⚠️ Error generating meeting link: {str(e)}")
                # Still mark as paid even if meeting link fails
                booking.status = "confirmed"
        
        # Create payment record
        try:
            payment = Payment(
                booking_id=booking.id,
                razorpay_order_id=razorpay_order_id,
                razorpay_payment_id=razorpay_payment_id,
                amount=booking.amount_paid,
                status="paid",
                payment_method="razorpay",
                created_at=datetime.utcnow()
            )
            db.add(payment)
            print("✅ Payment record created")
        except Exception as e:
            print(f"⚠️ Error creating payment record: {str(e)}")
        
        # Commit all changes
        db.commit()
        print("✅ Database changes committed")
        
        # Determine redirect URL
        redirect_url = "/dashboard"
        if is_digital and service:
            redirect_url = f"/digital-product/{service.id}"
        
        print(f"=== Payment Verification Complete ===")
        print(f"✅ Success! Redirecting to: {redirect_url}")
        
        return JSONResponse({
            "success": True, 
            "message": "Payment verified successfully!",
            "redirect_url": redirect_url,
            "is_digital": is_digital
        })
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error in verify_payment_api: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False, 
            "message": f"Payment verification failed: {str(e)}"
        })
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
    path = request.url.path
    # Check if it's a service URL pattern
    if re.match(r'^/[A-Za-z0-9_-]+/service/\d+$', path):
        return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
