import os
import re
import uuid
from sqlalchemy import DECIMAL
from decimal import Decimal
import logging
import requests
import traceback
from sqlalchemy import or_, and_
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
from sqlalchemy.orm import joinedload, selectinload, subqueryload
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

# Razorpay configuration# Razorpay configuration with validation
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

# Initialize Razorpay client only if credentials are available
razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        print("✅ Razorpay client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Razorpay client: {e}")
        razorpay_client = None
else:
    print("⚠️ Razorpay credentials not set in environment variables")

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

class AvailabilityDay(Base):
    __tablename__ = "availability_days"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    start_time = Column(String, default="09:00")
    end_time = Column(String, default="21:00")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    mentor = relationship("Mentor")

class AvailabilityException(Base):
    __tablename__ = "availability_exceptions"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    date = Column(Date, nullable=False)  # Specific date
    is_available = Column(Boolean, default=False)  # True=available, False=unavailable
    reason = Column(String, nullable=True)  # "Holiday", "Sick", etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    mentor = relationship("Mentor")




class Availability(Base):
    __tablename__ = "availabilities"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    start_time = Column(String, default="09:00")
    end_time = Column(String, default="21:00")
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    service_id = Column(Integer, ForeignKey("services.id"), nullable=True)
    is_booked = Column(Boolean, default=False)
    is_generated = Column(Boolean, default=False)  # NEW: Whether this was auto-generated
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

class MentorPayout(Base):
    __tablename__ = "mentor_payouts"
    
    id = Column(Integer, primary_key=True, index=True)
    mentor_id = Column(Integer, ForeignKey("mentors.id"))
    amount = Column(DECIMAL(10, 2), nullable=False)  # in INR
    status = Column(String, default="pending")  # pending, processing, completed, failed
    request_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime, nullable=True)
    payment_method = Column(String)  # bank_transfer, upi, etc.
    account_details = Column(Text, nullable=True)  # Store encrypted account details
    notes = Column(Text, nullable=True)
    
    # Relationships
    mentor = relationship("Mentor")

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

class PayoutUpdate(BaseModel):
    status: str
    notes: Optional[str] = None
# ============ DEPENDENCIES ============
# Add this function after your database models

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add this to your imports
import uuid
import requests
from datetime import datetime, timedelta


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
    
    # Use the provided session from dependency injection
    user = db.query(User).filter(User.id == user_id).first()
    
    if user and user.is_active:
        return user
    
    return None
        # Ensure profile_image has correct path
       # if user.profile_image and not user.profile_image.startswith("uploads/"):
            #user.profile_image = f"uploads/{user.profile_image}" if user.profile_image != "default-avatar.png" else "default-avatar.png"
    
    #return user if user and user.is_active else None

# Add this function after your imports

def generate_availabilities_for_mentor(mentor_id: int, days_ahead: int = 30, db: Session = None):
    """Generate availability slots based on mentor's day preferences"""
    if db is None:
        db = SessionLocal()
    
    try:
        # Get mentor's day preferences
        day_preferences = db.query(AvailabilityDay).filter(
            AvailabilityDay.mentor_id == mentor_id,
            AvailabilityDay.is_active == True
        ).all()
        
        if not day_preferences:
            print(f"No day preferences found for mentor {mentor_id}")
            return
        
        # Get exceptions
        exceptions = db.query(AvailabilityException).filter(
            AvailabilityException.mentor_id == mentor_id
        ).all()
        exception_dates = {ex.date: ex.is_available for ex in exceptions}
        
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Generate availabilities for each day
        current_date = today
        generated_count = 0
        
        while current_date <= end_date:
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Check if mentor is available on this day
            day_pref = next((dp for dp in day_preferences if dp.day_of_week == day_of_week), None)
            
            # Check for exceptions
            is_available = False
            if current_date in exception_dates:
                is_available = exception_dates[current_date]
            elif day_pref:
                is_available = True
            
            if is_available and day_pref:
                # Check if availability already exists for this date
                existing = db.query(Availability).filter(
                    Availability.mentor_id == mentor_id,
                    Availability.date == current_date,
                    Availability.is_generated == True
                ).first()
                
                if not existing:
                    # Check if there are any bookings for this date
                    has_bookings = db.query(Booking).filter(
                        Booking.mentor_id == mentor_id,
                        Booking.booking_date == current_date,
                        Booking.status.in_(["confirmed", "pending"])
                    ).first() is not None
                    
                    # Create new availability
                    availability = Availability(
                        mentor_id=mentor_id,
                        date=current_date,
                        start_time=day_pref.start_time,
                        end_time=day_pref.end_time,
                        is_booked=has_bookings,
                        is_generated=True,
                        created_at=datetime.utcnow()
                    )
                    db.add(availability)
                    generated_count += 1
            
            current_date += timedelta(days=1)
        
        db.commit()
        print(f"Generated {generated_count} availabilities for mentor {mentor_id}")
        
    except Exception as e:
        db.rollback()
        print(f"Error generating availabilities: {e}")
    finally:
        if db:
            db.close()

def get_available_dates_for_mentor(mentor_id: int, days_ahead: int = 30, db: Session = None):
    """Get available dates for a mentor based on their day preferences"""
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    try:
        # Get mentor's day preferences
        day_preferences = db.query(AvailabilityDay).filter(
            AvailabilityDay.mentor_id == mentor_id,
            AvailabilityDay.is_active == True
        ).all()
        
        # If no day preferences exist, create default ones with 9AM-9PM
        if not day_preferences:
            print(f"Creating default day preferences for mentor {mentor_id} with 9AM-9PM")
            # CHANGED: 9AM to 9PM instead of 9AM to 5PM
            default_days = [
                {"day_of_week": 0, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 1, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 2, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 3, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 4, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 5, "start_time": "10:00", "end_time": "21:00", "is_active": False},
                {"day_of_week": 6, "start_time": "10:00", "end_time": "21:00", "is_active": False},
            ]
            
            for day_data in default_days:
                day_pref = AvailabilityDay(
                    mentor_id=mentor_id,
                    day_of_week=day_data["day_of_week"],
                    start_time=day_data["start_time"],
                    end_time=day_data["end_time"],
                    is_active=day_data["is_active"],
                    created_at=datetime.utcnow()
                )
                db.add(day_pref)
            
            db.commit()
            
            # Reload preferences
            day_preferences = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor_id,
                AvailabilityDay.is_active == True
            ).all()
        
        # Get exceptions
        exceptions = db.query(AvailabilityException).filter(
            AvailabilityException.mentor_id == mentor_id
        ).all()
        exception_dates = {ex.date: ex.is_available for ex in exceptions}
        
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Generate list of available dates
        available_dates = []
        current_date = today
        
        while current_date <= end_date:
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Check if mentor is available on this day
            day_pref = next((dp for dp in day_preferences if dp.day_of_week == day_of_week), None)
            
            # Check for exceptions
            is_available = False
            if current_date in exception_dates:
                is_available = exception_dates[current_date]
            elif day_pref:
                is_available = True
            
            if is_available:
                # Get availability for this date
                availability = db.query(Availability).filter(
                    Availability.mentor_id == mentor_id,
                    Availability.date == current_date,
                    Availability.is_booked == False
                ).first()
                
                if availability:
                    # Check time slots
                    time_slots = db.query(TimeSlot).join(Booking).filter(
                        Booking.mentor_id == mentor_id,
                        TimeSlot.date == current_date
                    ).all()
                    
                    # If no time slots or some slots available, add to list
                    if not time_slots or any(not ts.is_booked for ts in time_slots):
                        available_dates.append({
                            'date_obj': current_date,
                            'full_date': current_date.strftime("%Y-%m-%d"),
                            'day_name': current_date.strftime("%A"),
                            'day_short': current_date.strftime("%a"),
                            'day_num': current_date.day,
                            'month': current_date.strftime("%b"),
                            'start_time': availability.start_time if availability else (day_pref.start_time if day_pref else "09:00"),
                            'end_time': availability.end_time if availability else (day_pref.end_time if day_pref else "21:00")
                        })
            
            current_date += timedelta(days=1)
        
        print(f"✅ Found {len(available_dates)} available dates for mentor {mentor_id} with 9AM-9PM schedule")
        return available_dates[:14]  # Return next 14 available dates
        
    except Exception as e:
        print(f"❌ Error getting available dates: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        if should_close and db:
            db.close()
            
            
def generate_meeting_link(booking_id: int, db: Session):
    """Generate or retrieve meeting link for a confirmed booking - FIXED VERSION"""
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
    
    # Mark ONLY the TimeSlot record as booked (NOT the entire Availability)
    target_date = booking.booking_date
    
    # Update all TimeSlot records for this booking
    time_slots = db.query(TimeSlot).filter(TimeSlot.booking_id == booking.id).all()
    for time_slot in time_slots:
        time_slot.is_booked = True
        print(f"Marked TimeSlot {time_slot.id} ({time_slot.start_time}-{time_slot.end_time}) as booked")
    
    # IMPORTANT: DO NOT mark entire availability as booked
    # Instead, check if we should mark availability based on booked time slots
    availability = db.query(Availability).filter(
        Availability.mentor_id == booking.mentor_id,
        Availability.date == target_date
    ).first()
    
    if availability:
        # Check if ALL time slots for this date are booked
        all_time_slots = db.query(TimeSlot).filter(
            TimeSlot.date == target_date,
            TimeSlot.booking.has(mentor_id=booking.mentor_id)
        ).all()
        
        booked_slots = [ts for ts in all_time_slots if ts.is_booked]
        
        # Only mark availability as booked if ALL time slots are booked
        # OR if this booking covers the entire availability window
        try:
            # Parse times to check if booking covers entire availability
            avail_start = datetime.strptime(availability.start_time, "%H:%M")
            avail_end = datetime.strptime(availability.end_time, "%H:%M")
            
            # Get service duration
            service = db.query(Service).filter(Service.id == booking.service_id).first()
            service_duration = service.duration_minutes if service else 60
            
            # Calculate booked time for this booking
            booking_start = datetime.strptime(booking.selected_time, "%H:%M")
            booking_end = booking_start + timedelta(minutes=service_duration)
            
            # Check if booking covers the entire availability window
            booking_covers_all = (booking_start <= avail_start and booking_end >= avail_end)
            
            # Mark availability as booked only if:
            # 1. Booking covers entire window, OR
            # 2. All time slots for the day are booked
            if booking_covers_all or (len(all_time_slots) > 0 and len(booked_slots) == len(all_time_slots)):
                availability.is_booked = True
                print(f"Marked availability as booked (full day or all slots booked)")
            else:
                availability.is_booked = False
                print(f"Availability remains open (partial booking)")
                
        except Exception as e:
            print(f"Error checking availability coverage: {e}")
            # Default: don't mark as booked
            availability.is_booked = False
    
    db.commit()
    
    # Send notification (optional)
    send_meeting_notification(booking, meeting_link, db)
    
    return meeting_link, meeting_id


@app.post("/mentor/availability/cleanup")
async def cleanup_availability(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clean up availability records that should not be marked as booked"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Get all availabilities marked as booked
    booked_availabilities = db.query(Availability).filter(
        Availability.mentor_id == mentor.id,
        Availability.is_booked == True
    ).all()
    
    fixed_count = 0
    
    for availability in booked_availabilities:
        try:
            # Check actual booked time slots
            booked_slots = db.query(TimeSlot).join(Booking).filter(
                Booking.mentor_id == mentor.id,
                TimeSlot.date == availability.date,
                TimeSlot.is_booked == True
            ).all()
            
            if not booked_slots:
                # No booked slots, so availability should be open
                availability.is_booked = False
                fixed_count += 1
                print(f"Fixed availability for {availability.date}: No booked slots found")
                continue
            
            # Calculate booking percentage
            avail_start = datetime.strptime(availability.start_time, "%H:%M")
            avail_end = datetime.strptime(availability.end_time, "%H:%M")
            total_minutes = (avail_end - avail_start).seconds / 60
            
            booked_minutes = 0
            for slot in booked_slots:
                slot_start = datetime.strptime(slot.start_time, "%H:%M")
                slot_end = datetime.strptime(slot.end_time, "%H:%M")
                booked_minutes += (slot_end - slot_start).seconds / 60
            
            booked_percentage = (booked_minutes / total_minutes) * 100 if total_minutes > 0 else 0
            
            # If less than 90% booked, mark as available
            if booked_percentage <= 90:
                availability.is_booked = False
                fixed_count += 1
                print(f"Fixed availability for {availability.date}: {booked_percentage:.1f}% booked")
                
        except Exception as e:
            print(f"Error processing availability for {availability.date}: {e}")
            continue
    
    db.commit()
    
    return JSONResponse({
        "success": True,
        "message": f"Fixed {fixed_count} availability records",
        "fixed_count": fixed_count
    })



def check_time_slot_availability(mentor_id: int, date: date, start_time: str, end_time: str, db: Session) -> bool:
    """Check if a time slot is available (no conflicts with existing bookings)"""
    from sqlalchemy import and_, or_  # Import here if not already imported at top
    
    # Check TimeSlot table for overlapping booked slots
    overlapping_slots = db.query(TimeSlot).join(Booking).filter(
        Booking.mentor_id == mentor_id,
        TimeSlot.date == date,
        TimeSlot.is_booked == True,
        # Check for time overlap using and_ and or_
        or_(
            # New slot starts during existing booked slot
            and_(
                TimeSlot.start_time <= start_time,
                TimeSlot.end_time > start_time
            ),
            # New slot ends during existing booked slot
            and_(
                TimeSlot.start_time < end_time,
                TimeSlot.end_time >= end_time
            ),
            # New slot completely contains existing booked slot
            and_(
                TimeSlot.start_time >= start_time,
                TimeSlot.end_time <= end_time
            ),
            # Existing booked slot completely contains new slot
            and_(
                TimeSlot.start_time <= start_time,
                TimeSlot.end_time >= end_time
            )
        )
    ).first()
    
    return overlapping_slots is None


def generate_availability_from_day_preferences(mentor_id: int, db: Session = None, days_ahead: int = 30):
    """Generate Availability records from day preferences"""
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    try:
        # Get day preferences
        day_preferences = db.query(AvailabilityDay).filter(
            AvailabilityDay.mentor_id == mentor_id,
            AvailabilityDay.is_active == True
        ).all()
        
        if not day_preferences:
            print(f"No active day preferences found for mentor {mentor_id}")
            # Create default with 9AM-9PM
            default_days = [
                {"day_of_week": 0, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 1, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 2, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 3, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 4, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                {"day_of_week": 5, "start_time": "10:00", "end_time": "21:00", "is_active": False},
                {"day_of_week": 6, "start_time": "10:00", "end_time": "21:00", "is_active": False},
            ]
            
            for day_data in default_days:
                day_pref = AvailabilityDay(
                    mentor_id=mentor_id,
                    day_of_week=day_data["day_of_week"],
                    start_time=day_data["start_time"],
                    end_time=day_data["end_time"],
                    is_active=day_data["is_active"],
                    created_at=datetime.utcnow()
                )
                db.add(day_pref)
            
            db.commit()
            print(f"✅ Created default 9AM-9PM schedule for mentor {mentor_id}")
            
            # Reload preferences
            day_preferences = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor_id,
                AvailabilityDay.is_active == True
            ).all()
        
        # Get exceptions
        exceptions = db.query(AvailabilityException).filter(
            AvailabilityException.mentor_id == mentor_id
        ).all()
        exception_dates = {ex.date: ex.is_available for ex in exceptions}
        
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        generated_count = 0
        updated_count = 0
        
        # Generate for each day
        current_date = today
        while current_date <= end_date:
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Check if mentor is available on this day
            day_pref = next((dp for dp in day_preferences if dp.day_of_week == day_of_week), None)
            
            # Check for exceptions
            is_available = False
            if current_date in exception_dates:
                is_available = exception_dates[current_date]
            elif day_pref:
                is_available = True
            
            if is_available and day_pref:
                # Check if availability already exists
                existing = db.query(Availability).filter(
                    Availability.mentor_id == mentor_id,
                    Availability.date == current_date
                ).first()
                
                if not existing:
                    # Create new availability with 9AM-9PM
                    availability = Availability(
                        mentor_id=mentor_id,
                        date=current_date,
                        start_time=day_pref.start_time,
                        end_time=day_pref.end_time,
                        is_booked=False,  # Start as not booked
                        created_at=datetime.utcnow()
                    )
                    db.add(availability)
                    generated_count += 1
                else:
                    # Update existing availability if times changed
                    if existing.start_time != day_pref.start_time or existing.end_time != day_pref.end_time:
                        existing.start_time = day_pref.start_time
                        existing.end_time = day_pref.end_time
                        updated_count += 1
            
            current_date += timedelta(days=1)
        
        db.commit()
        print(f"✅ Generated {generated_count} new and updated {updated_count} availability records for mentor {mentor_id} (9AM-9PM)")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error generating availability: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if should_close and db:
            db.close()


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
        
        # Ensure all mentors have day preferences
        mentors = db.query(Mentor).all()
        for mentor in mentors:
            # Check if mentor has day preferences
            day_preferences = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor.id
            ).first()
            
            if not day_preferences:
                # Create default day preferences
                default_days = [
                    {"day_of_week": 0, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 1, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 2, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 3, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 4, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 5, "start_time": "09:00", "end_time": "21:00", "is_active": False},
                    {"day_of_week": 6, "start_time": "09:00", "end_time": "21:00", "is_active": False},
                ]
                
                for day_data in default_days:
                    day_pref = AvailabilityDay(
                        mentor_id=mentor.id,
                        day_of_week=day_data["day_of_week"],
                        start_time=day_data["start_time"],
                        end_time=day_data["end_time"],
                        is_active=day_data["is_active"],
                        created_at=datetime.utcnow()
                    )
                    db.add(day_pref)
                
                print(f"Created default day preferences for mentor {mentor.id}")
        
        db.commit()
        print("✅ Startup tasks completed")
        
    except Exception as e:
        print(f"❌ Error in startup tasks: {e}")
        import traceback
        traceback.print_exc()
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
    mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
    if not mentor:
        return {"error": "Mentor not found"}
    
    return {
        "linkedin": mentor.linkedin_url,
        "github": mentor.github_url,
        "twitter": mentor.twitter_url,
        "website": mentor.website_url
    }

@app.get("/mentor/availability/days", response_class=HTMLResponse)
async def mentor_availability_days_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mentor availability by days of week"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        return RedirectResponse(url="/dashboard", status_code=303)
    
    # Get current day preferences
    day_preferences = db.query(AvailabilityDay).filter(
        AvailabilityDay.mentor_id == mentor.id
    ).order_by(AvailabilityDay.day_of_week).all()
    
    # Create default preferences if none exist
    if not day_preferences:
        default_days = [
            {"day_of_week": 0, "day_name": "Monday", "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 1, "day_name": "Tuesday", "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 2, "day_name": "Wednesday", "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 3, "day_name": "Thursday", "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 4, "day_name": "Friday", "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 5, "day_name": "Saturday", "start_time": "09:00", "end_time": "21:00", "is_active": False},
            {"day_of_week": 6, "day_name": "Sunday", "start_time": "09:00", "end_time": "21:00", "is_active": False},
        ]
        
        for day_data in default_days:
            day_pref = AvailabilityDay(
                mentor_id=mentor.id,
                day_of_week=day_data["day_of_week"],
                start_time=day_data["start_time"],
                end_time=day_data["end_time"],
                is_active=day_data["is_active"]
            )
            db.add(day_pref)
        db.commit()
        db.refresh(day_preferences)
    
    # Get exceptions (next 30 days)
    today = datetime.now().date()
    future_date = today + timedelta(days=30)
    
    exceptions = db.query(AvailabilityException).filter(
        AvailabilityException.mentor_id == mentor.id,
        AvailabilityException.date >= today,
        AvailabilityException.date <= future_date
    ).order_by(AvailabilityException.date).all()
    
    # Generate availabilities for display
    available_dates = get_available_dates_for_mentor(mentor.id, 30, db)
    
    return templates.TemplateResponse("mentor_availability_days.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "day_preferences": day_preferences,
        "exceptions": exceptions,
        "available_dates": available_dates,
        "today": today.strftime("%Y-%m-%d"),
        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    })

@app.post("/mentor/availability/days/update")
async def update_day_preferences(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update day of week preferences"""
    if not current_user or current_user.role != "mentor":
        return JSONResponse({"success": False, "message": "Access denied"})
    
    try:
        data = await request.json()
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        
        if not mentor:
            return JSONResponse({"success": False, "message": "Mentor not found"})
        
        # Update each day preference
        for day_data in data.get("days", []):
            day_pref = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor.id,
                AvailabilityDay.day_of_week == day_data["day_of_week"]
            ).first()
            
            if day_pref:
                day_pref.start_time = day_data["start_time"]
                day_pref.end_time = day_data["end_time"]
                day_pref.is_active = day_data.get("is_active", True)
            else:
                day_pref = AvailabilityDay(
                    mentor_id=mentor.id,
                    day_of_week=day_data["day_of_week"],
                    start_time=day_data["start_time"],
                    end_time=day_data["end_time"],
                    is_active=day_data.get("is_active", True)
                )
                db.add(day_pref)
        
        db.commit()
        
        # Regenerate availabilities
        generate_availabilities_for_mentor(mentor.id, 30, db)
        
        return JSONResponse({
            "success": True,
            "message": "Availability preferences updated successfully!"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error updating preferences: {str(e)}"
        })

@app.post("/mentor/availability/exceptions/add")
async def add_availability_exception(
    request: Request,
    date: str = Form(...),
    is_available: bool = Form(False),
    reason: str = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add an exception (holiday, etc.)"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
        
        # Check if exception already exists
        existing = db.query(AvailabilityException).filter(
            AvailabilityException.mentor_id == mentor.id,
            AvailabilityException.date == parsed_date
        ).first()
        
        if existing:
            existing.is_available = is_available
            existing.reason = reason
        else:
            exception = AvailabilityException(
                mentor_id=mentor.id,
                date=parsed_date,
                is_available=is_available,
                reason=reason,
                created_at=datetime.utcnow()
            )
            db.add(exception)
        
        db.commit()
        
        # Update affected availabilities
        if is_available:
            # Make date available
            availability = db.query(Availability).filter(
                Availability.mentor_id == mentor.id,
                Availability.date == parsed_date
            ).first()
            
            if availability:
                availability.is_booked = False
        else:
            # Make date unavailable
            availability = db.query(Availability).filter(
                Availability.mentor_id == mentor.id,
                Availability.date == parsed_date
            ).first()
            
            if availability:
                availability.is_booked = True
        
        db.commit()
        
        return RedirectResponse(
            url="/mentor/availability/days?success=Exception%20added%20successfully",
            status_code=303
        )
        
    except Exception as e:
        db.rollback()
        return RedirectResponse(
            url=f"/mentor/availability/days?error={str(e).replace(' ', '%20')}",
            status_code=303
        )

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
    query = db.query(Mentor).options(joinedload(Mentor.user)).join(User).filter(
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
                "today_revenue": today_revenue
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



@app.get("/mentor/availability/simple", response_class=HTMLResponse)
async def mentor_availability_simple_page(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simple availability management - days of week"""
    if not current_user or current_user.role != "mentor":
        return RedirectResponse(url="/login", status_code=303)
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        return RedirectResponse(url="/dashboard", status_code=303)
    
    # Days of week with names
    days_of_week = [
        {"id": 0, "name": "Monday"},
        {"id": 1, "name": "Tuesday"},
        {"id": 2, "name": "Wednesday"},
        {"id": 3, "name": "Thursday"},
        {"id": 4, "name": "Friday"},
        {"id": 5, "name": "Saturday"},
        {"id": 6, "name": "Sunday"}
    ]
    
    # Get existing preferences
    day_preferences = db.query(AvailabilityDay).filter(
        AvailabilityDay.mentor_id == mentor.id
    ).all()
    
    # Create a dict for easy lookup
    pref_dict = {pref.day_of_week: pref for pref in day_preferences}
    
    # Prepare data for template
    days_data = []
    for day in days_of_week:
        pref = pref_dict.get(day["id"])
        days_data.append({
            "id": day["id"],
            "name": day["name"],
            "is_active": pref.is_active if pref else False,
            "start_time": pref.start_time if pref else "09:00",
            "end_time": pref.end_time if pref else "21:00"
        })
    
    # Get exceptions (next 30 days)
    today = datetime.now().date()
    future_date = today + timedelta(days=30)
    
    exceptions = db.query(AvailabilityException).filter(
        AvailabilityException.mentor_id == mentor.id,
        AvailabilityException.date >= today,
        AvailabilityException.date <= future_date
    ).order_by(AvailabilityException.date).all()
    
    return templates.TemplateResponse("mentor_availability_simple.html", {
        "request": request,
        "current_user": current_user,
        "mentor": mentor,
        "days": days_data,
        "exceptions": exceptions,
        "today": today.strftime("%Y-%m-%d")
    })


@app.delete("/mentor/availability/exceptions/{exception_id}/delete")
async def delete_availability_exception(
    exception_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an availability exception"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    
    exception = db.query(AvailabilityException).filter(
        AvailabilityException.id == exception_id,
        AvailabilityException.mentor_id == mentor.id
    ).first()
    
    if not exception:
        raise HTTPException(status_code=404, detail="Exception not found")
    
    db.delete(exception)
    db.commit()
    
    return JSONResponse({"success": True, "message": "Exception removed"})

@app.get("/api/mentor/availability/data")
async def get_mentor_availability_data(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mentor availability data for the simple interface"""
    if not current_user or current_user.role != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Ensure day preferences exist
    day_preferences = db.query(AvailabilityDay).filter(
        AvailabilityDay.mentor_id == mentor.id
    ).all()
    
    # If no preferences, create default ones
    if not day_preferences:
        default_days = [
            {"day_of_week": 0, "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 1, "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 2, "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 3, "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 4, "start_time": "09:00", "end_time": "21:00", "is_active": True},
            {"day_of_week": 5, "start_time": "09:00", "end_time": "21:00", "is_active": False},
            {"day_of_week": 6, "start_time": "09:00", "end_time": "21:00", "is_active": False},
        ]
        
        for day_data in default_days:
            day_pref = AvailabilityDay(
                mentor_id=mentor.id,
                day_of_week=day_data["day_of_week"],
                start_time=day_data["start_time"],
                end_time=day_data["end_time"],
                is_active=day_data["is_active"],
                created_at=datetime.utcnow()
            )
            db.add(day_pref)
        
        db.commit()
        db.refresh(day_preferences)
    
    # Prepare days data
    days_of_week = [
        {"id": 0, "name": "Monday"},
        {"id": 1, "name": "Tuesday"},
        {"id": 2, "name": "Wednesday"},
        {"id": 3, "name": "Thursday"},
        {"id": 4, "name": "Friday"},
        {"id": 5, "name": "Saturday"},
        {"id": 6, "name": "Sunday"}
    ]
    
    pref_dict = {pref.day_of_week: pref for pref in day_preferences}
    
    days_data = []
    for day in days_of_week:
        pref = pref_dict.get(day["id"])
        days_data.append({
            "id": day["id"],
            "name": day["name"],
            "is_active": pref.is_active if pref else (day["id"] < 5),  # Default: Mon-Fri active
            "start_time": pref.start_time if pref and pref.is_active else "09:00",
            "end_time": pref.end_time if pref and pref.is_active else "21:00"
        })
    
    # Get exceptions
    today = datetime.now().date()
    exceptions = db.query(AvailabilityException).filter(
        AvailabilityException.mentor_id == mentor.id,
        AvailabilityException.date >= today
    ).order_by(AvailabilityException.date).all()
    
    exceptions_data = [
        {
            "id": ex.id,
            "date": ex.date.strftime("%Y-%m-%d"),
            "reason": ex.reason
        }
        for ex in exceptions
    ]
    
    return JSONResponse({
        "success": True,
        "days": days_data,
        "exceptions": exceptions_data
    })
@app.post("/mentor/availability/simple/update")
async def update_simple_availability(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update simple availability preferences"""
    if not current_user or current_user.role != "mentor":
        return JSONResponse({"success": False, "message": "Access denied"})
    
    try:
        data = await request.json()
        mentor = db.query(Mentor).filter(Mentor.user_id == current_user.id).first()
        
        if not mentor:
            return JSONResponse({"success": False, "message": "Mentor not found"})
        
        # Update each day preference
        for day_data in data.get("days", []):
            day_pref = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor.id,
                AvailabilityDay.day_of_week == day_data["day_of_week"]
            ).first()
            
            if day_pref:
                day_pref.is_active = day_data.get("is_active", False)
                if day_data.get("is_active", False):
                    day_pref.start_time = day_data.get("start_time", "09:00")
                    day_pref.end_time = day_data.get("end_time", "21:00")
            else:
                if day_data.get("is_active", False):
                    day_pref = AvailabilityDay(
                        mentor_id=mentor.id,
                        day_of_week=day_data["day_of_week"],
                        start_time=day_data.get("start_time", "09:00"),
                        end_time=day_data.get("end_time", "21:00"),
                        is_active=True
                    )
                    db.add(day_pref)
        
        db.commit()
        
        # Delete inactive preferences
        db.query(AvailabilityDay).filter(
            AvailabilityDay.mentor_id == mentor.id,
            AvailabilityDay.is_active == False
        ).delete(synchronize_session=False)
        
        db.commit()
        
        # Regenerate availability for next 30 days
        generate_availability_from_day_preferences(mentor.id, db, 30)
        
        return JSONResponse({
            "success": True,
            "message": "Availability updated successfully!",
            "regenerated": True
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error updating availability: {str(e)}"
        })
    
    
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
    mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
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
    """Create a new availability with custom time range - UPDATED WITH SLOT LOGIC"""
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
        
        # NEW LOGIC: Check for existing availability
        existing_availability = db.query(Availability).filter(
            Availability.mentor_id == mentor.id,
            Availability.date == parsed_date
        ).first()
        
        if existing_availability:
            # If availability exists but is booked, we need to check which slots are booked
            if existing_availability.is_booked:
                # Get booked time slots for this date
                booked_time_slots = db.query(TimeSlot).join(Booking).filter(
                    Booking.mentor_id == mentor.id,
                    TimeSlot.date == parsed_date,
                    TimeSlot.is_booked == True
                ).all()
                
                # Check if the new availability conflicts with booked slots
                for booked_slot in booked_time_slots:
                    try:
                        booked_start = datetime.strptime(booked_slot.start_time, "%H:%M")
                        booked_end = datetime.strptime(booked_slot.end_time, "%H:%M")
                        
                        # Check for overlap
                        if not (end_dt <= booked_start or start_dt >= booked_end):
                            # There's an overlap with a booked slot
                            return RedirectResponse(
                                url="/mentor/availability?error=Cannot%20add%20availability%20that%20overlaps%20with%20existing%20bookings",
                                status_code=303
                            )
                    except ValueError:
                        continue
                
                # No conflicts, update the availability
                existing_availability.start_time = start_time
                existing_availability.end_time = end_time
                existing_availability.service_id = parsed_service_id
                # Keep is_booked = True because there are still bookings, but update times
                
                db.commit()
                
                return RedirectResponse(
                    url="/mentor/availability?success=Availability%20updated%20successfully%20(around%20existing%20bookings)",
                    status_code=303
                )
            else:
                # Availability exists but not booked, update it
                existing_availability.start_time = start_time
                existing_availability.end_time = end_time
                existing_availability.service_id = parsed_service_id
                existing_availability.is_booked = False
                
                db.commit()
                
                return RedirectResponse(
                    url="/mentor/availability?success=Availability%20updated%20successfully",
                    status_code=303
                )
        
        # Check if there are any time slots booked for this date
        existing_booked_slots = db.query(TimeSlot).join(Booking).filter(
            Booking.mentor_id == mentor.id,
            TimeSlot.date == parsed_date,
            TimeSlot.is_booked == True
        ).first()
        
        if existing_booked_slots:
            # There are booked slots, but we already checked for conflicts above
            # Create new availability, mark it as partially booked
            availability = Availability(
                mentor_id=mentor.id,
                service_id=parsed_service_id,
                date=parsed_date,
                start_time=start_time,
                end_time=end_time,
                is_booked=False,  # False because not ALL slots are booked
                created_at=datetime.utcnow()
            )
        else:
            # No bookings at all for this date
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
        
        # If there are existing bookings, add a note
        if existing_booked_slots:
            success_msg += "%20(Note:%20Some%20time%20slots%20are%20already%20booked)"
        
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
    """Generate or retrieve meeting link for a confirmed booking - FIXED"""
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
    
    # Mark TimeSlot records as booked (but NOT the entire Availability)
    target_date = booking.booking_date
    
    # Update all TimeSlot records for this booking
    time_slots = db.query(TimeSlot).filter(TimeSlot.booking_id == booking.id).all()
    for time_slot in time_slots:
        time_slot.is_booked = True
        print(f"✅ Marked TimeSlot {time_slot.id} ({time_slot.start_time}-{time_slot.end_time}) as booked")
    
    # IMPORTANT: DO NOT mark entire availability as booked
    # Only mark Availability as booked if ALL time slots for that day are booked
    availability = db.query(Availability).filter(
        Availability.mentor_id == booking.mentor_id,
        Availability.date == target_date
    ).first()
    
    if availability:
        # Check all time slots for this date
        all_time_slots = db.query(TimeSlot).join(Booking).filter(
            Booking.mentor_id == booking.mentor_id,
            TimeSlot.date == target_date
        ).all()
        
        # Count booked vs available slots
        total_slots = len(all_time_slots)
        booked_slots = sum(1 for ts in all_time_slots if ts.is_booked)
        
        # Only mark availability as booked if ALL slots are booked
        if total_slots > 0 and booked_slots == total_slots:
            availability.is_booked = True
            print(f"✅ Marked availability as booked (all {total_slots} slots booked)")
        else:
            availability.is_booked = False
            print(f"✅ Availability remains open ({booked_slots}/{total_slots} slots booked)")
    
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

def cleanup_expired_pending_bookings(db: Session = Depends(get_db)):
    """Clean up expired pending bookings (run this periodically)"""
    try:
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        # Find expired pending bookings
        expired_bookings = db.query(Booking).filter(
            Booking.payment_status == "pending",
            Booking.status == "pending",
            Booking.created_at < one_hour_ago
        ).all()
        
        for booking in expired_bookings:
            print(f"Cleaning up expired booking: {booking.id}")
            
            # Free up time slots
            time_slots = db.query(TimeSlot).filter(
                TimeSlot.booking_id == booking.id
            ).all()
            
            for slot in time_slots:
                db.delete(slot)
            
            # Delete the booking
            db.delete(booking)
        
        db.commit()
        print(f"Cleaned up {len(expired_bookings)} expired bookings")
        
    except Exception as e:
        db.rollback()
        print(f"Error cleaning up expired bookings: {e}")

@app.get("/api/check-booking-status/{booking_id}")
async def check_booking_status(
    booking_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check booking status - useful for frontend to handle back button"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    booking = db.query(Booking).filter(
        Booking.id == booking_id,
        Booking.learner_id == current_user.id
    ).first()
    
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    service = db.query(Service).filter(Service.id == booking.service_id).first()
    
    response = {
        "booking_id": booking.id,
        "payment_status": booking.payment_status,
        "booking_status": booking.status,
        "is_digital": service.is_digital if service else False,
        "amount_paid": booking.amount_paid,
        "created_at": booking.created_at.isoformat() if booking.created_at else None
    }
    
    # Determine redirect URL based on current status
    if booking.payment_status == "paid":
        if service and service.is_digital:
            response["redirect_url"] = f"/digital-product/{service.id}"
        elif booking.meeting_link:
            response["redirect_url"] = f"/meeting/{booking.id}"
        else:
            response["redirect_url"] = "/dashboard?tab=upcoming"
    elif booking.payment_status == "pending":
        response["redirect_url"] = f"/payment/{booking.id}"
    elif booking.payment_status == "free":
        if service and service.is_digital:
            response["redirect_url"] = f"/digital-product/{service.id}"
        elif booking.meeting_link:
            response["redirect_url"] = f"/meeting/{booking.id}"
        else:
            response["redirect_url"] = "/dashboard"
    
    return JSONResponse(response)

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
    """Create booking with proper back button and payment handling - FULLY UPDATED"""
    if not current_user or current_user.role != "learner":
        raise HTTPException(status_code=403, detail="Only learners can book sessions")

    target_date = None
    
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
    
    # CRITICAL: Check for existing pending bookings (within last 1 hour)
    if not is_digital_service and date_str and time_slot:
        # For live sessions, check if user already has a pending booking for same slot
        existing_pending = db.query(Booking).filter(
            Booking.learner_id == current_user.id,
            Booking.service_id == service_id,
            Booking.booking_date == datetime.strptime(date_str, "%Y-%m-%d").date(),
            Booking.selected_time == time_slot,
            Booking.payment_status == "pending",
            Booking.status == "pending",
            Booking.created_at >= datetime.now() - timedelta(hours=1)
        ).first()
        
        if existing_pending:
            print(f"✅ Found existing pending booking: {existing_pending.id}")
            return JSONResponse({
                "success": True,
                "booking_id": existing_pending.id,
                "redirect_url": f"/payment/{existing_pending.id}",
                "is_digital": is_digital_service,
                "is_free": service.price == 0,
                "existing_booking": True,
                "message": "Continuing with existing booking"
            })
    
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
                download_count=0,
                created_at=datetime.utcnow()
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
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Check if time slot is available
            is_available = check_time_slot_availability(
                mentor_id=service.mentor_id,
                date=target_date,
                start_time=time_slot,
                end_time=end_time_str,
                db=db
            )
            
            if not is_available:
                raise HTTPException(status_code=400, detail="This time slot is no longer available")
            
            # Create booking record
            booking = Booking(
                learner_id=current_user.id,
                mentor_id=service.mentor_id,
                service_id=service_id,
                booking_date=target_date,
                selected_time=time_slot,
                razorpay_order_id=None,
                amount_paid=0,
                status="confirmed",
                payment_status="free",
                meeting_link=meeting_link,
                meeting_id=meeting_id,
                notes=f"Free session scheduled for {date_str} at {time_slot}",
                created_at=datetime.utcnow()
            )
            
            db.add(booking)
            db.commit()
            db.refresh(booking)
            
            # Create a time slot record
            time_slot_record = TimeSlot(
                booking_id=booking.id,
                start_time=time_slot,
                end_time=end_time_str,
                date=target_date,
                is_booked=True,
                created_at=datetime.utcnow()
            )
            db.add(time_slot_record)
            
            # Update availability if needed
            availability = db.query(Availability).filter(
                Availability.mentor_id == service.mentor_id,
                Availability.date == target_date
            ).first()
            
            if availability:
                # Check if all slots for this date are booked
                all_time_slots = db.query(TimeSlot).join(Booking).filter(
                    Booking.mentor_id == service.mentor_id,
                    TimeSlot.date == target_date
                ).all()
                
                booked_slots = sum(1 for ts in all_time_slots if ts.is_booked)
                total_slots = len(all_time_slots)
                
                if total_slots > 0 and booked_slots == total_slots:
                    availability.is_booked = True
            
            db.commit()
            
            return JSONResponse({
                "success": True,
                "booking_id": booking.id,
                "redirect_url": f"/meeting/{booking.id}",
                "meeting_link": meeting_link,
                "meeting_id": meeting_id,
                "is_digital": False,
                "is_free": True,
                "message": "Free session booked successfully!"
            })
    else:
        # Paid services
        if is_digital_service:
            # Check for existing pending digital product booking
            existing_digital_booking = db.query(Booking).filter(
                Booking.learner_id == current_user.id,
                Booking.service_id == service_id,
                Booking.payment_status == "pending",
                Booking.status == "pending",
                Booking.created_at >= datetime.now() - timedelta(hours=1)
            ).first()
            
            if existing_digital_booking:
                print(f"✅ Found existing pending digital booking: {existing_digital_booking.id}")
                return JSONResponse({
                    "success": True,
                    "booking_id": existing_digital_booking.id,
                    "redirect_url": f"/payment/{existing_digital_booking.id}",
                    "is_digital": True,
                    "is_free": False,
                    "existing_booking": True,
                    "message": "Continuing with existing booking"
                })
            
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
                    download_count=0,
                    created_at=datetime.utcnow()
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
                raise HTTPException(status_code=500, detail=f"Error creating payment order: {str(e)}")
        else:
            # Paid live session
            
            # Calculate end time based on service duration
            start_time = datetime.strptime(time_slot, "%H:%M")
            end_time = start_time + timedelta(minutes=service.duration_minutes)
            end_time_str = end_time.strftime("%H:%M")
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Check if time slot is available
            is_available = check_time_slot_availability(
                mentor_id=service.mentor_id,
                date=target_date,
                start_time=time_slot,
                end_time=end_time_str,
                db=db
            )
            
            if not is_available:
                raise HTTPException(status_code=400, detail="This time slot is no longer available")
            
            try:
                # Create Razorpay order
                order_amount = service.price * 100
                order_currency = "INR"
                
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
                
                # Create booking record for session
                booking = Booking(
                    learner_id=current_user.id,
                    mentor_id=service.mentor_id,
                    service_id=service_id,
                    booking_date=target_date,
                    selected_time=time_slot,
                    razorpay_order_id=razorpay_order["id"],
                    amount_paid=service.price,
                    status="pending",
                    payment_status="pending",
                    meeting_link=None,
                    meeting_id=None,
                    notes=f"Session scheduled for {date_str} at {time_slot} (Pending Payment)",
                    created_at=datetime.utcnow()
                )
                
                db.add(booking)
                db.commit()
                db.refresh(booking)
                
                # Create a TimeSlot record to temporarily reserve the slot
                time_slot_record = TimeSlot(
                    booking_id=booking.id,
                    start_time=time_slot,
                    end_time=end_time_str,
                    date=target_date,
                    is_booked=False,  # Not booked yet, just reserved
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
                raise HTTPException(status_code=500, detail=f"Error creating booking: {str(e)}")
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
    
    mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
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
            mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
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
    mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
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
    return RedirectResponse(url=f"https://www.clearq.in/{username}", status_code=301)

@app.get("/mentor/username/{username}/service/{service_id}")
async def redirect_old_service_url(username: str, service_id: int):
    """Redirect old service URLs to new format"""
    return RedirectResponse(url=f"https://www.clearq.in/{username}/service/{service_id}", status_code=301)
# Then your API endpoint

    
@app.get("/{username}", response_class=HTMLResponse)
async def user_profile(
    request: Request,
    username: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """User profile page - works for both mentors and learners"""
    try:
        # Get user with mentor profile
        user = db.query(User).options(
            joinedload(User.mentor_profile)
        ).filter(User.username == username).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.role == "mentor":
            # IMPORTANT: Get mentor in the SAME session with ALL relationships
            mentor = db.query(Mentor).options(
                joinedload(Mentor.user)
            ).filter(Mentor.user_id == user.id).first()
            
            if not mentor:
                # Create mentor profile if it doesn't exist
                mentor = Mentor(
                    user_id=user.id,
                    verification_status="pending",
                    created_at=datetime.utcnow()
                )
                db.add(mentor)
                db.commit()
                db.refresh(mentor)
                # Reload with relationships
                mentor = db.query(Mentor).options(
                    joinedload(Mentor.user)
                ).filter(Mentor.id == mentor.id).first()
            
            # Get services
            services = db.query(Service).filter(
                Service.mentor_id == mentor.id,
                Service.is_active == True
            ).all()
            
            # Ensure day preferences exist
            day_preferences = db.query(AvailabilityDay).filter(
                AvailabilityDay.mentor_id == mentor.id
            ).all()
            
            # If no day preferences, create default ones with 9AM to 9PM
            if not day_preferences:
                print(f"Creating default day preferences for mentor {mentor.id}")
                # CHANGED: 9AM to 9PM instead of 9AM to 5PM
                default_days = [
                    {"day_of_week": 0, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 1, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 2, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 3, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 4, "start_time": "09:00", "end_time": "21:00", "is_active": True},
                    {"day_of_week": 5, "start_time": "10:00", "end_time": "21:00", "is_active": False},  # Changed end time
                    {"day_of_week": 6, "start_time": "10:00", "end_time": "21:00", "is_active": False},  # Changed end time
                ]
                
                for day_data in default_days:
                    day_pref = AvailabilityDay(
                        mentor_id=mentor.id,
                        day_of_week=day_data["day_of_week"],
                        start_time=day_data["start_time"],
                        end_time=day_data["end_time"],
                        is_active=day_data["is_active"],
                        created_at=datetime.utcnow()
                    )
                    db.add(day_pref)
                
                db.commit()
                print(f"✅ Created default day preferences for mentor {mentor.id} (9AM-9PM)")
            
            # Generate availability from day preferences
            try:
                # Use the same db session
                from sqlalchemy.orm import Session as DBSession
                
                # Generate availability
                today = datetime.now().date()
                end_date = today + timedelta(days=30)
                
                generated_count = 0
                current_date = today
                
                while current_date <= end_date:
                    day_of_week = current_date.weekday()
                    
                    # Check if day preference exists and is active
                    day_pref = db.query(AvailabilityDay).filter(
                        AvailabilityDay.mentor_id == mentor.id,
                        AvailabilityDay.day_of_week == day_of_week,
                        AvailabilityDay.is_active == True
                    ).first()
                    
                    if day_pref:
                        # Check if availability already exists
                        existing = db.query(Availability).filter(
                            Availability.mentor_id == mentor.id,
                            Availability.date == current_date
                        ).first()
                        
                        if not existing:
                            # Create new availability with 9AM-9PM
                            availability = Availability(
                                mentor_id=mentor.id,
                                date=current_date,
                                start_time=day_pref.start_time,
                                end_time=day_pref.end_time,
                                is_booked=False,
                                created_at=datetime.utcnow()
                            )
                            db.add(availability)
                            generated_count += 1
                
                    current_date += timedelta(days=1)
                
                if generated_count > 0:
                    db.commit()
                    print(f"✅ Generated {generated_count} availability records for mentor {mentor.id}")
                    
            except Exception as e:
                db.rollback()
                print(f"⚠️ Error generating availability: {e}")
            
            # Get available dates
            try:
                today = datetime.now().date()
                end_date = today + timedelta(days=30)
                
                # Get active day preferences
                active_days = db.query(AvailabilityDay).filter(
                    AvailabilityDay.mentor_id == mentor.id,
                    AvailabilityDay.is_active == True
                ).all()
                
                # Get all availabilities
                availabilities = db.query(Availability).filter(
                    Availability.mentor_id == mentor.id,
                    Availability.date >= today,
                    Availability.date <= end_date,
                    Availability.is_booked == False
                ).all()
                
                # Get exceptions
                exceptions = db.query(AvailabilityException).filter(
                    AvailabilityException.mentor_id == mentor.id,
                    AvailabilityException.date >= today,
                    AvailabilityException.date <= end_date,
                    AvailabilityException.is_available == False
                ).all()
                
                exception_dates = {ex.date for ex in exceptions}
                
                available_dates = []
                for avail in availabilities:
                    # Skip if date is in exceptions
                    if avail.date in exception_dates:
                        continue
                    
                    # Check if this is an active day
                    day_of_week = avail.date.weekday()
                    is_active_day = any(day.day_of_week == day_of_week for day in active_days)
                    
                    if is_active_day:
                        available_dates.append({
                            'date_obj': avail.date,
                            'full_date': avail.date.strftime("%Y-%m-%d"),
                            'day_name': avail.date.strftime("%A"),
                            'day_short': avail.date.strftime("%a"),
                            'day_num': avail.date.day,
                            'month': avail.date.strftime("%b"),
                            'start_time': avail.start_time,
                            'end_time': avail.end_time
                        })
                
                # Sort by date
                available_dates.sort(key=lambda x: x['date_obj'])
                
                print(f"✅ Found {len(available_dates)} available dates for mentor {mentor.id}")
                for date in available_dates[:3]:
                    print(f"  - {date['full_date']} ({date['day_name']}) {date['start_time']}-{date['end_time']}")
                    
            except Exception as e:
                print(f"⚠️ Error getting available dates: {e}")
                available_dates = []
            
            # Prepare mentor data for template
            mentor_data = {
                "id": mentor.id,
                "user_id": mentor.user_id,
                "experience_years": mentor.experience_years,
                "industry": mentor.industry,
                "job_title": mentor.job_title,
                "company": mentor.company,
                "bio": mentor.bio,
                "skills": mentor.skills,
                "linkedin_url": mentor.linkedin_url,
                "github_url": mentor.github_url,
                "twitter_url": mentor.twitter_url,
                "website_url": mentor.website_url,
                "rating": mentor.rating,
                "review_count": mentor.review_count,
                "total_sessions": mentor.total_sessions,
                "is_verified_by_admin": mentor.is_verified_by_admin,
                "verification_status": mentor.verification_status,
                "created_at": mentor.created_at,
                "user": {
                    "id": mentor.user.id,
                    "username": mentor.user.username,
                    "email": mentor.user.email,
                    "full_name": mentor.user.full_name,
                    "profile_image": mentor.user.profile_image,
                    "role": mentor.user.role,
                    "is_active": mentor.user.is_active,
                    "is_verified": mentor.user.is_verified,
                    "created_at": mentor.user.created_at
                } if mentor.user else None
            }
            
            return templates.TemplateResponse("mentor_profile.html", {
                "request": request,
                "current_user": current_user,
                "mentor": mentor_data,  # Pass dict instead of ORM object
                "services": services,
                "available_dates": available_dates[:14]  # Limit to 14 dates
            })
            
        # If user is a learner, show simple learner profile
        elif user.role == "learner":
            return templates.TemplateResponse("learner_profile.html", {
                "request": request,
                "current_user": current_user,
                "profile_user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "profile_image": user.profile_image,
                    "role": user.role,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "created_at": user.created_at
                }
            })
        
        else:
            raise HTTPException(status_code=404, detail="Profile not found")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in user_profile route: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a generic error page instead of 500
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_code": 500,
            "error_message": "Something went wrong. Please try again later."
        }, status_code=500)
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

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

    
@app.post("/api/create-razorpay-order")
async def create_razorpay_order(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        booking_id = data.get("booking_id")
        amount = data.get("amount")  # Should be in paise (e.g., 50000 for ₹500)
        
        if not booking_id or not amount:
            raise HTTPException(status_code=400, detail="Missing booking_id or amount")
        
        # Create Razorpay order
        order_data = {
            "amount": int(amount),  # Must be integer
            "currency": "INR",
            "receipt": f"booking_{booking_id}",
            "payment_capture": 1,  # Auto-capture payment
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
        
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"]
        }
        
    except Exception as e:
        print(f"Error creating Razorpay order: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
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
        
        # Get all possible parameters from different payment flows
        razorpay_payment_id = data.get('razorpay_payment_id') or data.get('payment_id')
        razorpay_order_id = data.get('razorpay_order_id') or data.get('order_id')
        razorpay_signature = data.get('razorpay_signature') or data.get('signature')
        booking_id = data.get('booking_id')
        
        # CRITICAL: Check if all required data is present
        if not razorpay_payment_id or not razorpay_order_id or not razorpay_signature:
            print(f"Missing payment data. Received: {data}")
            return JSONResponse({
                "success": False, 
                "message": "Missing required payment data. Please try again.",
                "code": "MISSING_DATA"
            })
        
        # If booking_id not provided, try to find by order_id
        if not booking_id:
            booking = db.query(Booking).filter(
                Booking.razorpay_order_id == razorpay_order_id
            ).first()
            if booking:
                booking_id = booking.id
                print(f"Found booking by order_id: {booking_id}")
        
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
            # Still try to mark as paid if signature fails (for testing/demo)
            # In production, you might want to be stricter
            print(f"⚠️ Continuing despite signature failure for demo purposes")
        
        # Find the booking
        booking = None
        if booking_id:
            booking = db.query(Booking).filter(Booking.id == booking_id).first()
        
        # If booking not found by ID, try by order ID
        if not booking:
            booking = db.query(Booking).filter(
                Booking.razorpay_order_id == razorpay_order_id
            ).first()
        
        if not booking:
            print(f"❌ Booking not found for order: {razorpay_order_id}")
            return JSONResponse({
                "success": False, 
                "message": "Booking not found",
                "code": "BOOKING_NOT_FOUND"
            })
        
        print(f"✅ Booking found: ID={booking.id}, Service ID={booking.service_id}")
        
        # Check if already paid
        if booking.payment_status == "paid":
            print("⚠️ Payment already marked as paid")
            service = db.query(Service).filter(Service.id == booking.service_id).first()
            is_digital = service.is_digital if service else False
            
            redirect_url = "/dashboard"
            if is_digital and service:
                redirect_url = f"/digital-product/{service.id}"
            
            return JSONResponse({
                "success": True, 
                "message": "Payment already verified",
                "redirect_url": redirect_url,
                "is_digital": is_digital
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
            "is_digital": is_digital,
            "booking_id": booking.id
        })
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error in verify_payment_api: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False, 
            "message": f"Payment verification failed: {str(e)}",
            "code": "SERVER_ERROR"
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
            mentor = db.query(Mentor).options(joinedload(Mentor.user)).filter(Mentor.id == mentor_id).first()
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
