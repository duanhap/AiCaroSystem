from sqlalchemy.orm import Session
from app.models.user import User

def get_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def create(db: Session, username: str, email: str, password_hash: str, role: str = "user"):
    user = User(username=username, email=email, password_hash=password_hash, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
