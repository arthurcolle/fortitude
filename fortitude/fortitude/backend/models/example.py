from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
from . import FortitudeBaseModel

class User(FortitudeBaseModel):
    """User data model"""
    name: str
    email: str
    age: Optional[int] = None

class Product(FortitudeBaseModel):
    """Product data model"""
    name: str
    description: Optional[str] = None
    price: float
    stock: int = 0

class Order(FortitudeBaseModel):
    """Order data model"""
    user_id: str
    products: list[str]  # List of product IDs
    total: float
    status: str = "pending"
