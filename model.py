from database import *
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


class SessionB(Base):
    __tablename__ = "Session"

    id = Column(Integer, primary_key=True, index=True)
    productId = Column(String,)
    categoryId = Column(String)
    categoryCode = Column(String)
    brand = Column(String,)
    eventType = Column(String)
    eventTime = Column(String)
    price = Column(String)
    # eventTime = Column(String)