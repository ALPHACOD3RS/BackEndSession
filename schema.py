from pydantic import BaseModel

class Event(BaseModel):
    productId : str
    categoryId : str
    categoryCode : str
    eventType : str
    price: float
    brand : str
    eventTime : str
    userId : str
    userSession : str
