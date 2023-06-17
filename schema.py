from pydantic import BaseModel

class Event(BaseModel):
    productId : str
    categoryId : str
    categoryCode : str
    eventType : str
    price: str
    brand : str
    eventTime : str
    userId : str
    userSession : str
