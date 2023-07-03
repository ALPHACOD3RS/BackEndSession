from pydantic import BaseModel

class Event(BaseModel):
    productId : str
    categoryId : str
    categoryCode : str
    brand : str
    eventType: str
    eventTime : str
    price : float
    # userId : str
    # userSession : str

