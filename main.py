from fastapi import Depends, FastAPI
import pandas as pd
from preprocessing import preprocessing
from schema import Event
import database, model, schema
from database import SessionLocal, engine
from sqlalchemy.orm import Session
# import cudf




app = FastAPI()

model.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()





id = [100035477,
100035478,
100035479,
100035482,
100035483,
100035485,
100035486,
100035487,
100035488,
100035490,
100035493,
100035494,
100035496,
4100242,
4100243,
4100244,
4100245,
4100248,
4100249,
4100250,
4100251,
4100252,
4100253,
4100254,
4100255,
4100256,
4100257,
4100258,
4100259,
4100260,
4100261,
4100263,
4100265,
4100266,
4100268,
4100271,
4100275,
4100276,
4100277,
4100283,
4100286,
4100289,
4100290,]


user_data_list = []

data = {
    'productId': [],
    'categoryId': [],
    'categoryCode': [],
    'brand': [],
    # 'user_id': [587769686, 587769686, 587769686, 587769686, 587769686],
    'eventType': [],
    'eventTime': [],
    'price': [],
    'user_id': [],
    'user_session': []



    # 'user_session': ['179879', '179879', '179879', '179879', '179879']
    }           
@app.post('/event')
def home(event : Event):
    for key, value in event.dict().items():
        data[key].append(value)

    # raw_df = cudf.DataFrame(data)
    # preprocessing(raw_df)

    print(data)

    return data

@app.get('/getid')
def getId():
    x= id
    return x



@app.post('/eventdata')
def users(request: schema.Event, db: Session = Depends(get_db)):
    new_event = model.SessionB(productId = request.productId, categoryId = request.categoryId, categoryCode = request.categoryCode, brand = request.brand, eventType = request.eventType, eventTime = request.eventTime, price = request.price )
    
    db.add(new_event)
    db.commit()
    db.refresh(new_event)
    print(new_event)
    return new_event

@app.get('/geteventdata')
def getEventData(db: Session = Depends(get_db)):
    eventData = db.query(model.SessionB).all()
    # print(data)
    return eventData