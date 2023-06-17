from fastapi import FastAPI
from schema import Event



app = FastAPI()


id = [1000894,1000978,1001588,1001605,1001606,1001618,1001619,1001820,]

























































user_data_list = []

@app.post('/event')
def home(event : Event):
    # data = event
    user_data_list.append(event)
    # dataS.append([data])
    return user_data_list

@app.get('/getid')
def getId(id):
    x= id
    return x

   
    
