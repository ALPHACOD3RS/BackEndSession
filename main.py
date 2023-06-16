from fastapi import FastAPI



app = FastAPI()


id = [
1000894,
1000978,
1001588,
1001605,
1001606,
1001618,
1001619,
1001820,
]



@app.get('/home')
def home():

    x = id
    return x
    
