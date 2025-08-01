from hate.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI
import uvicorn
import sysfrom 
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import *


text : str = "what is machine learning"
app = FastAPI()

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url = "/docs")


@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")
    
    except Exception as e:
        raise Response(f"Error Occured {e}")
    
    
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.run_pipeline(text)
        return text

    except Exception as e:
        raise CustomException(e,sys) from e

if __name__ == "__main__":
    uvicorn.run(app, host = App_HOST, port = APP_PORT)
    