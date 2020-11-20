import uvicorn
from fastapi import FastAPI
from Model import IrisModel, IrisSpecies
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
model = IrisModel()

origins = [
    "https://iris-predictor.netlify.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


