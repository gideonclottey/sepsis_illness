from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
# this help to load your models
from typing import Dict
import joblib
#server to handle response and request
import uvicorn  # this is the server

#app initialization 
app = FastAPI(
    # this shows when app start running
    title=' Sepsis  Prediction api',
    description=' This will predict sepsis'
    )

# model path 
MODEL_PATHS ={
    'logistic_regression': 'model/Logistic Regression_pipeline.pkl',
    'random_forest':'model/Random Forest_pipeline.pkl',
    'knn': 'model/K-Nearest Neighbors_pipeline.pkl'
}

# load models
models ={}

for model, path in MODEL_PATHS.items():
    try:
        models[model] = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"fail to load '{model}' from '{path}'. Error: {e}")


#app end point

@app.get('/')
async def st_endpoint():
    return{"message":" welcome to sepsis prediction app"}


# prediction end point

@app.post('/predict')
async def predictor(model: str, file:UploadFile = File(...)):
    """
    accepts a model and loads a ile and 
    return a jason with prediction for each row
    """
    # loading csv file
    try:
        df =pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error detaile {e}')

    #verify required features

    required_features = models[model].n_features_in_
    if len(df.columns) != required_features:
        raise HTTPException(status_code=400, detail=f' the model excepts {required_features} but file has {len(df.columns)} columns')

    #convert to array
    # =df.values
    
    #select model
    selected_model = models[model]
    
    # making predictions
    try:
        predictions = selected_model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f' Error during prediction')

    #show response
    results ={
        'model_used': model,
        'predictions': predictions.tolist()
    }
    return results

if __name__== "__main__":
    uvicorn.run("mlapi:app", reload=True)