import json
import logging
import os
import pickle
from io import StringIO

import pandas as pd
from flask import jsonify


class HousePricePredictor:
    def __init__(self):
        self.model = None

    def load_model(self, file_path):
        self.model = pickle.load(open(file_path, 'rb'))

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        if self.model is None:
            try:
                model_repo = os.environ['MODEL_REPO']
                file_path = os.path.join(model_repo, "model.pkl")
                self.model = pickle.load(open(file_path, 'rb'))
            except KeyError:
                print("MODEL_REPO is undefined")
                self.model = pickle.load(open("xgboost_model.pkl", 'rb'))

        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        xNew = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement']]
        y_pred = self.model.predict(xNew)
        house_value = float(y_pred[0])  # ensure JSON-serializable
        return house_value
