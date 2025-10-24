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
        dfcp = df.copy()
        y_classes = self.model.predict(xNew)
        logging.info(y_classes)
        dfcp['house_value'] = y_classes.tolist()
        status = (dfcp['house_value'][0] > 0.5)
        # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return str(status)
