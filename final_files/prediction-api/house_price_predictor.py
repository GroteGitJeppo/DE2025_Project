import json
import logging
import os
import pickle
from io import StringIO

import pandas as pd
from flask import jsonify
from google.cloud import storage

storage_client = storage.Client()

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
                model_files = [blob for blob in storage_client.list_blobs(model_repo) if blob.name.endswith('.pkl')]
                if not model_files:
                    raise FileNotFoundError("No model files found in MODEL_REPO.")
                # Find the latest updated model file
                latest_blob = sorted(model_files, key=lambda tup: tup[1])[-1][0]
                # Get the file path of the latest model file in the repo
                latest_model_path = os.path.join(model_repo, latest_blob.name)
                self.model = pickle.load(open(latest_model_path, 'rb'))
            except KeyError:
                print("MODEL_REPO is undefined")
                self.model = pickle.load(open("lr_model.pkl", 'rb'))

        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        xNew = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement']]
        y_pred = self.model.predict(xNew)
        house_value = float(y_pred[0]) 
        return house_value
