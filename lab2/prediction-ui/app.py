# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/predicthouse', methods=["GET", "POST"])
def predict_house_price():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [
            {
            "Median_Income": float(request.form.get("Median_Income")),            # e.g. 8.3252
            "Median_Age": int(request.form.get("Median_Age")),                   # e.g. 41
            "Tot_Rooms": int(request.form.get("Tot_Rooms")),                     # e.g. 880
            "Tot_Bedrooms": int(request.form.get("Tot_Bedrooms")),               # e.g. 129
            "Population": int(request.form.get("Population")),                   # e.g. 322
            "Households": int(request.form.get("Households")),                   # e.g. 126
            "Latitude": float(request.form.get("Latitude")),                     # e.g. 37.88
            "Longitude": float(request.form.get("Longitude")),                   # e.g. -122.23
            "Distance_to_coast": float(request.form.get("Distance_to_coast")),   # e.g. 9263.04077285038
            "Distance_to_LA": float(request.form.get("Distance_to_LA")),         # e.g. 556529.1583418
            "Distance_to_SanDiego": float(request.form.get("Distance_to_SanDiego")), # e.g. 735501.80698384
            "Distance_to_SanJose": float(request.form.get("Distance_to_SanJose")),   # e.g. 67432.5170008434
            "Distance_to_SanFrancisco": float(request.form.get("Distance_to_SanFrancisco")) # e.g. 21250.2137667799
            }
        ]

        app.logger.debug("Prediction input : %s", prediction_input)

        # use requests library to execute the prediction service API by sending an HTTP POST request
        # use an environment variable to find the value of the diabetes prediction API
        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        app.logger.info("Prediction Output : %s", prediction_value)
        return render_template("response_page.html",
                               prediction_variable=eval(prediction_value))

    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/checkdiabetes' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
