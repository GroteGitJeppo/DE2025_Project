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
            "bedrooms": float(request.form.get("bedrooms")),
            "bathrooms": float(request.form.get("bathrooms")),
            "sqft_living": int(float(request.form.get("sqft_living"))),
            "sqft_lot": int(float(request.form.get("sqft_lot"))),
            "floors": float(request.form.get("floors")),
            "waterfront": int(request.form.get("waterfront")),
            "view": int(request.form.get("view")),
            "condition": int(request.form.get("condition")),
            "sqft_above": int(float(request.form.get("sqft_above"))),
            "sqft_basement": int(float(request.form.get("sqft_basement")))
            }
        ]

        app.logger.debug("Prediction input : %s", prediction_input)

        # use requests library to execute the prediction service API by sending an HTTP POST request
        # use an environment variable to find the value of the diabetes prediction API
        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(url.rstrip('/'), json=prediction_input, timeout=15)
        res.raise_for_status()
        try:
            payload = res.json()
        except requests.exceptions.JSONDecodeError:
            app.logger.error("Non-JSON response from predictor: %s", res.text[:200])
            return render_template("response_page.html", prediction_variable="Invalid response from predictor."), 502


    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/checkdiabetes' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
