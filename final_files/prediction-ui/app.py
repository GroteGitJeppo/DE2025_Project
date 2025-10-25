# importing Flask and other modules
import requests
from requests.exceptions import JSONDecodeError, RequestException
from flask import Flask, request, render_template, jsonify
import os, logging, json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = app.logger

@app.route('/predicthouse', methods=["GET", "POST"])
def predict_house_price():
    if request.method == "GET":
        return render_template("input_form_page.html")  # always returns

    # POST:
    try:
        prediction_input = [{
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
        }]
    except Exception:
        logger.exception("Invalid form input")
        return render_template("response_page.html", prediction_variable="Invalid input"), 400

    predictor_api_url = os.environ.get('PREDICTOR_API')
    if not predictor_api_url:
        logger.error("PREDICTOR_API environment variable is not set")
        return render_template("response_page.html", prediction_variable="Server not configured (PREDICTOR_API)."), 500

    url = predictor_api_url.rstrip('/')  # normalize /predicthouse

    try:
        res = requests.post(url, json=prediction_input, timeout=15)
        res.raise_for_status()
        try:
            payload = res.json()
        except JSONDecodeError:
            logger.error("Non-JSON response from predictor: %s", res.text[:200])
            return render_template("response_page.html", prediction_variable="Invalid response from predictor."), 502

        if isinstance(payload, dict) and 'result' in payload:
            prediction_value = payload['result']
            logger.info("Prediction Output : %s", prediction_value)
            return render_template("response_page.html", prediction_variable=float(prediction_value))

        logger.error("Unexpected JSON shape: %s", payload)
        return render_template("response_page.html", prediction_variable="Unexpected predictor response."), 502

    except RequestException:
        logger.exception("Error calling predictor API")
        return render_template("response_page.html", prediction_variable="Could not reach predictor."), 502
    
    
# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
