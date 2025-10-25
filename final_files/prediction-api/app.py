import os

from flask import Flask, request, jsonify

from house_price_predictor import HousePricePredictor

app = Flask(__name__)
app.config["DEBUG"] = True

dp = HousePricePredictor()
@app.route('/predicthouse/', methods=['POST']) # path of the endpoint. Except only HTTP POST request
def predict_str():
    prediction_input = request.get_json()
    try:
        result = dp.predict_single_record(prediction_input)
        return jsonify({"result": result})
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 400



# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)

