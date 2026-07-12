import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "iris_model.pkl"
model = joblib.load(MODEL_PATH)

FLOWER_MAP = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route("/", methods=["GET"])
def home():
    """Serve the web UI home page template."""
    return render_template("index.html"), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts feature JSON, processes it, and returns the predicted flower class."""
    try:
        data = request.get_json()

        features = [[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]]

        prediction = model.predict(features)[0]
        predicted_class = FLOWER_MAP[int(prediction)]
        
        return jsonify({
            "class": predicted_class
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Invalid request payload: {str(e)}"
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)