from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "credit_card.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not found or could not be loaded", 500

    try:
        # Extract input values from form
        features = [float(request.form[f"feature{i}"]) for i in range(10)]
        
        # Convert input to NumPy array and reshape
        input_data = np.array(features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_data)[0]
        result = "Good Customer" if prediction == 0 else "Bad Customer"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
