import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load dataset and trained Ridge model
data = pd.read_csv("cleaned_data.csv")
with open("RidgeModel.pkl", "rb") as f:   # ✅ use consistent filename
    ridge_model = pickle.load(f)

@app.route('/')
def index():
    # Pass sorted unique locations to template
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    sqft = request.form.get("sqft")

    # Validate inputs
    try:
        sqft_val = int(sqft)
        bhk_val = int(bhk)
        bath_val = int(bath)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values"}), 400

    # Build dataframe for model
    input_df = pd.DataFrame(
        [[location, sqft_val, bath_val, bhk_val]],
        columns=["location", "total_sqft", "bath", "bhk"]
    )
    # input_df["Unnamed: 0"] = 0
    # Predict using trained Ridge pipeline
    prediction = ridge_model.predict(input_df)[0]

    return jsonify({"predicted_price": f"${prediction:,.2f}"})

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
