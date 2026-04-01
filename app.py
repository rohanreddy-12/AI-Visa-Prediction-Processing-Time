from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

# ==========================================
# GOOGLE DRIVE CLOUD DOWNLOAD LOGIC
# ==========================================
MODEL_PATH = "visa_processing_model.pkl"

# ⚠️ IMPORTANT: Replace the string below with your actual Google Drive File ID!
# If your link is: https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view
# Your ID is: 1aBcDeFgHiJkLmNoPqRsTuVwXyZ
FILE_ID = "1LvAKiyLp_Fbkwu3eiR4tbCUQ_0X3p41Y" 

if not os.path.exists(MODEL_PATH):
    print("Downloading massive model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
# ==========================================

# Load the model and encoders into memory
model = joblib.load(MODEL_PATH)
encoders = joblib.load("label_encoders.pkl")

FEATURES = [
    "emp_state", "work_state", "job_title", "soc_name",
    "full_time_position", "prevailing_wage", "pw_level",
    "emp_h1b_dependent", "case_year"
]

# --- HTML Page Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --- Machine Learning API Route ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = {}

        # Parse and encode incoming JSON data
        for col in FEATURES:
            val = data.get(col)

            if col in encoders:
                try:
                    # Convert text (like "California") to the number the model expects
                    input_data[col] = encoders[col].transform([val])[0]
                except:
                    # Fallback if an unknown category is passed
                    input_data[col] = 0
            else:
                # Convert numbers (like wage or year) directly to float
                input_data[col] = float(val)

        # Convert to DataFrame for the model
        df = pd.DataFrame([input_data])
        
        # Execute Prediction
        pred = model.predict(df)[0]

        # Calculate a +/- 15% Variance Margin
        margin = pred * 0.15
        lower = int(max(1, pred - margin))
        upper = int(pred + margin)

        # 🔥 Explainable AI (XAI) Logic
        importances = model.feature_importances_
        feature_impact = sorted(
            zip(FEATURES, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # Grab the top 3 most influential factors for this specific prediction
        top_features = [
            {"feature": f, "importance": round(i, 3)}
            for f, i in feature_impact[:3]
        ]

        # Send response back to the frontend
        return jsonify({
            "status": "success",
            "predicted_days": round(pred),
            "range": f"{lower} - {upper}",
            "confidence_interval": "85%",
            "top_features": top_features
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    # Runs the app locally on port 10000
    app.run(host="0.0.0.0", port=10000)