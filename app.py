from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os

from src.predict import predict

app = Flask(__name__)

# -------------------------------
# Load metrics safely
# -------------------------------
try:
    with open("artifacts/training_report.json") as f:
        metrics = json.load(f)
except Exception as e:
    print("Metrics load error:", e)
    metrics = {"error": "metrics not found"}


# -------------------------------
# Generate dynamic plots
# -------------------------------
def generate_plots(df):
    plots = {}

    try:
        # Histogram
        plt.figure()
        sns.histplot(df["processing_time_days"], kde=True)
        plt.title("Processing Time Distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots["hist"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # Visa Type
        plt.figure()
        sns.boxplot(x="visa_type", y="processing_time_days", data=df)
        plt.title("Processing Time by Visa Type")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots["visa"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # Country (FIXED)
        plt.figure()
        sns.boxplot(x="country", y="processing_time_days", data=df)
        plt.title("Processing Time by Country")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots["country"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

    except Exception as e:
        print("🔥 Plot Error:", e)

    return plots


# -------------------------------
# MAIN ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template(
            "index.html",
            metrics=json.dumps(metrics, indent=2)
        )

    try:
        # Inputs
        country = request.form.get("country")
        visa_type = request.form.get("visa_type")
        office = request.form.get("office")
        month = request.form.get("month")

        print("📥 Inputs:", country, visa_type, office, month)

        # Validation
        if not month:
            raise ValueError("Month is required")

        month = int(month)

        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")

        data = {
            "country": country,
            "visa_type": visa_type,
            "processing_office": office,
            "month": month
        }

        # Prediction
        result = predict(data)
        print("🔮 Prediction:", result)

        # Load dataset safely
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "h1b_data_preprocessed_final.csv")

        print("📂 Loading dataset from:", file_path)

        df = pd.read_csv(file_path)

        print("📊 Columns:", df.columns)

        # FIXED filtering
        filtered_df = df[
            (df["country"] == country) &
            (df["visa_type"] == visa_type)
        ]

        if filtered_df.empty:
            print("⚠️ No matching data, using full dataset")
            filtered_df = df

        plots = generate_plots(filtered_df)

        return render_template(
            "index.html",
            prediction=round(result, 2),
            plots=plots,
            metrics=json.dumps(metrics, indent=2)
        )

    except ValueError as e:
        print("⚠️ ValueError:", e)
        return render_template(
            "index.html",
            error=str(e),
            metrics=json.dumps(metrics, indent=2)
        )

    except Exception as e:
        print("🔥 ERROR:", e)
        return render_template(
            "index.html",
            error=str(e),   # shows real error
            metrics=json.dumps(metrics, indent=2)
        )


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
