import os
from flask import Flask, request, render_template

from src.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("index.html")

    try:
        # 🔥 SAFE INPUT (NO ERROR EVEN IF EMPTY)
        data = CustomData(
            G1=int(request.form.get("G1", 0)),
            G2=int(request.form.get("G2", 0)),
            studytime=int(request.form.get("studytime", 1)),
            absences=int(request.form.get("absences", 0)),
            failures=int(request.form.get("failures", 0)),
            sleep_hours=float(request.form.get("sleep_hours", 6)),
            social_media=float(request.form.get("social_media", 2)),
            Medu=int(request.form.get("Medu", 2)),
            Fedu=int(request.form.get("Fedu", 2)),
            famrel=int(request.form.get("famrel", 3)),
            internet=int(request.form.get("internet", 1)),
            higher=int(request.form.get("higher", 1)),
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        prediction = round(float(result[0]), 2)

        return render_template("index.html", results=prediction)

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html", results="Error occurred")


if __name__ == "__main__":
    app.run(debug=True)