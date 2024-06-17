from flask import Flask, request, jsonify

from starter import apply_model


app = Flask("duration-predictor")

@app.route("/predict", methods=["POST"])
def predict():
    year = request.json["year"]
    month = request.json["month"]
    
    y_pred = apply_model(year, month, save_files=False)
    print(f"The mean predicted duration for {year:04d}/{month:02d} " +\
          f"is: {y_pred.mean():.2f}")

    object_to_response = {
        "mean_predicted_duration": float(y_pred.mean())
    }
    return jsonify(object_to_response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)