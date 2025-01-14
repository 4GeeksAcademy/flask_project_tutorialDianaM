# This is a small change to ensure Git detects the file

from flask import Flask, request, render_template
from pickle import load
import os

app = Flask(__name__)

# Correct the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'optimized_random_forest_model_1000estimators_max_depth5_min_samples_split5_min_samples_leaf2_42.sav')
with open(model_path, "rb") as model_file:
    model = load(model_file)

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtain values from form
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])

        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)