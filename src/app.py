from flask import Flask, request, render_template
import json
import os

app = Flask(__name__)

# Correct the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'optimized_random_forest_model.json')

# Load the model parameters and classes from the JSON file
with open(model_path, 'r') as f:
    model_data = json.load(f)

model_params = model_data['params']
model_classes = model_data['classes']

# Reconstruct the model (without using scikit-learn)
class RandomForestClassifier:
    def __init__(self, **params):
        self.params = params
        self.classes_ = None

    def predict(self, X):
        # Dummy predict method
        return [0] * len(X)

model = RandomForestClassifier(**model_params)
model.classes_ = model_classes

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