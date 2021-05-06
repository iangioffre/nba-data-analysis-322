# we are going to use Flask, a micro web framework
import os
import pickle
from flask import Flask, jsonify, request
import mysklearn.myutils as myutils

# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage


@app.route("/", methods=["GET"])
def index():
    # return content and a status code
    return "<h1>Welcome to my App</h1><p>Possible arguments: year_start, height (ft-in), weight_class ([1 -> 5]), position ([G, F, C, G-F, F-G, F-C, C-F])</p>", 200

# one for the /predict


@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    year_start = request.args.get("year_start", "")
    height = request.args.get("height", "")
    weight_class = request.args.get("weight_class", "")
    position = request.args.get("position", "")
    print("year, height, weight, position:",
          year_start, height, weight_class, position)
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    # prediction = predict_interviews_well([level, lang, tweets, phd])
    prediction = predict_salary_class(
        [year_start, height, weight_class, position])
    # if anything goes wrong, predict_interviews_well() is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        # failure!!
        return "Error making prediction", 400


def rf_predict(header, forest, instance):
    candidate_predictions = []
    for tree in forest:
        candidate = myutils.predict_helper(tree, instance)
        if candidate is not None:
            candidate_predictions.append(candidate)
    prediction = myutils.compute_majority_vote_prediction(
        candidate_predictions)
    return prediction


def predict_salary_class(instance):
    infile = open("forest.p", "rb")
    header, forest = pickle.load(infile)
    infile.close()
    # print("header:", header)
    # print("forest:", forest)

    # use the forest to make a prediction
    try:
        prediction = rf_predict(header, forest, instance)
        # print("Prediction:", prediction)
        return prediction  # recursive function
    except:
        return None


if __name__ == "__main__":
    # by default, Flask runs on port 5000
    port = os.environ.get("PORT", 5000)
    # TODO: set debug to False for production
    app.run(debug=False, host="0.0.0.0", port=port)
