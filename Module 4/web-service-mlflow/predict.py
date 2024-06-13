import mlflow
from flask import Flask, request, jsonify
import pickle

# get model from S3
RUN_ID = ""
AWS_S3_BUCKET = ""
logged_model = f's3://{AWS_S3_BUCKET}/1/{RUN_ID}/artifacts/model'

model = mlflow.pyfunc.load_model(logged_model)
with open('dict_vectorizer.pkl', 'rb') as f:
    dict_vectorizer = pickle.load(f)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return preds[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    features = dict_vectorizer.transform(features)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)