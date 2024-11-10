from flask import Flask, request, jsonify
import pickle

model = "xgbm_final_model_local.pkl"

# Load the model
with open(model, 'rb') as file:
    dv, loaded_model = pickle.load(file)

app = Flask('kidney')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    patient = request.get_json()
    X = dv.transform([patient])

    score = loaded_model.predict_proba(X)[0, 1]

    if round(float(score), 3) > 0.95:

        result = {
            'score' : float(score),
            'diagnosis' : 'positive'
        }
    elif round(float(score), 3) > 0.9:
        result = {
            'score' : float(score),
            'diagnosis' : 'uncertain'
        }
    else:
        result = {
            'score' : float(score),
            'diagnosis' : 'negative'
        }



    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True, host='http://217.160.226.158', port=9696)
