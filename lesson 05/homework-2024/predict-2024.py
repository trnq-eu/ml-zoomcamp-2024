import pickle
from flask import Flask, request, jsonify

model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

app = Flask('credit')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    client = request.get_json()
    X = dv.transform(client)

    score = model.predict_proba(X)[0,1]
    
    result = {
        'subscription probability' : score
    }

    return jsonify(result)
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)