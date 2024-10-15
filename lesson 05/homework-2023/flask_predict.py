import pickle
from flask import Flask, request, jsonify

model_file = 'app/model2.bin'
dv_file = 'app/dv.bin'

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

app = Flask('credit')

@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform(customer)
    score = model.predict_proba(X)[0,1]

    result = {
        'credit_probability': score
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
