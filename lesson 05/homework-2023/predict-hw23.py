import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])

score = model.predict_proba(X)[0,1]

print(f"The score for the client is: {score:.3f}")


