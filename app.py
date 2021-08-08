from flask import Flask , render_template , request 

import pickle
import numpy as np

model = pickle.load(open('readmission_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def makePredict():
    int_features=[int(x) for x in request.form.values()]
    final=np.array([int_features])

    
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('index.html',pred = 'Patients likely to get readmitted. \n Probability is {}'.format(output))
    
    else:
        return render_template('index.html',pred='Patient is not likely to readmit.\n Probability of readmission is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True) 