import flask
import io
import string
import time
import os
import numpy as np
import pandas as pd
import traceback
import joblib
from flask import Flask, jsonify, request, render_template

#cancer_model = joblib.load('../my project codes/cancer_model.mdl')

#Test Jsons for the input
# [
#    {"radius":23, "texture" : 12, "perimeter" : 120, "area": 140, "smoothness": 0.123, "compactness" : 0.2, "symmetry" : 0.20, "fractal_dimension":0.02},
#    {"radius":26, "texture" : 19, "perimeter" : 150, "area": 140, "smoothness": 0.123, "compactness" : 0.6, "symmetry" : 0.34, "fractal_dimension":0.05}
# ]


app = Flask(__name__)


#This function is the contains the actual facilty that is used for the api. The endpoint is the /predict which is what you attach to the end of the
#url that allows you to call and use the built model.
@app.route('/predict', methods=['POST'])
def predict():
    if cancer_model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(cancer_model.predict(query))

            print (prediction)

            #For some reason numpy and jsonify are quarelling so I just did the next two lines to by pass that 
            #and ensure that the prediction is returned in postmani
            pre = str(prediction)

            predicted = eval(pre)

            return jsonify(predicted)
            #return ''.join(prediction)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Model For Final Year Project. For  quick clarification, ' \
           'the key for the API is simple if the output is 1 the tumour in question is Malignant and if the output is 0, ' \
           'then the output is Benign.  Thank you for choosing our API, it means a lot. '


if __name__ == '__main__':

    cancer_model = joblib.load('../my project codes/cancer_model.mdl')

    print("Loaded Model")

    model_columns = joblib.load("../my project codes/model_columns.pkl")

    print("Model Columns loaded")

    app.run(debug=True, host='0.0.0.0')