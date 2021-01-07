import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

with open('classifier.pkl','rb') as pickle_in:
    classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Hello'

@app.route('/predict')
def predict_noteAuth():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return f"Predicted values is {prediction}"


@app.route('/predict_file', methods=['Post']) 
def predict_noteAuth_testfile():
    
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return f"Prediction values of files is {list(prediction)}"
    
if __name__ == '__main__':
    app.run()
    