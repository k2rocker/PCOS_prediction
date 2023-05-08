import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
dmodel=pickle.load(open('dtreepcos.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=dmodel.predict(new_data)
    print("predicted output: ",output[0])
    if(output[0]==0):
        return render_template('home.html')
    else:
        return render_template('home.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    output=dmodel.predict(final_input)
    if(output[0]==0):
        return render_template("home.html",prediction_text="The prediction status of PCOS is : NEGATIVE")
    else:
        return render_template("home.html",prediction_text="The prediction status of PCOS is : POSITIVE")


if __name__=="__main__":
    app.run(debug=True)