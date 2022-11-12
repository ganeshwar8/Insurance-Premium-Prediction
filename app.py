import pickle
from flask import Flask,app,redirect,render_template,request,jsonify,url_for
import pandas as pd
import numpy as np

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/health_api',methods=['POST'])
def health_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_input=(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_input)[0]
    print(output)
    return jsonify(output)


@app.route('/health',methods=['POST'])
def health():
    data=[x for x in request.form.values()]
    print(data)
    new_input=(np.array(data).reshape(1,-1))
    print(new_input)
    output=model.predict(new_input)[0]
    print(output)
    return render_template('result.html',
    result="insurance expenses predicton based on individual health situation {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)