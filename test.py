from flask import Flask, render_template, request
import pickle
import numpy as numpy
import pandas as pd

filename = 'lr_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def inicio():
    #return str(loaded_model.predict([[123]]))
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def home():
	val_to_predict = request.form['val']
	prediction = loaded_model.predict([[int(val_to_predict)]])
	return render_template('after.html', data=prediction[0], valor_prueba=val_to_predict)
	#return render_template('after.html', data=prediction)



if __name__ == '__main__':
    app.run(debug=True)