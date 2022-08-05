import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
from keras.models import load_model
model = load_model('churnmodelANN.h5')
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:,3:13].values
# Extracting dependent variable:
y = dataset.iloc[:, 13].values
# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
#dummy encoding.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('yash', OneHotEncoder(), [1])],remainder='passthrough')
X=columnTransformer.fit_transform(X)
#dummy encoding.

 # Dummy Variable trapping
X = X[:, 1:] 
# Splitting the Dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Feature Scaling
# Standard Scaling:  Standardization = X'=X-mean(X)/standard deviation
# normal scaling : Normalization= X'=X-min(X)/max(x)-min(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  creditscore = int(request.args.get('creditscore'))
  geo = int(request.args.get('geo'))
  age = int(request.args.get('age'))
  tenure = int(request.args.get('tenure'))  
  balance = int(request.args.get('balance'))
  numofproducts = int(request.args.get('numofproducts')) 
  creditcards=int(request.args.get('creditcards'))
  activemember = int(request.args.get('activemember'))
  
  salary = int(request.args.get('salary')) 
  
  
  y_pred= model.predict(sc_X.transform(np.array([[0,1, creditscore,geo,age,tenure,balance,
                                                  numofproducts,creditcards,activemember,salary]])))
  y_pred = (y_pred > 0.5)
  if y_pred>0.5:
    result="Customer will not exit Bank"
  else:
    result="Customer will exit bank"
        
  return render_template('index.html', prediction_text='Model  has predicted  : {}'.format(result))
