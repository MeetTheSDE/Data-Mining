from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pickle
import sklearn
import load 
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

model = open('C:/Users/patel/Google Drive/Colab Notebooks/model_ridge_class.sav', 'rb')
clf = joblib.load(model)

@app.route('/predict',methods = ['POST'])
def predict():
    
    vectorizer = joblib.load(open('C:/Users/patel/Google Drive/Colab Notebooks/vectorizer.sav'))
    transformer = joblib.load(open('C:/Users/patel/Google Drive/Colab Notebooks/transformer.sav'))
    f = [x for x in request.form.values()]
    final = [np.array(f)]
    pred = clf.predict(request.form.values())
    out = round(pred[0],2)

    #if request.method == 'POST':
    #    message = request.form['message']
    #    data = [message]
    #    text = vectorizer.transform(data)
    #    text = transformer.fit_transform(text)
    #    my_prediction = clf.predict(text)
    #    my_prediction = np.round(clf.predict(text))
    return render_template('result.html',prediction = out)

if __name__ == "__main__":
    app.run(debug=True)