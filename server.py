from flask import Flask,render_template,request
import numpy as np
import pickle
import pandas as pd
model=pickle.load(open('spam.pickle','rb'))
df=pd.read_csv('spam.csv')
df["label"]=df.Category.apply(lambda x:0 if x=='ham'else 1)
df.drop("Category",inplace=True,axis=1)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
xtrain,xtest,ytrain,ytest=train_test_split(df.Message,df.label,test_size=0.25)
xtraincount=v.fit_transform(xtrain.values)
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(xtrain, ytrain)

app=Flask(__name__)
@app.route('/')
def home():

    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    input=[str(request.form["input"])]
    test=v.transform(input)


    res=clf.predict(input)

    return render_template('index.html',ans="predicted value :{}".format(res))


if __name__=='__main__':
    app.run(debug=True)
