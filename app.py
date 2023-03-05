from flask import Flask, render_template, request
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model import Model

model = Model()
model.load_model()


app = Flask(__name__,static_folder='static')

def preprocess(text):
    return text.split('.')

def score_terms(terms_list):
    # return [random.uniform(0, 1) for _ in range(len(terms_list))]
    return [x[0] for x in model.predict(terms_list)]

def highlight(list_terms,scores):
    cutoff = 0.9
    sent = ''
    for sentence,scr in zip(list_terms,scores):
        sent += '<mark style="background-color: rgba(255,0,0,{});">'.format(scr)+sentence+'</mark>. '
    return sent

@app.route('/validate', methods=('GET', 'POST'))
def validate():
    text = request.form['document_text']
    list_terms = preprocess(text)
    scores = score_terms(list_terms)
    highlighted = highlight(list_terms,scores)
    return render_template('TOSVerifyHighlight.html',data={'text':highlighted})



@app.route('/')
def index():
    return render_template('TOSVerify.html')

app.run(debug=True,)
