import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem
import re

from flask import Flask, request, jsonify, Response
from flasgger import Schema, Swagger, SwaggerView, fields
from flask_restful import Api, Resource, abort, reqparse
try:
    from http import HTTPStatus
except ImportError:
    import httplib as HTTPStatus
try:
    import simplejson as json
except ImportError:
    import json

app = Flask(__name__)
swagger = Swagger(app)

vectorizer = joblib.load('vektorizer.pkl')
clf = joblib.load('clf.joblib')

m = Mystem()


def model(text):
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
    clear = " ".join(re.sub(r'[^а-яА-ЯёЁ ]', ' ', lemm_text).split())
    data = []
    data.append(clear)
    arr = vectorizer.transform(data)
    preds = clf.predict(arr.toarray())
    proba = clf.predict_proba(arr.toarray())[:, 1]
    d = dict()
    d["pred"] = preds[0]
    d["proba"] = proba[0]
    return d


@app.route('/response', methods=['POST'])
def post():
    """
    Response check endpoint
    ---
    tags:
      - response
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          id: ResponseAnalyze
          required:
            - text
          properties:
            text:
              description: Respinse text
              type: string
    responses:
      200:
        description: Successful operation
      400:
        description: Invalid input
    """
    rq = request.json['text']
    mod = str(model(rq))
    return jsonify(mod), HTTPStatus.OK


@app.route('/response', methods=['PUT'])
def put():
    """
    Response teach endpoint
    ---
    tags:
      - response
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          id: ResponseTeach
          required:
            - text
            - pred
          properties:
            text:
              description: Repsonse text
              type: string
            pred:
              desccription: Response prediction (0 or 1)
              type: integer
    responses:
      200:
        description: Successful operation
      400:
        description: Invalid input
    """
    return jsonify(""), HTTPStatus.OK


app.run(host='0.0.0.0')
