# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
from future.standard_library import install_aliases
import json
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import os
from flask import Flask
from Intent import IntentClassifier
from Entity import EntityClassifier
from models.extras import normalize_text
install_aliases()
app = Flask(__name__)
cors = CORS(app)

threshold_confidence = 0.5

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request! Thiếu thông tin'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/conversation', methods=['POST'])
@cross_origin()
def conversation():
    req = request.get_json(silent=True, force=True)
    # print(json.dumps(req,encoding='utf8',ensure_ascii=False,indent=4))
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processRequest(req):
    query = req["query"]
    query = normalize_text(query)
    sessionId = req["sessionId"]

    Intent_Class = IntentClassifier()
    intent = Intent_Class.classify_intent(query,threshold_confidence=threshold_confidence)

    entity_class = EntityClassifier()
    entities = entity_class.predict_entiy(query)

    response = {
        'sessionId': sessionId,
        'resolvedQuery': query,
        'intentName':intent[0],
        'confidence': intent[1],
        'response': intent[2],
        'entities':entities,
    }
    return response


@app.route('/train', methods=['POST'])
# @cross_origin()


def train():
    req = request.get_json(silent=True, force=True)
    res = processTrain(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def processTrain(req):
    botId = req["botId"]
    print('BOT ID = '+botId)
    intent_class = IntentClassifier()
    entity_class = EntityClassifier()
    intent_class.trainmodel()
    entity_class.train_entity_model()
    response = {'querry': 'Train done!','BotID':botId }
    return response


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port,host = '0.0.0.0')