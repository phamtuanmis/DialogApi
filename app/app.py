# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
from future.standard_library import install_aliases
import json
from flask import Flask, jsonify, request, make_response
import os.path
import sys
import re
import requests
import os
import codecs
from models.tokenizer import Tokenizer,MyTokenizer
from models.postagger import PosTagger
from flask_cors import CORS, cross_origin
import os
from data import PROJECT_PATH
from sklearn.externals import joblib
import csv
from models.train import *
from models.classifier import *
from flask import Flask
from conect_db import get_train_data,get_answers

install_aliases()
app = Flask(__name__)
cors = CORS(app)


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
    print(r)
    return r

def processRequest(req):
    query = req["query"]
    sessionId = req["sessionId"]
    myclass = TrainClassifierTests()
    result = myclass.classify_intent(query, sessionId)
    print(result)
    # # print(result)
    # intent = result[0]
    # intent_confident = result[1]
    # response = result[2]
    # # myclass = Entity_Classifier()
    # # entities = myclass.postagger(query)
    # print(req)
    # response = {
    #     'resolvedQuery': query,
    #     'intentName':intent,
    #     'response': response,
    #     'sessionId':sessionId,
    #     'confidence': intent_confident,
    #     # 'entities':entities,
    #
    # }
    return response


@app.route('/train', methods=['POST'])
@cross_origin()

def train():
    req = request.get_json(silent=True, force=True)
    res = processTrain(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processTrain(req):
    botId = req["botId"]
    myclass = TrainClassifierTests()
    myclass.trainmodel()
    response = {'querry': 'Train done!','BotID':botId }
    return response

class TrainClassifierTests():
    def load_data_set2(self):
        dataset = list()
        with open(os.path.join(PROJECT_PATH, 'data', 'intents.txt')) as f:
            lines = f.readlines()
            intent_name = ''
            for row in lines:
                sample = row.strip().decode('utf-8')
                if not sample: continue
                if sample[-1] == '_': continue
                if sample[:3] == '---':
                    intent_name = sample[3:].strip()
                    print(intent_name)
                    continue
                if intent_name:
                    dataset.append((sample, intent_name))

        return dataset

    def load_data_set(self):
        dataset = list()
        with open(os.path.join(PROJECT_PATH, 'data', 'intents.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = row['sample'].strip().lower().decode('utf-8')
                intent_name = row['intent_name'].strip()
                dataset.append((sample, intent_name))
        return dataset

    def load_synonyms(self):

        dataset = dict()
        with open(os.path.join(PROJECT_PATH, 'data', 'synonyms.txt')) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(',')
                for token in tokens[1:]:
                    dataset[token.strip().decode('utf-8')] = tokens[0].strip().decode('utf-8')

        return dataset

    def load_intents_dictionary(self):
        dictionary = dict()
        with open(os.path.join(PROJECT_PATH, 'data', 'intents_dictionary.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row['key'].strip().decode('utf-8')
                value = row['value'].strip().decode('utf-8')
                dictionary[key] = value
                dictionary[key.lower()] = value

        return dictionary

    def datasource(self):

        dataset = get_train_data()#self.load_data_set2()
        word_dictionary = self.load_intents_dictionary()
        return dataset, word_dictionary

    def trainmodel(self):

        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        tokenizer = None  # Tokenizer(model=model)
        trainer = TrainClassifier(tokenizer=tokenizer)
        trainer.datasource = self.datasource
        trainer.model.answers = get_answers()
        trainer.classifiers = [
            # RandomForestClassifier,
            # MultinomialNB,
            # LinearSVC_proba,
            # DecisionTreeClassifier,
            # LogisticRegression,
            # AdaBoostClassifier,
            # SGDClassifier,
            # KNeighborsClassifier,
            MLPClassifier,
        ]
        # trainer.tokenizer.synonyms = self.load_synonyms()
        # trainer.is_overfitting = False
        # trainer.train()
        trainer.is_overfitting = True
        model = trainer.train()
        with open(os.path.join(PROJECT_PATH, 'data/intents.model'), 'w') as f:
            joblib.dump(model, f)

    def test_classifier_intents(self):

        documents = [
            u'Xin chào',
            u'em là ai thế nhỉ',
            u'sao kém thế nhỉ',
            u'Cho hỏi chỗ mua thuốc',
            u'em ơi cho anh hỏi giá thuốc tràng phục linh thế nào nhỉ',
            u'giỏi đấy',
            u'thuốc vương bảo dùng như thế nào ấy nhỉ',
            u'cám ơn em nhé',
            u'tạm biệt cậu nhé',

        ]

        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        # tokenizer = Tokenizer(model=model)

        with open(os.path.join(PROJECT_PATH, 'data/intents.model')) as f:
            model = joblib.load(f)

        classifier = Classifier(model=model)
        for document in documents:
            labels = classifier.predict(document)
            print(labels)

    def classify_intent(self,query, sessionId):
        with open(os.path.join(PROJECT_PATH, 'data/intents.model')) as f:
            model = joblib.load(f)
        classifier = Classifier(model=model)
        intent = classifier.predict(query)
        answers = classifier.anwers
        # print(classifier.predict(query)[0])
        if intent[1] >= 0.5:
            for data in answers:
                if intent[0] == data[0]:
                    intent.append(data[1])
        else:
            intent[0]='not_define'
            intent.append(u'Chatbot chưa được học vấn đề này')

        return intent

class Entity_Classifier():

    def normalize_text(self, text):
        dict = {
            u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè',u'óe': u'oé',
            u'ỏe': u'oẻ',u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý',u'ủy': u'uỷ', u'ũy': u'uỹ',u'ụy': u'uỵ'
        }
        for k, v in dict.iteritems():
            text = text.replace(k, v)
        return text

    def load_word_dictionary(self):
        word_dictionary = dict()
        with codecs.open(os.path.join(PROJECT_PATH, 'data','lexicon.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token] = len(word)

        return word_dictionary

    def load_data_set(self):
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'train.txt'), 'r', encoding='utf-8') as f:
            tagged_sentences = []
            for sent in f.read().rsplit('\n'):
                tagged_sentence = []
                tokens = sent.split()
                for token in tokens:
                    pos = tuple(token.split("/"))
                    pos = (u'/'.join(pos[:-1]), pos[-1])
                    tagged_sentence.append(pos)
                tagged_sentences.append(tagged_sentence)
        return tagged_sentences

    def datasource(self):
        dataset = self.load_data_set()
        word_dictionary = self.load_word_dictionary()

        return dataset, word_dictionary

    def test_train_postagger(self):

        trainer = TrainPosTagger()
        trainer.datasource = self.datasource
        trainer.train()
        trainer.is_overfitting = True
        with open(os.path.join(PROJECT_PATH, 'pretrained_models/postagger.model'), 'w') as f:
            joblib.dump(trainer.model, f)


    def postagger(self,sent):
        from models.conect_db import test_entity_data

        tokenizer = MyTokenizer()

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model')) as f:
            model = joblib.load(f)
        tagger = PosTagger(model=model, tokenizer=tokenizer)

        tokens = tagger.predict(sent.lower())
        result = []
        for token, tag in tokens:
            if tag!='0':
                result.append([token, tag])

        return result

if __name__ == '__main__':
    # port = int(os.getenv('PORT', 5000))
    # print("Starting app on port %d" % port)
    # app.run(debug=False, port=port,host = '0.0.0.0')
    aclass = TrainClassifierTests()
    aclass.test_classifier_intents()
    # # myclass = Entity_Classifier()
    # # sent = {
    # #     "query": "Ở Xuân Trần Duy Hưng Cầu giấy Hà Nội thì mua thuốc Maxxhair chỗ nào",
    # #     "sessionId": "123456789"
    # # }
    # # processRequest(sent)
    # # print(myclass.postagger(sent))
    # query = u'Xin chào các bạn'
    # sesionId = '123'
    # myclass = TrainClassifierTests()
    # result = myclass.classify_intent(query, sesionId)
    # print(result)