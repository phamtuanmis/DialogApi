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
from flask_cors import CORS, cross_origin
import unittest
import os
from data import PROJECT_PATH
from sklearn.externals import joblib
import csv
from models.train import *
from models.classifier import *
from flask import Flask
from conect_db import get_train_data,get_answers


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

    def test_train_intent_classifier(self):

        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        tokenizer = None  # Tokenizer(model=model)
        trainer = TrainClassifier(tokenizer=tokenizer)
        trainer.datasource = self.datasource
        trainer.model.answers = get_answers()
        # trainer.is_overfitting = False
        trainer.classifiers = [
            # RandomForestClassifier,
            # MultinomialNB,
            LinearSVC_proba,
            # DecisionTreeClassifier,
            # LogisticRegression,
            # AdaBoostClassifier,
            # SGDClassifier,
            # KNeighborsClassifier,
            # MLPClassifier,
        ]
        # trainer.tokenizer.synonyms = self.load_synonyms()

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

    def test_classify_intent(self):
        with open(os.path.join(PROJECT_PATH, 'data/intents.model')) as f:
            model = joblib.load(f)
        classifier = Classifier(model=model)
        intent = classifier.predict(query)[0]
        answers = classifier.anwers
        print(intent)
        return [intent,'']
        # for data in answers:
        #     if intent == data[0]:
        #         return [intent,data[1]]
        #     else:
        #         return ['default',u'Em chưa được dạy vấn đề này ạ']

if __name__ == '__main__':
    pass