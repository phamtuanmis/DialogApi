# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import os
from data import PROJECT_PATH
from sklearn.externals import joblib
import csv
from models.train import *
from models.classifier import *

class TrainClassifierTests(unittest.TestCase):
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

        dataset = self.load_data_set2()
        word_dictionary = self.load_intents_dictionary()
        return dataset, word_dictionary

    def test_train_intent_classifier(self):

        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        tokenizer = None#Tokenizer(model=model)

        trainer = TrainClassifier(tokenizer=tokenizer)
        trainer.datasource = self.datasource
        trainer.is_overfitting = False
        trainer.model.use_tfidf = False
        trainer.classifiers = [
            # RandomForestClassifier,
            # MultinomialNB,
            # LinearSVC_proba,
            # DecisionTreeClassifier,
            LogisticRegression,
            # AdaBoostClassifier,
            # SGDClassifier,
            # KNeighborsClassifier,
            # MLPClassifier,
        ]
        # trainer.tokenizer.synonyms = self.load_synonyms()

        trainer.train()
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

if __name__ == '__main__':
    # unittest.main()
    pass