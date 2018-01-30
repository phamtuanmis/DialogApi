# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import os
from data import PROJECT_PATH
from sklearn.externals import joblib
import csv
from chappieml.train import *
from chappieml.tokenizer import *

class TrainClassifierTests(unittest.TestCase):

    def load_data_set(self):
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
                    # print(intent_name)
                    continue
                if intent_name:
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

        dataset = self.load_data_set()
        word_dictionary = self.load_intents_dictionary()
        return dataset, word_dictionary

    def test_train_intent_classifier(self):

        with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
            model = joblib.load(f)
        tokenizer = Tokenizer(model=model)

        trainer = TrainClassifier(tokenizer=tokenizer)
        trainer.datasource = self.datasource
        trainer.is_overfitting = False
        trainer.classifiers = [
            RandomForestClassifier,
            MultinomialNB,
            LinearSVC_proba,
            RidgeClassifier,
            DecisionTreeClassifier,
            LogisticRegression,
            AdaBoostClassifier,
            SGDClassifier,
            KNeighborsClassifier,
            MLPClassifier,
        ]
        trainer.tokenizer.synonyms = self.load_synonyms()

        model = trainer.train()

        with open(os.path.join(PROJECT_PATH, 'data/intents.model'), 'w') as f:
            joblib.dump(model, f)


if __name__ == '__main__':
    # unittest.main()
    pass