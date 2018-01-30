# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from tokenizer import SimpleTokenizer
from sklearn.neural_network import MLPClassifier
import inspect
import textwrap
import re
import time
import math
from models import TrainModel

from models import LinearSVC_proba


class Trainer(object):
    '''
    Trainer
    '''
    def __init__(self, tokenizer=None, separator=' '):

        self.separator = separator

        self.model = TrainModel()

        self.model.word_dictionary = dict()
        self.model.pipeline = None
        self.model.features = textwrap.dedent(inspect.getsource(self.features))
        self.model.use_tfidf = False

        self.is_overfitting = False

        self.tokenizer = tokenizer

        self.feature_extractions = [
            ('count', CountVectorizer(
                ngram_range=(1,2),
                max_features=self.model.max_features,
                tokenizer=self.tokenizer.tokenize if self.tokenizer else None,
             )),
            # ('dict', DictVectorizer(sparse=False))
        ]

        self.classifiers = [
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

        self.taggers = None
        self.dumper = None


    def get_classifier(self, cls):
        cls_ = None
        for c in self.classifiers:
            if c == cls:
                if cls.__name__ == 'LogisticRegesstion':
                    cls_ = LogisticRegression(penalty='l2', dual=False, tol=0.01, max_iter=60, )
                elif cls.__name__ == 'AdaBoostClassifier':
                    cls_ = AdaBoostClassifier(n_estimators=100)
                elif cls.__name__ == 'RandomForestClassifier':
                    cls_ = RandomForestClassifier(n_estimators=300)
                else:
                    cls_ = c()
        return ('classifier', cls_)

    def datasource(self):
        '''
        Set data source to self.dataset and self.word_dictionary
        :return: (dataset, word_dictionary)
        '''
        return list((), {})

    def features(self, sent, index=0):
        word = sent[index]
        return {
            'word': word,
        }

    def untag(self, tagged_sentence):
        return [w for w, t in tagged_sentence]


    def classify_transform_to_dataset(self, dataset):
        X, y = [], []
        taggers = list()
        for document, topic in dataset:
            items, tagger = self.features(document)
            taggers.append(tagger)
            if not isinstance(items, list):
                items = [items]
            for item in items:
                # print(item)
                X.append(item)
                y.append(topic)
        return X, y, taggers

    def features__(self, document, index=0):
        return document, None

    def preprocessing(self):
        pass

    def train(self, test_size=0.5, dumper=None):

        dataset, word_dictionary = self.datasource()

        if self.tokenizer:
            self.model.synonyms = self.tokenizer.synonyms
            self.tokenizer.word_dictionary = word_dictionary
        self.model.word_dictionary = word_dictionary
        self.dataset = dataset

        self.preprocessing()

        best_classifier = None
        max_accuracy = 0
        clf = None

        print('Dataset %s' % len(self.dataset))
        if len(self.dataset) == 0: return

        train_set, test_set = train_test_split(self.dataset, test_size=test_size, random_state=42)


        if not train_set or self.is_overfitting:
            train_set = self.dataset
            test_set = self.dataset

        taggers = list()


        feature_func = self.features

        for feature_extraction in self.feature_extractions:

            if feature_extraction[0] == 'count':
                if isinstance(train_set[0][0], dict):
                    continue
                self.features = self.features__
            else:
                self.features = feature_func

            # train all classification models
            X_train, y_train, taggers = self.classify_transform_to_dataset(train_set)
            X_test, y_test, taggers = self.classify_transform_to_dataset(test_set)

            for classifier in self.classifiers:

                steps = list()
                steps.append(feature_extraction)
                if self.model.use_tfidf:
                    steps.append(('tfidf', TfidfTransformer()))
                    # steps.append(('kbest', SelectKBest(k=100)))
                steps.append(self.get_classifier(classifier))
                clf = Pipeline(steps)
                try:
                    clf.fit(X_train, y_train)
                except Exception as e:
                    print('ERROR', e)
                    continue

                y_pred = clf.predict(X_test)
                classes = list(clf.classes_)
                from sklearn import metrics
                print(metrics.classification_report(y_test, y_pred,
                                                    target_names=classes,digits=3))
                accuracy = clf.score(X_test, y_test)

                for y1,y2,x in zip(y_test,y_pred,X_test):
                    if y1!=y2:
                        print('Sentence: ',x)
                        print('True label',y1)
                        print('Predict label',y2)


                print('feature extraction %s, classifier %s, accuracy: %s' % \
                      (feature_extraction[0], classifier.__name__, accuracy))

                if accuracy >= max_accuracy:
                    max_accuracy = accuracy
                    best_classifier = clf

        if not best_classifier:
            best_classifier = clf

        feature_extraction = 'dict' if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[0][0]

        classifier_name = best_classifier.__class__.__name__ if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[-1][1].__class__.__name__

        print('Number of labels: %i' % len(best_classifier.classes_))

        print('Best model: feature extraction %s, classifier %s, accuracy: %s' % \
              (feature_extraction, classifier_name, max_accuracy))

        self.model.pipeline = best_classifier

        if feature_extraction == 'count':
            self.model.pipeline.steps[0][1].tokenizer = None

        self.model.build_version = time.time()

        self.dataset = zip(X_train, y_train)
        self.taggers = taggers

        return self.model



class TrainClassifier(Trainer):

    def __init__(self, tokenizer=None):

        super(TrainClassifier, self).__init__(tokenizer=tokenizer)

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)

    def features(self, document, index=0):

        import string
        import re

        sent = document.lower()

        # remove all punctuation
        sent = ''.join(ch for ch in sent if ch not in string.punctuation)

        # strip punctuation
        sent = sent.strip(string.punctuation)

        # remove multiple spaces
        sent = re.sub(r' +', ' ', sent).strip()

        # remove numbers
        sent = ''.join([i for i in sent if not i.isdigit()])

        feature_set = dict()

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(sent)
        else:
            tokens = self.punct_regex.findall(sent)

        for token in tokens:
            if token not in feature_set:
                feature_set[token] = 1
            else:
                feature_set[token] += 1
        return feature_set
