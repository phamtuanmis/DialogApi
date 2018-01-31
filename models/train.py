# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from sklearn_crfsuite import CRF

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import RidgeClassifier
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
        self.model.pipeline = None  # Pipeline()
        self.model.features = textwrap.dedent(inspect.getsource(self.features))
        self.model.use_tfidf = False

        self.is_overfitting = False

        self.tokenizer = tokenizer

        self.feature_extractions = [
            ('count', CountVectorizer(
                ngram_range=(1, 2),
                max_features=self.model.max_features,
                tokenizer=self.tokenizer.tokenize if self.tokenizer else None,
             )),
            ('dict', DictVectorizer(sparse=False))
        ]

        self.classifiers = [
            MultinomialNB,
        ]

        self.taggers = None
        self.dumper = None

    def set_feature_extraction(self, mode='dict'):
        if mode == 'dict':
            self.feature_extractions = [
                ('dict', DictVectorizer(sparse=False))
            ]
        elif mode == 'count':
            self.feature_extractions = [
                ('count', CountVectorizer(
                    ngram_range=(1, 2),
                    max_features=self.model.max_features,
                    tokenizer=self.tokenizer.tokenize if self.tokenizer else None,

                ))
            ]

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
                elif cls.__name__ == 'MLPClassifier':
                    cls_ = MLPClassifier(hidden_layer_sizes=(200,))
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

    def transform_to_dataset(self, tagged_sentences):
        X, y = [], []
        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                items = self.features(self.untag(tagged), index)
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    X.append(item)
                    y.append(tagged[index][-1])
        return X, y

    def crf_transform_to_dataset(self, tagged_sentences):
        Xs, ys = [], []
        for tagged in tagged_sentences:
            X, y = [], []
            for index in range(len(tagged)):
                items = self.features(self.untag(tagged), index)
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    X.append(item)
                    y.append(tagged[index][-1])
            Xs.append(X)
            ys.append(y)
        return Xs, ys

    def classify_transform_to_dataset(self, dataset):
        X, y = [], []
        for document, topic in dataset:
            items = self.features(document)
            if not isinstance(items, list):
                items = [items]
            for item in items:
                X.append(item)
                y.append(topic)
        return X, y

    def features__(self, document, index=0):
        return document, None

    def preprocessing(self):
        pass

    def train(self, test_size=0.25, dumper=None):

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

        # print('Dataset %s' % len(self.dataset))
        if len(self.dataset) == 0: return

        train_set, test_set = train_test_split(self.dataset, test_size=test_size, random_state=0)


        if not train_set or self.is_overfitting:
            train_set = self.dataset
            test_set = self.dataset

        best_feature_func = self.features

        taggers = list()

        if self.classifiers[0].__name__ == 'CRF':
            from sklearn_crfsuite import metrics
            # CRF algorithm
            X_train, y_train = self.crf_transform_to_dataset(train_set)
            X_test, y_test = self.crf_transform_to_dataset(test_set)

            if dumper:
                self.dumper = dumper
                dumper(X_train, self.__class__.__name__.lower() + 'X_train.txt')
                dumper(X_test, self.__class__.__name__.lower() + 'X_test.txt')

            print('Train_set %s' % len(X_train))
            print('Test_set %s' % len(X_test))
            # print(len(X_train), len(y_train))

            clf = CRF()
            clf.fit(X_train, y_train)

            accuracy = clf.score(X_test, y_test)
            max_accuracy = accuracy

            #Print F1 score of each label
            if self.is_overfitting ==True:
                y_pred = clf.predict(X_test)
                classes = list(clf.classes_)
                labels = []
                for label in classes:
                    if label[:1]!='_':
                        labels.append(label)
                print(metrics.flat_classification_report(y_test, y_pred, labels = labels, digits = 3))
            else:
                accuracy = clf.score(X_test, y_test)
                max_accuracy = accuracy

        else:

            feature_func = self.features
            best_feature_func = feature_func

            for feature_extraction in self.feature_extractions:

                if feature_extraction[0] == 'count':
                    if isinstance(train_set[0][0], dict):
                        continue
                    self.features = self.features__
                else:
                    self.features = feature_func

                # train all classification models
                X_train, y_train = self.classify_transform_to_dataset(train_set)
                X_test, y_test = self.classify_transform_to_dataset(test_set)

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
                    # print(classes)
                    from sklearn import metrics
                    print(metrics.classification_report(y_test, y_pred,
                                                        target_names=classes,digits=3))
                    accuracy = clf.score(X_test, y_test)

                    # for y1,y2,x in zip(y_test,y_pred,X_test):
                    #     if y1!=y2:
                    #         print('Sentence: ',x)
                    #         print('True label',y1)
                    #         print('Predict label',y2)


                    print('feature extraction %s, classifier %s, accuracy: %s' % \
                          (feature_extraction[0], classifier.__name__, accuracy))

                    if accuracy >= max_accuracy:
                        max_accuracy = accuracy
                        best_classifier = clf
                        best_feature_func = self.features

        if not best_classifier:
            best_classifier = clf
            best_feature_func = self.features

        feature_extraction = 'dict' if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[0][0]

        classifier_name = best_classifier.__class__.__name__ if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[-1][1].__class__.__name__

        print('Number of labels: %i' % len(best_classifier.classes_))

        print('Best model: feature extraction %s, classifier %s, accuracy: %s' % \
              (feature_extraction, classifier_name, max_accuracy))

        self.model.pipeline = best_classifier

        # func = textwrap.dedent(inspect.getsource(best_feature_func))
        # self.model.features = func.replace('__(self,', '(self,')

        if feature_extraction == 'count':
            self.model.pipeline.steps[0][1].tokenizer = None

        self.model.build_version = time.time()

        self.dataset = zip(X_train, y_train)
        self.taggers = taggers

        return self.model

    def visualize(self, labels=None):
        mode = 'dict' if self.model.pipeline.__class__.__name__ == 'CRF' \
        else self.model.pipeline.steps[0][0]

        if not labels:
            labels = self.model.pipeline.classes_

        from sklearn.decomposition import PCA
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d

        dataset, word_dictionary = self.datasource()

        classifier_name = self.model.pipeline.__class__.__name__ if self.model.pipeline.__class__.__name__ == 'CRF' \
            else self.model.pipeline.steps[-1][1].__class__.__name__

        pipeline = Pipeline([
            (mode, self.model.pipeline.steps[0][1] if 'steps' in self.model.pipeline.__dict__ \
                else self.model.pipeline),
        ])

        if mode == 'count':
            self.features = self.features__

        if classifier_name == 'CRF':
            X_set, y_set = self.crf_transform_to_dataset(dataset)
        else:
            X_set, y_set = self.classify_transform_to_dataset(dataset)

        if mode == 'dict':
            X = pipeline.fit_transform(X_set, y_set)
        else:
            X = pipeline.fit_transform(X_set,y_set).todense()

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 8

        pca = PCA(n_components = 3).fit(X)
        data2D = pca.transform(X)
        n_label = len(labels)

        color = ['r','b','y','m','g','m','c']
        n = len(color)
        for i in range(n, n_label):
            color.append(str(1.0*i/n_label))
        mark  = [ '*', 'x', 'o','^', '+', '>', '<', 'p', '^', 'h', 'H', 'D', 'd']
        n = len(mark)
        for i in range(n, n_label):
            mark.append(mark[i % n])

        x = data2D[:, 0]
        y = data2D[:, 1]
        z = data2D[:, 2]
        plots = []
        for i in range(n_label):
            plot = None
            for j in range(len(x)):
                if y_set[j] == labels[i]:
                    plot = ax.scatter(x[j], y[j],z[j],c=color[i], marker=mark[i],label = labels[i])
            if plot:
                plots.append(plot)
        ax.legend(plots,labels)
        # ax.set_xlim3d(-100, 150)#limit range of X
        # ax.set_ylim3d(-60, 60)
        # ax.set_zlim3d(-60, 40)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class TrainTokenizer(Trainer):
    def __init__(self, tokenizer=None):
        super(TrainTokenizer, self).__init__()
        self.tokenizer = tokenizer
        self.classifiers = [
            CRF
        ]

    def features(self, sent, index=0):
        import string
        word = sent[index]

        features = {
            'word': word,
            'len':len(word),
            'word_lowwer': word.lower(),
            'word_upper': word.upper(),
            'is_first': index == 0,
            'is_last': index == len(sent) - 1,
            'word[:1]': word[:1],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            'word[:4]': word[:4],
            'word[:5]': word[:5],
            'word[:6]': word[:6],
            'word[-6:]': word[-6:],
            'word[-5:]': word[-5:],
            'word[-4:]': word[-4:],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            'word.is_lower': word.islower(),
            'word.is_upper': word.isupper(),
            'word.is_title': word.istitle(),
            'word.is_digit': word.isdigit(),
            'is_all_caps': word.upper() == word,
            'capitals_inside': word[1:].lower() != word[1:],
            'prev_word': '' if index == 0 else sent[index - 1],
            'prev_word2': ' ' if index == 0 or index == 1 else sent[index - 2],
            'next_word': '' if index == len(sent) - 1 else sent[index + 1],
            'next_word2': ' ' if index == len(sent) - 1 or index == len(sent) - 2 else sent[index + 2],
            'is_punctuation': word in string.punctuation

        }

        n_grams = (4, 3, 2)
        size_sent = len(sent)
        for n_gram in n_grams:
            tokens = list()
            for i in range(index, index + n_gram):
                if i < size_sent:
                    tokens.append(sent[i])

            word = ' '.join(tokens)
            gram = self.model.word_dictionary.get(word.lower(), -1) + 1
            feature_name = '%s-gram' % gram
            features.update({
                feature_name: gram > 0,
                '%s.word[0]'% feature_name: word.split(' ')[0],
                # '%s.word'% feature_name : word,
                # '%s.word.is_lower' % feature_name: word.islower(),
                # '%s.word.is_upper' % feature_name: word.isupper(),
                # '%s.word.is_title' % feature_name: word.istitle(),
                # '%s.word.is_digit' % feature_name: word.isdigit(),
                # '%s.is_all_caps' % feature_name: word.upper() == word,
                # '%s.capitals_inside': word[1:].lower() != word[1:],
            })
        return features




class TrainClassifier(Trainer):

    def __init__(self, tokenizer=None):

        super(TrainClassifier, self).__init__(tokenizer=tokenizer)

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        # from sklearn.neural_network import MLPClassifier

        self.classifiers = [
            # RandomForestClassifier,
            MultinomialNB,
            LinearSVC_proba,
            # RidgeClassifier,
            # DecisionTreeClassifier,
            LogisticRegression,
            # AdaBoostClassifier,
            SGDClassifier,
            KNeighborsClassifier,
            # MLPClassifier, # Multi-layer Perceptron Neural network
        ]

        # Predict probabilities for Sklearn LinearSVC
        # http://www.erogol.com/predict-probabilities-sklearn-linearsvc/

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
