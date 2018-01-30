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

import inspect
import textwrap
import re
import time
import math
from models import TrainModel

from models import LinearSVC_proba


class ChappieTrainer(object):
    '''
    Chappie Trainer
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


class ChappieTrainTokenizer(ChappieTrainer):
    def __init__(self, tokenizer=None):
        super(ChappieTrainTokenizer, self).__init__()
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


class ChappieTrainPosTagger(ChappieTrainer):
    def __init__(self, tokenizer=None):
        super(ChappieTrainPosTagger, self).__init__(tokenizer=tokenizer)
        self.classifiers = [
            CRF
        ]

    def features(self, sent, index=0):

        import string
        word = sent[index]
        return {
            'word': word,
            'is_first': index == 0,
            'is_last': index == len(sent) - 1,
            # 'is_capitalized': word[0].upper() == word[0],
            # 'is_second_capitalized': word[1].upper() == word[1] if len(word) > 1 else False,
            'word[:1]': word[:1],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            # 'word[:4]': word[:4],
            # 'word[:5]': word[:5],
            # 'word[:6]': word[:6],
            # 'word[:-6]': word[:-6],
            # 'word[-5:]': word[-5:],
            # 'word[-4:]': word[-4:],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            # 'word.is_lower': word.islower(),
            # 'word.is_upper': word.isupper(),
            'word.is_digit': word.isdigit(),
            'has_hyphen': '-' in word,
            'has_space': '_' in word,
            # 'capitals_inside': word[1:].lower() != word[1:],
            # 'capitals_': word[:1].upper() == word[:1],
            'prev_word': '' if index == 0 else sent[index - 1],
            'prev_word2': ' ' if index == 0 or index == 1 else sent[index - 2],
            'next_word': '' if index == len(sent) - 1 else sent[index + 1],
            'next_word2': ' ' if index == len(sent) - 1 or index == len(sent) - 2 else sent[index + 2],
            'is_punctuation': word in string.punctuation
        }


class ChappieTrainClassifier(ChappieTrainer):

    def __init__(self, tokenizer=None):

        super(ChappieTrainClassifier, self).__init__(tokenizer=tokenizer)

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


class ChappieTrainCarTagger(ChappieTrainPosTagger):
    def __init__(self, tokenizer=None):
        super(ChappieTrainCarTagger, self).__init__(tokenizer=tokenizer)
        self.classifiers = [
            CRF
        ]

    def features(self, sent, index=0):
        # import datetime
        # import re
        # import string
        # now = datetime.datetime.now()

        word = sent[index]
        prev_word = '' if index == 0 else sent[index - 1]
        prev2_word = '' if index < 2 else sent[index - 2]
        prev3_word = '' if index < 3 else sent[index - 3]
        prev20_word = '' if index < 2 else u' '.join(sent[index - 1 : index - 2])

        next_word = '' if index == len(sent) - 1 else sent[index + 1]
        next2_word = '' if index >= len(sent) - 2 else sent[index + 2]
        next3_word = '' if index >= len(sent) - 3 else sent[index + 3]
        next20_word = '' if index >= len(sent) - 2 else u' '.join(sent[index + 1 : index + 2])

        # prev_word_in_dict = self.model.word_dictionary.get(prev_word, [0,0,0])[1]
        word_in_dict = self.model.word_dictionary.get(word, [0,0,0])[1]
        length = len(word)

        return {
            'word': word,
            # 'lower_word': word.lower(),
            # 'is_first': index == 0,
            # 'is_last': index == len(sent) - 1,
            # 'is_capitalized': word[0].upper() == word[0],
            # 'is_second_capitalized': word[1].upper() == word[1] if len(word) > 1 else False,
            'word[:1]': word[:1], # prefix
            'word[:2]': word[:2] if length > 2 else '',
            'word[:3]': word[:3] if length > 3 else '',
            'word[:4]': word[:4] if length > 4 else '',
            'word[:5]': word[:5] if length > 5 else '',
            'word[:6]': word[:6] if length > 6 else '',
            'word[:-6]': word[:-6] if length > 6 else '',
            'word[:-5]': word[:-5] if length > 5 else '',
            'word[:-4]': word[:-4] if length > 4 else '',
            'word[:-3]': word[:-3] if length > 3 else '',
            'word[:-2]': word[:-2] if length > 2 else '',
            'word[:-1]': word[:-1], # suffix
            # 'word.is_lower': word.islower(),
            # 'word.is_upper': word.isupper(),
            'prev-word': prev_word,
            'prev2-word': prev2_word,
            'prev3-word': prev3_word,
            'prev20-word': prev20_word,
            'next-word': next_word,
            'next2-word': next2_word,
            'next3-word': next3_word,
            'next20-word': next20_word,
            # 'prev-word-in-dict': prev_word_in_dict,
            'word-in-dict': word_in_dict,
            # 'has_hyphen': '-' in word or '/' in word,
            'has-symbol': '@' in word,
            # 'is_digit': word.isdigit(),
            '2digit': word[0].isdigit() and word[-1].isdigit() and ('-' in word or '/' in word),
            '4digit': length == 4 and word.isdigit(),
            '10-11-digit': length in [10,11] and word.isdigit(),
            'number-with-comas': word[0].isdigit() and word[-1].isdigit() and ('.' in word or ',' in word),
            # 'first_digit': word[0].isdigit(),
            # 'last_digit': word[-1].isdigit(),
            # 'length': length,
            # 'has_space': '_' in word,
            # 'is_year': len(word) == 4 and word.isdigit() and now.year - 100 < int(word) < now.year + 100,
            # 'is_currency': len(re.findall(
            #     r'\b((\d{6,12})|(\d{1,3}[,.]\d{3}[,.]\d{3}[,.]\d{3}(\s+|$))|(\d{1,3}[,.]\d{3}[,.]\d{3})|(\d{1,2}[,.]\d{1,2}))(\s+|$)', word, re.DOTALL | re.UNICODE)) > 0,
            # 'is_datetime': len(re.findall(
            #     r'((\d{1,2})[\/|-](\d{1,2})[\/|-]((2[0|1]\d{2})|(19\d{2})|(0[0-9])))', word, re.DOTALL | re.UNICODE)) > 0,
            # 'is_datetime_short': len(re.findall(
            #     r'((\d{1,2})[\/|-]((2[0|1]\d{2})|(19\d{2})))', word, re.DOTALL | re.UNICODE)) > 0,
        }


class ChappieTrainIntentClassifier(ChappieTrainer):

    def __init__(self, tokenizer=None):

        super(ChappieTrainIntentClassifier, self).__init__(tokenizer=tokenizer)

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.intent_entities_split_regex = re.compile(self.model.intent_entities_split_regex, re.UNICODE | re.MULTILINE | re.DOTALL)

        self.classifiers = [
            # MultinomialNB,
            # LogisticRegression,
            # SGDClassifier,
            LinearSVC_proba,
        ]

        if not self.tokenizer:
            self.tokenizer = SimpleTokenizer(word_dictionary=self.model.word_dictionary)
            self.tokenizer.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)

        # self.set_feature_extraction(mode='count')

        # self.feature_extractions = [
        #     ('count', CountVectorizer(
        #         ngram_range=(1, 2),
        #         max_features=self.model.max_features,
        #         tokenizer=self.tokenizer.tokenize if self.tokenizer else None,
        #     ))
        # ]

        self.feature_extractions = [
            ('count', CountVectorizer(
                ngram_range=(1, 2),
                max_features=self.model.max_features,
                tokenizer=None,
                # strip_accents='unicode',
            ))
        ]


        self.entity_tagger = list()


    def features__(self, document, index=0):
        '''
        Count vectorizer features
        :param document:
        :param index:
        :return:
        '''

        import re

        tokens = self.tokenizer.tokenize(document)

        documents = [document]

        # entity regex dictionary
        entity_regex_dictionary = self.model.entity_regex_dictionary \
            if self.model.entity_regex_dictionary else dict()

        i = 0
        o = 0
        tagger = [(v,'O') for v in tokens]
        # print(tokens)
        while i < len(tokens):
            o += 1
            if o > 1000:
                print(document, tokens)
                raise ValueError('oops %d' % o)
            token = tokens[i]
            # print(token)
            is_car_model = (i > 0 and tokens[i - 1] == '(?CarBrand)' and token[:2] != '(?')
            is_digit = (token in self.model.digit_strings and i < len(tokens) - 1 and
                        (not tokens[i + 1].isdigit() and tokens[i + 1] != '(?CarYear)')) or \
                       token.isdigit() or \
                       '@' in token or \
                       is_car_model or \
                       (token[0].isdigit() and token[-1] in ['i', 'e', 'x', '-']) or \
                       (len(token) > 1 and token[-1].isdigit() and token[-2] in ['i', 'm', 'x', '-']) or \
                       (token[-1].isdigit() and len(token) < 4) or \
                       (token[0].isdigit() and u''.join(token[-2:]) in [u'ty', u'tr', u'tỷ']) or \
                       (token[0].isdigit() and u''.join(token[-5:]) in [u'trieu', u'triệu']) or \
                       (token[0].isdigit() and (token[-1].isdigit() or token[-1].lower() in ['l', 'g', 'd', 'v', 'e']) and
                        ('.' in token or '/' in token or '-' in token or ',' in token)) # 1.1.2 2/9/2017 2-9-2019 9-2017
            is_continue = False
            key_, name_, _ = self.model.word_dictionary.get(token, (None, None, None))
            if name_ or is_digit:
                is_match = False
                step = min(len(tokens) - 1, 4, i) if i > 1 else 1
                # step = 1
                for name in entity_regex_dictionary.keys():
                    if name != name_ and not is_digit: continue
                    st = ' '.join(tokens[i - step:])
                    if len(tokens) > 3: st = ' '.join([st, ' '.join(tokens[i: i + 3])])
                    st = st.strip()

                    for pattern in entity_regex_dictionary[name]:
                        match = re.findall(pattern[0].decode('utf-8'), st, re.UNICODE | re.DOTALL | re.IGNORECASE)
                        if match:
                            if isinstance(match[0], unicode):
                                match[0] = [match[0]]
                            m_ = match[0][pattern[1] - 1].strip()
                            # print(m_, match[0], token, m_ in token, name, pattern[0].decode('utf-8'))
                            # print(st, token)
                            if m_ and m_[:2] != '(?' and (m_ in token or token in m_):
                                if is_digit or tokens[i] in self.model.word_dictionary:
                                    # print(is_digit, tokens[i], self.model.word_dictionary.get(tokens[i]))
                                    is_match = True
                                    if is_car_model and name != 'CarModel' and not is_digit:
                                        # print(token, name)
                                        continue
                                    if is_car_model and name_ and name_ != 'CarModel':
                                        name = name_
                                    tagger[i] = (tokens[i], name)
                                    tokens[i] = u'(?%s)' % name
                                    st = u' '.join(tokens)
                                    # check previous token
                                    i -= step if i > step else 0
                                    is_continue = True

                if not is_match and token in self.model.word_dictionary:
                    if name_ and (name_ not in entity_regex_dictionary or name_ in ['CarModel']):
                        tagger[i] = (tokens[i], name_)
                        tokens[i] = u'(?%s)' % name_
                        # check previous token
                        i -= step if i > step else 0
                        is_continue = True

            elif tokens[i][:2] != '(?':
                tokens[i] = tokens[i].lower()
                tagger[i] = (tokens[i], 'O')

            if is_continue:
                continue

            i += 1

        new_document = u' '.join(tokens)
        if document != new_document:
            documents.append(new_document)

        # print(documents)
        # print(tagger)
        return documents, tagger

    def features(self, document, index=0):
        '''
        Dict vectorizer features
        :param document:
        :param index:
        :return:
        '''

        feature_set = dict()

        single_word_score = 1.0
        next_word_score = 2.0
        phrase_word_score = 4.0

        tokens_ = self.intent_entities_split_regex.split(document)

        document_ = []
        for token in tokens_:
            if not token or token[:2] == '(?':continue
            token = token.strip()
            if not token: continue
            # print('* %s %i' % (token, score))
            score = phrase_word_score * len(token)/len(document)
            # score = len(token)
            feature_set[token] = score

            document_.append(token.lower())

        document = u' '.join(document_)

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(document)
        else:
            tokens = self.punct_regex.findall(document)

        i = 0
        # print(u'|'.join(tokens))
        for token in tokens:
            # set weight of token = 1
            if token not in feature_set:
                if token in self.model.word_dictionary:
                    token = self.model.word_dictionary[token]

                # if token in feature_set:
                #     feature_set[token] += single_word_score
                # else:
                #     feature_set[token] = single_word_score

                feature_set[token] = single_word_score

            # set weight of next word
            grams = [4, 3, 2, 1]
            for n in grams:
                if i < len(tokens) - n:
                    p = [token]
                    p.extend(tokens[i + 1: i + 1 + n])
                    token_ = u' '.join(p)
                    if token_.lower() != token_:
                        continue
                    # print(token_)
                    if token_ not in feature_set and token_.lower() == token_:
                        feature_set[token_] = next_word_score * len(token_)/len(tokens)
                        # feature_set[token_] = single_word_score
            i += 1

        if self.model.max_features > 0:
            import operator
            feature_set = dict(sorted(feature_set.items(), key=operator.itemgetter(0), reverse=True)[:self.model.max_features])

        return feature_set

    # def train_entities(self):
    #
    #     import string
    #
    #     intent_models = dict()
    #     word_dictionary = dict()
    #     strip_pattern = string.punctuation.replace('(', '').replace(')', '')
    #
    #     # build pos tagger train set from intent's samples
    #     intent_entities = dict()
    #     tag_labels = dict()
    #     tag_index = dict()
    #
    #     for sample, intent_name in self.dataset:
    #         if '(?' not in sample: continue
    #         # print(sample, intent_name)
    #
    #         if intent_name not in tag_index:
    #             tag_index[intent_name] = 0
    #
    #         index = tag_index[intent_name]
    #
    #         if intent_name not in intent_entities:
    #             intent_entities[intent_name] = list()
    #
    #         if intent_name not in tag_labels:
    #             tag_labels[intent_name] = dict()
    #
    #         components = sample.split(' ')
    #
    #         entity_name = ''
    #         tags = list()
    #         i = 0
    #         has_entity = False
    #         for component in components:
    #             component = component.strip(strip_pattern)
    #             if not component: continue
    #             if component[:2] == '(?':
    #                 entity_name = component[2:-1]
    #                 has_entity = True
    #                 tags.append(tuple(('*' + str(i), entity_name)))
    #                 for i_ in range(i - 1, -1, -1):
    #                     sample_, name_ = tags[i_]
    #                     if name_ == '_':
    #                         if sample_ in tag_labels[intent_name]:
    #                             label = tag_labels[intent_name][sample_]
    #                         else:
    #                             label = '_' + entity_name + str(i_)
    #                         tags[i_] = tuple((sample_, label))
    #                         tag_labels[intent_name][sample_] = label
    #             else:
    #                 tag = tuple((component, '_' + entity_name))
    #                 tags.append(tag)
    #                 word_dictionary[component] = 1
    #             i += 1
    #         index += 1
    #
    #         tag_index[intent_name] = index
    #         if has_entity:
    #             # print(sample)
    #             # print(tags)
    #             # print(' | '. join(['/'.join(v) for v in tags]))
    #             intent_entities[intent_name].append(tags)
    #
    #     # intent_entities['find_car'] = []
    #     # tags = [('*0', u'CarBrand'), (u'gia', u'_CarBrand'), (u'tu', u'_CarBrand'), ('*3', u'CarPrice'),
    #     #      (u'den', u'_CarPrice'), ('*5', u'CarPrice'), (u'ty', u'_CarPrice')]
    #     # intent_entities['find_car'].append(tags)
    #
    #     for intent_name in intent_entities:
    #         dataset = intent_entities[intent_name]
    #
    #         def datasource():
    #             return dataset, dict()
    #
    #         trainer = ChappieTrainPosTagger()
    #         trainer.datasource = datasource
    #         trainer.is_overfitting = True
    #         model = trainer.train(dumper=self.dumper)
    #
    #         intent_models[intent_name] = model
    #
    #     self.intent_models = intent_models

    def train_taggers(self, postagger_class=None):

        taggers = self.taggers
        def datasource():
            return taggers, dict()

        trainer = ChappieTrainPosTagger() if not postagger_class else postagger_class()
        trainer.datasource = datasource
        trainer.is_overfitting = True
        model = trainer.train(dumper=self.dumper)

        self.entity_tagger = model


class ChappieTrainArticleClassifier(ChappieTrainer):

    def __init__(self, tokenizer=None):

        super(ChappieTrainArticleClassifier, self).__init__(tokenizer=tokenizer)

        self.classifiers = [
            # MultinomialNB,
            LinearSVC_proba,
            DecisionTreeClassifier,
            LogisticRegression,
            # KNeighborsClassifier,
            SGDClassifier
        ]

        self.set_feature_extraction('dict')

    def features(self, document, index=0, feature_extraction='dict'):

        if feature_extraction == 'count':
            return document

        from lxml import html as html_parser

        features = dict()

        ignored_tags = ['html', 'body', 'script', 'head', 'meta', 'link', 'style']
        inline_tag = ['li', 'blockquote', 'span', 'li', 'a', 'b', '', 'strong', 'code']
        para_tag = ['p', 'li', 'blockquote', 'code', 'td']

        doc = html_parser.fromstring(document)
        # previous_tag = ''
        # previous_2_tag = ''
        tags_heap = list()
        for el in doc.iter():

            key = str(el.tag).lower()
            if key in ignored_tags: continue

            tags_heap.append(key)

            try:
                text = el.text_content()
            except:
                text = el.text

            # if key in para_tag:

            if key in features:
                features[key] += 1
            else:
                features[key] = 1

            features[key + '-text-length'] = len(text) if text else 1

            if key in inline_tag:
                features[key + '-tags-heap'] = '/'.join(tags_heap)
                tags_heap =list()

        return features, None




class ChappieTrainSentiment(ChappieTrainer):
    '''
    Extend class of Chappietrainer for Sentiment Analysi
    '''

    def classify_transform_to_dataset(self, dataset):
        from preprocessing import preprocessing_sentiment

        X, y = [], []
        taggers = list()
        for document, topic in dataset:
            if topic[:9] == 'sentiment':

            #     items, tagger = self.features(document)
            #     taggers.append(tagger)
            #     if not isinstance(items, list):
            #         items = [items]
            #     tags = []
            #     if (len(items)>1):
            #         X.append(items[1]+' abczyz123 '+' '.join(tags))
            #         y.append(topic)
            #     for tag in tagger:
            #         if tag[1] !='O':
            #             tags.append(tag[1])
            #     X.append(document+' abczyz123 '+' '.join(tags))
            #     y.append(topic)
            # else:

                if topic[-8:]=='positive' or topic[-8:]=='negative':
                    X.append(document)
                    y.append(topic[-8:])
                elif topic[-7:]=='neutral':
                    X.append(document)
                    y.append(topic[-7:])
        print('Data set %s' % len(X))
        X = preprocessing_sentiment(X) #ngoại thất thiết kế đồng điệu POS và kém NAG hiệu quả-N
        return X, y, taggers