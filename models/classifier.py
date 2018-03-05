# -*- coding: utf-8 -*-
from __future__ import print_function
import re
# from models.tokenizer import SimpleTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from lxml import html as html_parser



class Classifier(object):
    '''
    Chappie Classifier
    '''
    def __init__(self, model, tokenizer=None, separator=' '):

        self.separator = separator
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer and separator:
            self.tokenizer.separator = separator

        self.predict_method = self.model.pipeline.predict
        #self.model.pipeline.predict_single \
        #     if self.model.pipeline.__class__.__name__ == 'CRF' else \
        #     self.model.pipeline.predict_proba

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary
        self.anwers = self.model.answers
        exec(self.model.features, self.__dict__)

    def predict(self, document):
        import numpy as np
        if not self.model.pipeline:
            raise Exception('Need load model first')
        labels = self.predict_method([self.features(self, document)])
        proba = self.model.pipeline.predict_proba([document])
        return labels
        # return [labels[0], np.amax(proba[0][0])]
