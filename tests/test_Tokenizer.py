# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import os
import codecs
from data import PROJECT_PATH
# import pickle
from sklearn.externals import joblib

from models.train import *
from models.tokenizer import Tokenizer


class TrainTokenizerTests(unittest.TestCase):

    def normalize_text(self, text):
        # return text
        dict = {
            u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè',u'óe': u'oé',
            u'ỏe': u'oẻ',u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý',u'ủy': u'uỷ', u'ũy': u'uỹ',u'ụy': u'uỵ'
        }
        for k, v in dict.iteritems():
            text = text.replace(k, v)
        return text

    def load_word_dictionary(self):
        word_dictionary = dict()
        with codecs.open(os.path.join(PROJECT_PATH, 'data','full_dict_46k.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token] = len(word)
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'commune.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token.lower()] = len(word)
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'district.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token.lower()] = len(word)
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'province.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token.lower()] = len(word)
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token.lower()] = len(word)
        return word_dictionary

    def load_data_set(self):
        with codecs.open(os.path.join(PROJECT_PATH, 'data','train.txt'), 'r', encoding='utf-8') as f:
            tagged_sentences = list()
            for sent in f.read().rsplit('\n'):
                sent = self.normalize_text(sent)
                tagged_sentence = list()
                tagged_tokens = sent.split(' ')
                for tagged_token in tagged_tokens:
                    tagged_token = tuple(tagged_token.split("/"))
                    token = u'/'.join(tagged_token[:-1])
                    if len(token.split('_')) > 1:
                        syllables = token.split('_')
                        for syl in syllables[:-1]:
                            tagged_syl = tuple((syl, '1'))
                            tagged_sentence.append(tagged_syl)
                        tagged_syl = tuple((syllables[-1], '2'))
                        tagged_sentence.append(tagged_syl)
                    else:
                        tagged_syl = list((token, '0'))
                        tagged_sentence.append(tagged_syl)
                tagged_sentences.append(tagged_sentence)
        # print(tagged_sentences)
        return tagged_sentences

    def load_word_dictionary2(self):
        word_dictionary2 = dict()
        from models.conect_db import get_answers,get_entities,get_synonyms,get_train_data
        word_dictionary = []
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'lexicon.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                word_dictionary.append(token)
        entity = get_entities()
        synonym = get_synonyms()
        synonym_entity = list(set(synonym + entity))
        tagged_sentences = list(set(synonym_entity + word_dictionary))

        for token in tagged_sentences:
            token = self.normalize_text(token)
            word = token.split(' ')
            word_dictionary2[token.lower()] = len(word)
        return word_dictionary2

    def datasource(self):
        dataset = self.load_data_set()
        word_dictionary = self.load_word_dictionary2()

        return dataset, word_dictionary

    def test_train_tokenizer(self):

        trainer = TrainTokenizer()
        trainer.datasource = self.datasource
        trainer.is_overfitting = True
        model = trainer.train()

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/tokenizer.model'), 'wb') as f:
            joblib.dump(model, f)

    def test_tokenizer(self):

        sentences = [
            u'tôi sống ở hà nội, thủ đô yên bình của quốc gia việt nam. sài gòn ở phía nam trù phú và nhộn nhịp hơn rất nhiều',
            u'tôi sống ở An bình tây, thủ đô yên bình của quốc gia việt nam. sài gòn ở phía nam trù phú và nhộn nhịp hơn rất nhiều',

        ]
        import time


        with open(os.path.join(PROJECT_PATH, 'pretrained_models/tokenizer.model')) as f:
            model = joblib.load(f)
        tokenizer = Tokenizer(model=model)
        tokenizer.is_remove_punctuation = True
        self.is_strip_punctuation = True
        end = time.time()

        # tokenizer.synonyms = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            output = u' | '.join(tokens)
            start = time.time()

            print(output)
        print( start - end)

    def test_aaa(self):
        from models.conect_db import get_answers,get_entities,get_synonyms,get_train_data
        entity = get_entities()
        synonym = get_synonyms()
        synonym_entity = list(set(synonym + entity))
        print(synonym_entity)

if __name__ == '__main__':
    unittest.main()