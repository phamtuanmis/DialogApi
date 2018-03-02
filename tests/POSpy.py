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
from models.tokenizer import Tokenizer,MyTokenizer
from models.postagger import PosTagger

class TrainPosTaggerTests(unittest.TestCase):

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


    def test_postagger(self):
        from models.conect_db import test_entity_data

        sentences = (
            u'nhà em có ở an bình tây minh hóa đâu, em ở mường khương bắc kạn mà các bác hồ chí minh',
            u'mua thuốc maxxhair ở thái thụy thái bình chỗ nào',
            u'cho tôi hỏi ở xã thanh chương huyện thanh chương nghệ an thì mua tràng phục linh ở đâu',
            u'tôi ở mèo vạc có ở xuân thủy cầu giấy đâu nhỉ hà nội hay hà nam thì cũng thế thôi, ở xã minh quán huyện trấn yên yên bái là nhất',
            u'hôm qua lang thang em bắt xe đi đô lương nghệ an nhưng thế nào lại lên nhầm xe phù ninh huyện Phú Thọ cuối cùng em phải xuống xe ở vĩnh phúc'
        )
        # with open(os.path.join(PROJECT_PATH, 'pretrained_models/tokenizer.model')) as f:
        #     model = joblib.load(f)
        tokenizer = MyTokenizer()
        # tokenizer.is_remove_punctuation = True

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model')) as f:
            model = joblib.load(f)
        tagger = PosTagger(model=model, tokenizer=tokenizer)

        for sent in sentences:
            sentence = sent
            # print(sentence)
            # print('-'*100)
            tokens = tagger.predict(sentence.lower())
            for token, tag in tokens:
                if tag!='0':
                    print(token,'\t\t', tag)

if __name__ == '__main__':
    unittest.main()