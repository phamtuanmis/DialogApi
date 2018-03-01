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
from models.postagger import PosTagger


class TrainPosTaggerTests(unittest.TestCase):

    def normalize_text(self, text):
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
        trainer.train(test_size=0.5)

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/postagger.model'), 'w') as f:
            joblib.dump(trainer.model, f)

        # trainer.visualize()

    def test_postagger(self):

        sentences = (
            u'nhà em có ở an bình tây minh hóa đâu, em ở mường khương bắc kạn mà các bác',
            u'cho tôi hỏi ở thị trấn thanh chương huyện thanh chương nghệ an thì mua tràng phục linh ở đâu',
            u'tôi ở mèo vạc có ở xuân thủy cầu giấy đâu nhỉ hà nội hay hà nam thì cũng thế thôi, ở xã minh quán huyện trấn yên yên bái là nhất',
            u'hôm qua lang thang em bắt xe đi đô lương nghệ an nhưng thế nào lại lên nhầm xe phù ninh phú thọ cuối cùng em phải xuống xe ở vĩnh phúc'
        )

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/tokenizer.model')) as f:
            model = joblib.load(f)
        tokenizer = Tokenizer(model=model)

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model')) as f:
            model = joblib.load(f)
        tagger = PosTagger(model=model, tokenizer=tokenizer)

        for sentence in sentences:
            print('.'*100)
            tokens = tagger.predict(sentence)
            for token, tag in tokens:
                if tag!='0':
                    print(token,'\t\t', tag)


if __name__ == '__main__':
    unittest.main()