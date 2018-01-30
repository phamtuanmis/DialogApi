# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import os
import codecs
from data import PROJECT_PATH
# import pickle
from sklearn.externals import joblib

from chappieml.train import *
from chappieml.tokenizer import ChappieTokenizer


class TrainTokenizerTests(unittest.TestCase):

    def normalize_text(self, text):
        return text
        # dict = {
        #     u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè',u'óe': u'oé',
        #     u'ỏe': u'oẻ',u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý',u'ủy': u'uỷ', u'ũy': u'uỹ',u'ụy': u'uỵ'
        # }
        # for k, v in dict.iteritems():
        #     text = text.replace(k, v)
        # return text

    def load_word_dictionary(self):
        word_dictionary = dict()
        with codecs.open(os.path.join(PROJECT_PATH, 'data','lexicon.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = self.normalize_text(token)
                word = token.split(' ')
                word_dictionary[token] = len(word)

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
                        tagged_syl = tuple((token, '0'))
                        tagged_sentence.append(tagged_syl)
                tagged_sentences.append(tagged_sentence)
        return tagged_sentences

    def datasource(self):
        dataset = self.load_data_set()
        word_dictionary = self.load_word_dictionary()

        return dataset, word_dictionary

    def test_train_tokenizer(self):

        trainer = ChappieTrainTokenizer()
        trainer.datasource = self.datasource
        model = trainer.train(test_size=0.9)

        with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model'), 'wb') as f:
            joblib.dump(model, f)

    def test_tokenizer(self):

        sentences = (
            u'Đôi khi, cô gái trẻ Hà Nội Nguyễn Thị Thanh Huyền mặc chiếc áo màu xanh hoà bình thoáng đãng và trang điểm như công chúa trong truyện cổ tích mỗi khi Hoàng Tuấn Lâm đến . Điều này khiến cho bố của nàng bực dọc vô cùng, vầng trán nhăn nhúm của ông lại giần giật liên hồi, khuôn mặt xám ngoét lại.',
            u'Sài Gòn nắng mưa thất thường lắm em  biết không?',
        )

        with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
            model = joblib.load(f)

        tokenizer = ChappieTokenizer(model=model)

        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            output = u' | '.join(tokens)
            print(output)










    def test_simple_tokenizer(self):

        sentences = (
            u'Đôi khi, cô gái trẻ Hà Nội Phùng Thị Lan Hương mặc chiếc áo màu xanh hoà bình thoáng đãng và trang điểm như công chúa trong truyện cổ tích mỗi khi Hoàng Tuấn Lâm đến . Điều này khiến cho bố của nàng bực dọc vô cùng, vầng trán nhăn nhúm của ông lại giần giật liên hồi, khuôn mặt xám ngoét lại.',
        )
        tokenizer = SimpleTokenizer()

        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            output = u' | '.join(tokens)
            print(output)


if __name__ == '__main__':
    unittest.main()