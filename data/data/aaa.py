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

# dict = []
# mydict = []
# with codecs.open(os.path.join(PROJECT_PATH, 'data', 'lexicon.txt'), 'r', encoding='utf-8') as fin:
#     for token in fin.read().split('\n'):
#         dict.append(token)
# dict2 = []
# with codecs.open(os.path.join(PROJECT_PATH, 'data', 'Viet39K.txt'), 'r', encoding='utf-8') as fin:
#     for token in fin.read().split('\n'):
#         dict2.append(token)
#
# for token in dict2:
#     if token not in mydict:
#         mydict.append(token.lower())
#
# for token in dict:
#     if token.lower() not in mydict:
#         mydict.append(token.lower())
#
# # for x in mydict:
# #     print(x)
# with codecs.open(os.path.join(PROJECT_PATH, 'data', 'full_dict_46k.txt'), 'wb', encoding='utf-8') as file:
#     for line in sorted(mydict):
#         file.write(line)
#         file.write('\n')

from nltk.tokenize import regexp_tokenize
dict_list = []
# with codecs.open(os.path.join(PROJECT_PATH, 'data', 'lexicon.txt'), 'r', encoding='utf-8') as fin:
#     for token in fin.read().split('\n'):
#         dict_list.append(token)

with codecs.open(os.path.join(PROJECT_PATH, 'data', 'commune.txt'), 'r', encoding='utf-8') as fin:
    for token in fin.read().split('\n'):
        dict_list.append(token.lower())

with codecs.open(os.path.join(PROJECT_PATH, 'data', 'district.txt'), 'r', encoding='utf-8') as fin:
    for token in fin.read().split('\n'):
        dict_list.append(token.lower())

with codecs.open(os.path.join(PROJECT_PATH, 'data', 'province.txt'), 'r', encoding='utf-8') as fin:
    for token in fin.read().split('\n'):
        dict_list.append(token.lower())

with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
    for token in fin.read().split('\n'):
        dict_list.append(token.lower())

sentences = (
    u'nhà em có ở an bình tây minh hóa đâu, em ở mường khương bắc kạn mà các bác',
    u'mua thuốc maxxhair ở thái thụy thái bình chỗ nào',
    u'cho tôi hỏi ở thị trấn thanh chương huyện thanh chương nghệ an thì mua tràng phục linh ở đâu',
    u'tôi ở mèo vạc có ở xuân thủy cầu giấy đâu nhỉ hà nội hay hà nam thì cũng thế thôi, ở xã minh quán huyện trấn yên yên bái là nhất',
    u'hôm qua lang thang em bắt xe đi đô lương nghệ an nhưng thế nào lại lên nhầm xe phù ninh huyện Phú Thọ cuối cùng em phải xuống xe ở vĩnh phúc'
)
import time
start = time.time()
newdict = {}
for item in dict_list:
    dk = item.replace(" ", "_")
    newdict[item] = dk

newdict_sorted = sorted(newdict, key=len, reverse=True)

for sent in sentences:
    print(sent)

    for item in newdict_sorted:
        if item in sent:
            sent = sent.replace(item, newdict[item])
    res = regexp_tokenize(sent, pattern='\S+')
    end = time.time()
    print(end-start)
    for token in res:
        print(token)

# from models.tokenizer import SimpleTokenizer
# tokenize = SimpleTokenizer(word_dictionary=dict_list)
# import time
#
# end = time.time()
#
# for sent in sentences:
#     tokens = tokenize.tokenize(sent)
# for t in tokens:
#     print(t)
# start = time.time()
# print(end - start)
# from models.conect_db import get_synonyms,get_entities
# print(get_synonyms())