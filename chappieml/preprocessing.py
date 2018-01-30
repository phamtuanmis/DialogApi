# !/usr/bin/env python
# -*- coding: utf-8 -*-
def preprocessing_sentiment(sentences):
    import joblib
    import os
    from data import PROJECT_PATH
    from chappieml.tokenizer import ChappieTokenizer
    import codecs

    # level_list = []
    # with codecs.open(os.path.join(PROJECT_PATH, 'data/sent_dict/level_word.txt'), 'r', encoding='utf8') as fin:
    #     for token in fin.read().split('\n'):
    #         level_list.append(token)

    not_words = []
    with codecs.open(os.path.join(PROJECT_PATH, 'data/sentiment_dict/not_word.txt'), 'r', encoding='utf8') as fin:
        for token in fin.read().split('\n'):
            not_words.append(token)

    pos_list = []
    with codecs.open(os.path.join(PROJECT_PATH, 'data/sentiment_dict/pos_word.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            pos_list.append(token)

    nag_list = []
    with codecs.open(os.path.join(PROJECT_PATH, 'data/sentiment_dict/nag_word.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            nag_list.append(token)

    with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        model = joblib.load(f)
    tokenizer = ChappieTokenizer(model=model)
    # textt = u'điều đáng tiếc nhất của xe là khả năng tiết kiệm POS nhiên liệu'
    # print('|'.join(tokenizer.tokenize(textt)))
    result = []
    for text in sentences:
        my_sentence = []
        token = tokenizer.tokenize(text)
        n = len(token)
        for i in range(n):
            my_sentence.append(token[i])
            if token[i] in not_words:
                for j in range(1, 6):
                    if i< n-j:
                        # if token[i+1] in level_list:
                        #     # my_sentence.append('NPOS '+token[i+1]+'-N')
                        #     token[i + 1] = ''
                        if token[i+j] in pos_list:
                            my_sentence.append('NAG')
                            token[i + j] = token[i+j]+'-N'
                        if token[i+j] in nag_list:
                            my_sentence.append('POS')
                            token[i + j] = token[i+j]+'-N'
                        # continue
            else:
                if token[i] in pos_list:
                    my_sentence.append('POS')
                if token[i] in nag_list:
                    my_sentence.append('NAG')
        result.append(' '.join(my_sentence))
    return result

# import time
# texts = [u'ngoại thất thiết kế đồng điệu và kém hiệu quả']
# begin = time.time()
# print(preprocessing_sentiment(texts)[0])
# print(time.time()-begin)