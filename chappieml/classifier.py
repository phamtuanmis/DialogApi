# -*- coding: utf-8 -*-
from __future__ import print_function
import re
from chappieml.postagger import ChappiePosTagger
from chappieml.tokenizer import SimpleTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from lxml import html as html_parser
import string
import time


class ChappieClassifier(object):
    '''
    Chappie Classifier
    '''
    def __init__(self, model, tokenizer=None, separator=' '):

        self.separator = separator

        self.model = model
        self.tokenizer = tokenizer
        if tokenizer and separator:
            self.tokenizer.separator = separator

        self.predict_method = self.model.pipeline.predict_single \
            if self.model.pipeline.__class__.__name__ == 'CRF' else \
            self.model.pipeline.predict

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary

        # if self.tokenizer and self.model.pipeline.__class__.__name__ != 'CRF' and \
        #         self.model.pipeline.steps[0][0] == 'count':
        #     self.model.pipeline.steps[0][1].tokenizer = self.tokenizer.tokenize

        exec(self.model.features, self.__dict__)

    def predict(self, document):

        if not self.model.pipeline:
            raise Exception('Need load model first')

        labels = self.predict_method(
            [self.features(self, document)])

        return labels


class ChappieIntentClassifier(object):
    '''
    Chappie Intent Classifier
    '''
    def __init__(self, model, entity_tagger=None, tokenizer=None, separator=' '):

        self.separator = separator

        self.model = model
        self.entity_tagger = entity_tagger

        self.MIN_CONFIDENCE = 0.0
        self.GAP_CONFIDENCE = 0.20

        self.tokenizer = tokenizer
        if tokenizer and separator:
            self.tokenizer.separator = separator

        self.tag_tokenizer = SimpleTokenizer(word_dictionary=self.model.word_dictionary)
        self.tag_tokenizer.synonyms = self.model.synonyms

        if not self.tokenizer:
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.synonyms = self.model.synonyms

        try:
            self.predict_method = self.model.pipeline.predict_proba
        except:
            self.predict_method = self.model.pipeline.predict

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        if 'intent_entities_split_regex' in self.model.__dict__:
            self.intent_entities_split_regex = re.compile(self.model.intent_entities_split_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary
        # self.parent_dictionary = self.model.parent_dictionary

        if self.tokenizer and self.model.pipeline.__class__.__name__ != 'CRF' and \
                self.model.pipeline.steps[0][0] == 'count':
            self.model.pipeline.steps[0][1].tokenizer = self.tokenizer.tokenize

        # exec(self.model.features, self.__dict__)

    def predict(self, sents):

        # return zip(self.model.pipeline.classes_, probs[0])

        # sent = ' '.join(self.tokenizer.tokenize(sent))
        # print(sent)

        # sents = self.features(sent)
        # sents, _ = self.features(self, sent)
        # print(sents)

        probs = self.predict_method(sents)

        if self.predict_method.__name__ == 'predict_proba':

            classes = self.model.pipeline.classes_
            results = list()
            index = 0
            for prob in probs:
                i = 0
                result = None
                max_accuracy = 0
                for accuracy in prob:
                    if accuracy > max_accuracy:
                        result = (classes[i], accuracy)
                        max_accuracy = accuracy
                    i += 1
                if len(results) > 0 and results[-1][0] == result[0]:
                    if results[-1][1] < result[1]:
                        results[-1] = result
                else:
                    results.append(result)
                sent = sents[index]
                index += 1
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results = [(probs[0], 1)]
        return results, sent

    def determine_(self, sent):
        '''
        Entities extraction
        :param sent:
        :return:
        '''

        # predict tags from intent model
        if self.entity_tagger:
            tagger = ChappiePosTagger(model=self.entity_tagger, tokenizer=self.tag_tokenizer)
            tags = tagger.predict(sent)
        else:
            tags = []
            tagger = None

        # print(tags)

        sents = [sent]
        # sents.append('_'.join(sent.split(' ')))
        sent_ = ' '.join([v[0] if v[1] == 'O' else ('(?%s)' % v[1]) for v in tags])
        if sent_ != sent: sents.append(sent_)

        # print(sents)

        # predict intent name (intent classifier)
        probs, sent_ = self.predict(sents)
        # print(sent_)

        results = list()

        # print(probs)

        for prob in probs:

            result = dict()
            result['slots'] = dict()
            # result['tagger'] = tags

            intent_name = prob[0]

            result['intent_name'] = intent_name
            result['confidence'] = prob[1]
            result['build'] = self.model.build_version
            result['created_time'] = time.time()

            def set_entity(name, value):
                key_, value_, coord_ = self.word_dictionary.get(value, (None, None, None))
                if key_ and isinstance(coord_, tuple) and value_ == name: value = (key_, coord_)
                elif isinstance(coord_, tuple) and value_ == name: value = (value, coord_)
                elif key_: value = key_
                if name in result['slots'] and isinstance(result['slots'][name], list):
                    if value not in result['slots'][name]:
                        result['slots'][name].append(value)
                elif name in result['slots'] and \
                        value not in result['slots'][name] and \
                                result['slots'][name] != value:
                    result['slots'][name] = [result['slots'][name], value]
                else:
                    result['slots'][name] = value

                # extract car brand
                if name in ['CarModel', 'CarSeries']:
                    comp = value.split(' ')
                    if len(comp) > 0:
                        v = comp[0]
                        _ = None
                        x, y = self.parent_dictionary.get(v, (None, None))
                        if y != 'CarBrand':
                            x, y, _ = self.word_dictionary.get(v, (None, None, None))
                            if y != 'CarBrand':
                                v = u' '.join(comp[:2])
                                x, y = self.parent_dictionary.get(v, (None, None))
                                if y != 'CarBrand':
                                    x, y, _ = self.word_dictionary.get(v, (None, None, None))

                        if y == 'CarBrand':
                            if y not in result['slots']:
                                result['slots'][y] = list()
                            if not isinstance(result['slots'][y], list):
                                result['slots'][y] = [result['slots'][y]]
                            v_ = x or _ or v
                            if v_ not in result['slots'][y]:
                                result['slots'][y].append(v_)

            # values_ = list() # merge nearest neighbor value of the same entity name
            removed_entities = dict() # removed entities

            for value, entity in tags:
                if entity == 'O': continue
                set_entity(entity, value)

            for k,v in result['slots'].items():
                if isinstance(v, list) and len(v) == 1:
                    result['slots'][k] = v[0]

            # developing conversation with retain slots
            requires = dict()
            if tagger:
                for v in tagger.model.pipeline.classes_:
                    if v != 'O' and v not in result['slots'] and v not in removed_entities:
                        requires[v] = 1
            if requires:
                result['developing'] = list(requires)

            # map intent title
            result['intent_title'] = self.model.intents.get(result['intent_name'], '')
            results.append(result)
        return results

    def determine(self, sent):
        results = self.determine_(sent)
        return results[0] if results else {}


class ChappieArticleClassifier(ChappieClassifier):
    '''
    Chappie Article Classification and Extraction
    '''

    @staticmethod
    def similarity(text1, text2):
        f = CountVectorizer().fit_transform([text1, text2])
        return (f * f.T).A[0, 1]

    def get_element_text(self, node, heading, ignored_tags):
        if node is None: return ''
        starting = False
        if heading is not None:
            for el in [e for e in node.iter()]:
                if el == heading:
                    starting = True
                if not starting:
                    el.getparent().remove(el)

        for el in [e for e in node.iter()]:
            if not isinstance(el.tag, str):
                el.getparent().remove(el)
                continue
            key = str(el.tag).lower()
            if key in ignored_tags:
                if el.getparent() is not None:
                    el.getparent().remove(el)

        return node.text_content()

    def extract_article(self, document):

        if self.predict(document) != 'article':
            raise Exception('Invalid article content')

        ignored_tags = ['html', 'body', 'script', 'head', 'meta', 'link', 'noscript', 'iframe',
                        'select', 'option', 'input', 'textarea', 'style', 'title', 'dt', 'dd', 'dl', 'font']

        # headings = ['h1', 'h2', 'h3']

        # inline_tags = ['span', 'i', 'strong', 'b', 'em', 'li']
        candidate_tags = ['p', 'br']

        doc = html_parser.fromstring(document)
        max = 0
        candidate = None

        heading = None
        parent = None

        title = doc.find('.//title')
        title = title.text.strip() if title is not None else ''

        article_title = None
        for name in ['h1', 'h2', 'h3']:
            article_title = doc.find('.//' + name)
            title_ = article_title.text.strip() if article_title is not None and article_title.text else ''
            if title_ and article_title is not None and title_ in title:
                title = title_
                break

        # find candidate with max text length
        cosin_max = 0
        i = 0
        for el in doc.iter():
            if not isinstance(el.tag, str):
                continue
            key = str(el.tag).lower()
            if key in ignored_tags: continue

            # try:
            #     text = el.text_content()
            # except:
            #     text = el.text if el.text else ''

            text = ''
            # n = len(text)

            n = 0
            for elx in el.getchildren():
                key = str(elx.tag).lower()
                if key in candidate_tags:
                    n += 1

            if max < n:
                max = n
                candidate = (key, el, text, max)
                # print(el.get('class'))

        parent = candidate[1].getparent()

        while parent is not None and len(parent) < 4:
            parent = parent.getparent()

        text = self.get_element_text(parent, heading, ignored_tags)

        # find sub candidate by element contains max p
        p_candidate = None
        max = 0
        for el in candidate[1].iter():
            ps = el.findall('.//p')
            if ps is not None and max < len(ps):
                max = len(ps)
                p_candidate = el

        # let's try agian
        if article_title is not None and p_candidate is None and parent.tag == 'body':
            while article_title is not None and len(article_title) < 2:
                article_title = article_title.getparent()
            parent = article_title.getparent()
            heading = article_title
            text = self.get_element_text(parent, heading, ignored_tags)

        p_text = self.get_element_text(p_candidate, None, ignored_tags)

        # compare 2 sub candidates by text length
        if len(text) < len(p_text):
            text = text + '\n' + p_text
            parent = p_candidate

        # if title not in text:
        #     text = title + '\n' + text

        text = text.strip()

        # find base url
        base_url = ''
        links = doc.findall('.//link')
        for link in links:
            if link.get('rel') == 'canonical':
                base_url = link.get('href')
                break
        if not base_url:
            links = doc.findall('.//meta')
            for link in links:
                if link.get('property') == 'og:url':
                    base_url = link.get('content')
                    break

        if base_url:
            base_url = '/'.join(base_url.split('/')[:-1])

        images = list()
        for v in parent.findall('.//img'):
            src = v.get('src')

            if src[0] == '/':
                src = '/'.join(base_url.split('/')[:-1]) + src
            elif src[0] == '../':
                src = '/'.join(base_url.split('/')[:-2]) + src
            elif src[:4] != 'http':
                src = base_url + '/' + src

            images.append(src)

        return text, images, title



