# -*- coding: utf-8 -*-
from __future__ import print_function

from pipeline import BasePipe
from chappieml.classifier import ChappieIntentClassifier
import nltk
import re


class DocumentToSentences(BasePipe):

    @staticmethod
    def document_to_sents(document):
        document_ = document
        document = document.replace('...', '.')
        document = re.sub(r'(\d+)\.(\d+)', r'\1_\2', document)
        regex = re.compile(
            r'\.|;'.decode('utf-8'), re.IGNORECASE)
        sents = []
        for v in regex.split(document):
            v = v.strip()
            if v: sents.append(v)

        regexs = [
            r'nếu\s+(.*)\s+thì(.*)$',
            r'so với\s+(.*)\s+thì(.*)$',
            r'mặc dù\s+(.*)\s+nhưng(.*)$',
            r'mặc cho\s+(.*)\s+nhưng(.*)$',
            r'dù cho\s+(.*)\s+nhưng(.*)$',
            r'dẫu\s+(.*)\s+nhưng(.*)$',
            r'tuy\s+(.*)\s+nhưng(.*)$',
            r'mặc dù\s+(.*)\s+tuy vậy(.*)$',
            r'mặc dù\s+(.*)\s+tuy nhiên(.*)$',
            r'không chỉ\s+(.*)\s+mà(.*)$',
            r'không những\s+(.*)\s+mà(.*)$',
        ]
        for pattern in regexs:
            match = re.findall(pattern.decode('utf-8'), document, re.IGNORECASE)
            if match and len(match[0]) > 1:
                sents.extend(match[0])
        return (sents, document_)

    def fit(self, response):

        content = response

        sents = nltk.sent_tokenize(content)
        candidates = []
        for sent in sents:
            sents_ = self.document_to_sents(sent)
            if sents_:
                candidates.append(sents_)

        return candidates


class ArticleDetermine(ChappieIntentClassifier):

    def fit(self, response):

        result = []
        intents = {}
        for sents, origin in response:
            previous_slots = {}

            for sent in sents:

                items = self.determine_(sent)
                for intent in items:
                    intent_name = intent.get('intent_name')
                    intents[intent_name] = intent.get('intent_title')

                    # print(intent_name, sent)
                    # print(intent)

                    # if sent is a part of long sentence
                    if not intent.get('slots') or \
                            (not intent.get('slots').get('CarSeries')
                             and not intent.get('slots').get('CarModel')
                             and not intent.get('slots').get('CarBrand')):
                        intent['slots'] = previous_slots

                    slots = intent.get('slots')
                    # for k,v in previous_slots.items():
                    #     slots[k] = v

                    result.append((intent_name, slots))

                    if intent.get('slots'):
                        previous_slots = intent.get('slots')

        return result


class ArticleSentiment(ChappieIntentClassifier):

    def fit(self, response):

        result = []
        intents = {}
        for sents, origin in response:
            previous_slots = {}

            for sent in sents:
                items = self.determine_(sent)
                for intent in items:
                    intent_name = intent.get('intent_name')
                    intents[intent_name] = intent.get('intent_title')

                    # print(intent_name, sent)
                    # print(intent)

                    # if sent is a part of long sentence
                    if not intent.get('slots') or \
                            (not intent.get('slots').get('CarSeries')
                             and not intent.get('slots').get('CarModel')
                             and not intent.get('slots').get('CarBrand')):
                        intent['slots'] = previous_slots

                    if intent_name[:10] == 'sentiment_':
                        # print(sent, intent_name, intent.get('slots'))

                        slots = intent.get('slots')
                        # for k,v in previous_slots.items():
                        #     slots[k] = v

                        result.append((intent_name, slots, origin))

                    if intent.get('slots'):
                        previous_slots = intent.get('slots')

        # return result

        items = dict()
        items['entities'] = dict()
        recent_entity = None
        car_brands = {}
        for intent_name, slots, origin in result:
            for slot in slots:

                if slot in ['CarYear'] and recent_entity:
                    items['entities'][recent_entity]['year'] = slots[slot]
                    continue

                if slot not in ['CarBrand', 'CarSeries', 'CarModel']: continue

                entities = slots[slot]
                if not isinstance(entities, list):
                    entities = [entities]

                if slot in ['CarBrand']:
                    for e_ in entities:
                        car_brands[e_] = slot

                for entity in entities:

                    # validate entity, its must be car brand/car series/car model
                    if self.word_dictionary.get(entity) not in ['CarBrand', 'CarSeries', 'CarModel']: continue

                    # try to merge with car brand
                    if slot in ['CarModel', 'CarSeries']:
                        for car_brand in car_brands.keys():
                            car_model = ('%s %s' % (car_brand, entity)).lower()
                            if self.word_dictionary.get(car_model) in ['CarModel', 'CarSeries']:
                                entity = car_model
                                break
                    # print(entity, self.word_dictionary.get(entity))

                    if entity not in items['entities']:
                        items['entities'][entity] = dict()
                    if 'sentiment' not in items['entities'][entity]:
                        items['entities'][entity]['sentiment'] = dict()
                    components = intent_name[10:].split('_')
                    items['entities'][entity]['type'] = slot
                    items['entities'][entity]['sentiment']['_'.join(components[:-1])] = components[-1]
                    items['entities'][entity]['origin'] = origin
                    recent_entity = entity

        items['intents'] = intents
        return items

