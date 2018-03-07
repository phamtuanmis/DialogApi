# !/usr/bin/env python
# -*- coding: utf-8 -*-
def get_train_data():
    import MySQLdb

    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         charset='utf8',
                         use_unicode=True,
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT content,intent from samples, intents where samples.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_answers():
    import MySQLdb

    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         charset='utf8',
                         use_unicode=True,
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT intent, answer from intents,answers where answers.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_entities():
    import MySQLdb

    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         charset='utf8',
                         use_unicode=True,
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT content, entity from entity_samples,entities where entity_samples.entity_id = entities.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_synonyms():
    import MySQLdb

    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         charset='utf8',
                         use_unicode=True,
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT entity_samples.content,synonym.content from entity_samples, synonym where entity_samples.id = synonym.entity_sample_id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


# def get_synonym_fromdb():
#     synonyms = get_synonyms()
#     return dict([[x[1], x[0]] for x in synonyms])

# synonyms = get_synonyms()
# my_list= []
# for synonym in synonyms:
#     my_list.append([synonym[1],synonym[0]])
# my_list = get_synonym_fromdb()
# print(my_list)
# token = u'Ha Noi'
# if token in my_list:
#     token = my_list.get(token)
#     print(token)# .lower()
