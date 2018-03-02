import MySQLdb

db = MySQLdb.connect(host="127.0.0.1",
                     user="root",
                     passwd="",
                     charset='utf8',
                     use_unicode=True,
                     db="chatbot")
cur = db.cursor()


def get_train_data():
    cur.execute("SELECT content,intent from samples, intents where samples.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_answers():
    cur.execute("SELECT intent, answer from intents,answers where answers.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_entities():
    cur.execute("SELECT content, entity from entity_samples,entities where entity_samples.entity_id = entities.id")
    mydata = [list(x) for x in cur.fetchall()]
    # db.close()
    return mydata

def get_synonyms():
    cur.execute("SELECT content from synonym")
    mydata = [list(x)[0] for x in cur.fetchall()]
    # db.close()
    return mydata

def test_entity_data():

    cur.execute("SELECT content from samples where samples.intent_id = '7'")
    mydata = [list(x) for x in cur.fetchall()]
    # db.close()
    return mydata