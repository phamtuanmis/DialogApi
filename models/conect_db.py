import MySQLdb

def get_train_data():
    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         db="chatbot")

    cur = db.cursor()
    cur.execute("SELECT content,intent from samples, intents where samples.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_answers():
    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT intent, answer from intents,answers where answers.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata

def get_entities():
    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         db="chatbot")
    cur = db.cursor()
    cur.execute("SELECT content, entity from entity_samples,entities where entity_samples.entity_id = entities.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata
# for dat in get_train_data():
#     print dat[0],dat[1]