def get_train_data():
    import MySQLdb
    db = MySQLdb.connect(host="127.0.0.1",
                         user="root",
                         passwd="",
                         db="chatbot")

    cur = db.cursor()

    # Use all the SQL you like
    cur.execute("SELECT content,intent from samples, intents where samples.intent_id = intents.id")
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata
# print(conectdb())


def get_answers():
    import MySQLdb
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

    import MySQLdb
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