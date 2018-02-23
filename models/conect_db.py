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

# for dat in get_train_data():
#     print dat[0],dat[1]