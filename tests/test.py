def features(document, index=0):
    import string
    import re
    sent = document.lower()

    # remove all punctuation
    sent = ''.join(ch for ch in sent if ch not in string.punctuation)

    # strip punctuation
    sent = sent.strip(string.punctuation)

    # remove multiple spaces
    sent = re.sub(r' +', ' ', sent).strip()

    # remove numbers
    sent = ''.join([i for i in sent if not i.isdigit()])

    feature_set = dict()

    PUNCT_REGEX = r'((((\d{1,3}[.,])|)\d{1,4}[.,]\d{3}[.,]\d{3})|(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})|(\w+[\-\+]\w+)|([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(\w+[.,\/\+]\w+)|(\w+\.)|(\w+|[^\w\s]+))'
    punct_regex = re.compile(PUNCT_REGEX, re.UNICODE | re.MULTILINE | re.DOTALL)
    tokens = punct_regex.findall(sent)

    for token in tokens:
        if token not in feature_set:
            feature_set[token] = 1
        else:
            feature_set[token] += 1
    return feature_set

print(features('xin chao cac ban',2))