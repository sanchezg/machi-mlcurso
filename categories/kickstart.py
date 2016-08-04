import json
import sys
import re


STOPWORDS = [
    'para', 'de', 'y', 'con', 'la', 'las', 'los', 'el', 'envio', 'gratis',
    'o', 'por', 'en', '-', '!', '\\', '/', 'a'
]


def get_default(train):
    dcount = {}
    for sample in train:
        if sample[u'top_level_category'] in dcount.keys():
            dcount[sample[u'top_level_category']] += 1
        else:
            dcount[sample[u'top_level_category']] = 1
    return dcount.keys()[dcount.values().index(max(dcount.values()))]


def remove_words(input_text, stopwords):
    resultwords  = [word.lower() for word in input_text.split()
                        if word.lower() not in stopwords and not word.isdigit()]
    result = re.sub(" \d+", "", ' '.join(resultwords))
    return result


def model_with_title(train):
    """First version: for each training sample, creates a relation between the words in
    title and top level category."""
    predictor = {}
    for sample in train:
        title_wout_sw = remove_words(sample[u'title'], STOPWORDS)
        if sample[u'top_level_category'] not in predictor.keys():
            predictor[sample[u'top_level_category']] = [title_wout_sw]
        else:
            predictor[sample[u'top_level_category']].append(title_wout_sw)
    return predictor


def predict_result_title(predictor, title, default):
    prediction = default
    title_wout_sw = remove_words(title, STOPWORDS)
    find = False
    for tlc in predictor.keys():
        for pt in predictor[tlc]:
            if all([word in pt for word in title_wout_sw.split()]):
                prediction = tlc
                find = True
                break
        if find:
            break
    return prediction


def build_prediction(train, test):
    """
    Edit here. Should return one prediction for each element in test.
    """
    default = get_default(train)
    predictor = model_with_title(train)
    return [predict_result_title(predictor, x[u'title'], default) for x in test]


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    train = load_data("mlm_items_train.jsonlines")
    test = load_data("mlm_items_test.jsonlines")
    prediction = build_prediction(train, test)
    with open(sys.argv[1], "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
