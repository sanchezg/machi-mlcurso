import json


def euclidean_distance(v1, v2):
    """
   Ex:
       euclidean_distance([0, 0], [0, 1]) == 1
       euclidean_distance([0, 0], [1, 1]) == 1.41...
   """
    if len(v1) != len(v2):
        raise ValueError("Vectors have different dimension")
    accum = 0.0
    for x1, x2 in zip(v1, v2):
        accum += (x1 - x2) ** 2
    return accum ** 0.5


def build_prediction(train, test):
    # Train
    import random
    from collections import Counter
    random.shuffle(train)
    neighbors = []
    for_sale = Counter(x["seller_id"] for x in train)
    f = lambda datum: (datum["price"], for_sale[datum["seller_id"]])
    for datum in train[:10000]:
        v = f(datum)
        label = datum["top_level_category"]
        neighbors.append((v, label))

    # Evaluate
    result = []
    for datum in test:
        v = f(datum)
        nearest_neighbor = min(neighbors, key=lambda x: euclidean_distance(v, x[0]))
        _, label = nearest_neighbor
        result.append(label)
    return result


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    train = load_data("mlm_items_train.jsonlines")
    test = load_data("mlm_items_test.jsonlines")
    prediction = build_prediction(train, test)
    with open("prediction.jsonlines", "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
