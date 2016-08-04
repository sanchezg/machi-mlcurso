import json
import sys

def convert_image(image, margin=160):
    result = [pixel >= margin for pixel in image]
    return result


def mean_number(train, looked_num):
    nums = [convert_image(sample['image']) for sample in train if sample['label'] == looked_num]
    nums_count = [num.count(True) for num in nums]
    nums_prom = sum(nums_count)/len(nums_count)
    return nums_prom


def calc_train_means(train):
    means = []
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for num in numbers:
        means.append(mean_number(train, num))
    return means


def build_prediction(train, test):
    means = calc_train_means(train)
    result = []
    for sample in test:
        ct_img = convert_image(sample['image']).count(True)
        def calc_distance(b):
            return abs(ct_img-b)
        distances = list(map(calc_distance, means))

        # Initial value is all differents
        mdist = 784
        candidate = -1
        for idx in xrange(len(distances)):
            if distances[idx] < mdist:
                mdist = distances[idx]
                candidate = idx
        result.append(candidate)
    return result


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    train = load_data("digits_train.jsonlines")
    test = load_data("digits_test.jsonlines")
    prediction = build_prediction(train, test)
    with open(sys.argv[1], "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
