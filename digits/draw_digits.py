import json
from itertools import islice


def iter_data(path):
    return (json.loads(line) for line in open(path))


def draw_ascii_image(image):
    it = iter(image)
    for _ in range(28):
        line = []
        for b in islice(it, 28):
            if b < 80:
                c = "."
            elif b < 160:
                c = "*"
            else:
                c = "#"
            line.append(c * 2)
        print("".join(line))


if __name__ == "__main__":
    path = "digits_train.jsonlines"
    for x in iter_data(path):
        draw_ascii_image(x["image"])
        print()
