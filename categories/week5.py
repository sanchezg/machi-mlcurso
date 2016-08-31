from kickstart import load_data

dataset = load_data("mlm_items_train.jsonlines")

from random import shuffle

shuffle(dataset)

import pandas as pd
columns = [u'seller_id', u'price', 'title']

df = pd.DataFrame(dataset)

y = df['top_level_category']
X = df[columns]

pd.get_dummies(X)
