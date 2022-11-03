import pandas as pd
from datasets import DatasetDict
from datasets import load_dataset

from original import ESPD

dataset = load_dataset('csv', data_files={'train':'data/PANC-train.csv','test':'data/PANC-test.csv'})

# TODO Labels -> Numbers
print(dataset['train'].features)


eSPD = ESPD()
eSPD.fit(dataset)
