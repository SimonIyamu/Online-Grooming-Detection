import pandas as pd
from datasets import load_dataset
import math

from original import ESPD

#dataset = load_dataset('csv', data_files={'train':'../data/PANC-train.csv','test':'../data/PANC-test.csv'})
print("loading csv files...")
X_train = pd.read_csv('../../data/PANC-train.csv', header=0)
X_test = pd.read_csv('../../data/PANC-test.csv', header=0)

eSPD = ESPD()
train_split = 0.8
idx_train = int(math.floor(len(X_train)*train_split))
preprocessed_data_train = eSPD.preprocess(X_train[:idx_train])
preprocessed_data_eval = eSPD.preprocess(X_train[idx_train:])

preprocessed_data = {"train": preprocessed_data_train, "eval": preprocessed_data_eval}

eSPD.fit(preprocessed_data)
