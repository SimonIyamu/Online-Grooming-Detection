import pandas as pd
from sklearn.model_selection import train_test_split
import os

TRAIN_PATH = '../data'
TRAIN_FILE = 'PANC-train.csv'
TRAIN_FILE_SPLIT = 'PANC-train-split.csv'
EVAL_FILE_SPLIT = 'PANC-eval-split.csv'

def create_val_split():
    train_df = pd.read_csv(os.path.join(TRAIN_PATH, TRAIN_FILE))
    train, eval = train_test_split(train_df, test_size=0.1, random_state=42)
    train.to_csv(os.path.join(TRAIN_PATH, TRAIN_FILE_SPLIT))
    eval.to_csv(os.path.join(TRAIN_PATH, EVAL_FILE_SPLIT))

if __name__ == '__main__':
    create_val_split()
