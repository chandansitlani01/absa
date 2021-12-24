import tensorflow as tf
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
import sys
import os

print("Loading Models")

main_path = Path(__file__).parent / "../"
print(main_path)
with open(main_path / "models/tokenizer.pkl", "rb") as f:
  tokenizer=pickle.load(f)

def clean_row(row):
  row=re.sub("[^a-zA-Z0-9]", " ", row)
  row=re.sub(" +", " ", row)
  row=row.lower()
  
  
  return row


def substitute_aspect(row):
  row["text"]=row["text"].replace(row["aspect"], "mask")
  return row

model=tf.keras.models.load_model(main_path / "models/model.h5")
def evaluate(df):
  '''
  Read Test Dataframe and predict the labels from inputs and save it to data/results/test.csv
  
  Args:
    df: The test dataframe
  Returns:
    classes: the array of classes
  Raises:
    KeyError: If column names are different than ["text", "aspect"]
  '''
  #embeddings=load_embeddings_to_dict("/content/glove.6B.100d.txt")
  df["text"]=df["text"].map(lambda x:clean_row(x))
  df=df.apply(lambda x:substitute_aspect(x), axis=1)
  df=df.apply(lambda x:substitute_aspect(x), axis=1)
  text=df["text"].values
  te=tokenizer.texts_to_sequences(text)
  sequences=tf.keras.preprocessing.sequence.pad_sequences(
  te, maxlen=30, dtype='int32', padding='post',
  truncating='post', value=0.0
  )

  pred=model.predict(sequences)

  classes=np.argmax(pred, axis=1)
  return classes
  

test=pd.read_csv(main_path / "data/test.tsv", delimiter="\t")
classes=evaluate(test.copy())

test["label"]=classes

test.to_csv(main_path / "data/results/test.csv", index=False)