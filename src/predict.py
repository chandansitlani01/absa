import tensorflow as tf
import pickle
import pandas as pd
import re
import numpy as np
from src.utils import *

tokenizer=load_tokenizer("models/tokenizer.pkl")
model=load_model("models/model.h5")


def pred(sentence, aspect):
  '''
  Predicts the label of a sentence and aspect.
  
  Args:
    Sentence: The sentence in string format
    aspect: The Aspect
  Returns:
    class: The class for sentence and aspect. 0:Negative, 1: Neutral, 2: Positive
  Raises:
    None
  '''
  df=pd.DataFrame()
  df["text"]=[sentence]
  df["aspect"]=[aspect]
  df["text"]=df["text"].map(lambda x:clean_row(x))
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

print(pred("hello thats great how are", "hello"))