import pandas as pd
import re
import pickle
import tensorflow as tf

def clean_row(row):
  '''
  Takes a sentence and cleans it using re.
  
  Args:
    row: a sentence as a string
  Returns:
    row: The cleaned sentence
  Raises:
    None
  '''
  row=re.sub("[^a-zA-Z0-9]", " ", row)
  row=re.sub(" +", " ", row)
  row=row.lower()
  return row


def substitute_aspect(row):
  '''
  Substitute the aspect in sentence with mask.
  
  Args:
    row: sentence as string
  Returns:
    row: sentence with aspect replaced as mask.
  Raises:
    None
  '''
  
  row["text"]=row["text"].replace(row["aspect"], "mask")
  return row
  

def load_model(model_path):
    '''
    Loads model from path.
    
    Args:
        model_path: the path o the model file.
    Returns:
        model: the model in tf.
    Raises:
        FileNotFound: If model does not exist at path
    '''
    model=tf.keras.models.load_model(model_path)
    
    return model

def load_tokenizer(tokenizer_path):
    '''
    Load tokenizer from pickle file.
    
    Args:
        tokenizer_path: path of tokenizer.
    Returns:
        tokenizer: the tokenizer in tf.
    Raises:
        FileNotFound: If tokenizer not found at path
    '''
    with open(tokenizer_path, "rb") as f:
        tokenizer=pickle.load(f)
    return tokenizer