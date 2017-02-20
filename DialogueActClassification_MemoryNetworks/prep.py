# coding:utf-8

import os    
os.environ["THEANO_FLAGS"] = "device=gpu0" 
os.environ["floatX"] = "float32"
os.environ["base_compiledir"] = "dir0"

import numpy as np
import pandas as pd
import random
import keras
from polyglot.mapping import Embedding as PolyglotEmbedding
from keras.preprocessing.sequence import pad_sequences

def divide_list(xs, n):
  q = len(xs) // n
  m = len(xs) % n
  return reduce(lambda acc, 
    i:(lambda fr = sum([ len(x) for x in acc ]):acc + [xs[fr:(fr + q + (1 if i < m else 0))]])(),
    range(n),[])

def cross_validation(session_ids, n):
  random.shuffle(session_ids)
  cross_divide_list = divide_list(session_ids, n)
  data = []
  for i, l in enumerate(cross_divide_list):
    data.append(
      [[id for i_, id in enumerate(cross_divide_list) if i_ != i],
      [id for i_, id in enumerate(cross_divide_list) if i_ == i]]
      )
    data[i][0] = [flatten for inner in data[i][0] for flatten in inner]
    data[i][1] = [flatten for inner in data[i][1] for flatten in inner]
  return data

def texts_to_sequences(texts, nb_words):
  texts = [x.encode("utf-8", errors ="ignore") for x in texts]
  tokenizer = Tokenizer(nb_words = nb_words)
  texts = np.array(texts)
  tokenizer.fit_on_texts(texts)
  return tokenizer.texts_to_sequences(texts)

def sequences_to_matrix(sequences, nb_words, mode):
  tokenizer = Tokenizer(nb_words = nb_words)
  return tokenizer.sequences_to_matrix(sequences, mode=mode)

def get_dummies(label):
  dummies = np.array(pd.get_dummies(label))  
  d = {np.argmax(x):label[i] for i, x in enumerate(dummies)}
  return dummies, d
  
def GloveEmbedding(embedding_dim):
  res = PolyglotEmbedding.from_glove("/home/is/seiya-ka/embedding_vector/glove.6B."+str(embedding_dim)+"d.txt")
  return res

def pad_sequence(sequences, maxlen):
  sequences = [x for x in sequences]
  return pad_sequences(np.array(sequences), maxlen=maxlen)
