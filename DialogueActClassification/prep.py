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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Input
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Merge
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.regularizers import l2, activity_l2, l1, activity_l1, l1l2,  activity_l1l2
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, ParametricSoftplus, ThresholdedReLU, SReLU, LeakyReLU
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

np.random.seed(1337)

LSTM = GRU

# リスト分割
def divide_list(xs, n):
  q = len(xs) // n
  m = len(xs) % n
  return reduce(lambda acc, 
    i:(lambda fr = sum([ len(x) for x in acc ]):acc + [xs[fr:(fr + q + (1 if i < m else 0))]])(),
    range(n),[])

# 交差検証用データ作成
def cross_validation(session_ids, n):
  print("split session data ...")
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

def pretrain(maxlen, dim):
  print("Build pre-train model ...")
  convs = []
  graph_in = Input(shape=(maxlen, dim))
  ngram_filters = (3, 5, 7)
  for n_gram in ngram_filters:
    conv = Convolution1D(nb_filter=64,
                         filter_length=n_gram,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
  out = Merge(mode='concat')(convs)
  graph = Model(input=graph_in, output=out)
  return graph

def model_read(dense_dim, embedding_dim, sequence_maxlen, max_features_uni, max_features_bi, max_features_pos, bowlen, boblen, boplen, uni_weights):
  # unigram
  uni = Sequential()
  uni.add(Embedding(max_features_uni, embedding_dim, dropout=0.5, input_length=sequence_maxlen, weights=uni_weights))
  #graph = pretrain(sequence_maxlen, embedding_dim)
  #uni.add(graph)  
  #uni.add(LSTM(128))
  #"""
  uni.add(Convolution1D(nb_filter=128,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
  uni.add(GlobalMaxPooling1D())
  #"""
  # bigram
  bi = Sequential()
  bi.add(Embedding(max_features_bi, embedding_dim, dropout = 0.5, input_length=sequence_maxlen))
  #graph = pretrain(sequence_maxlen, embedding_dim)
  #bi.add(graph)
  #bi.add(LSTM(128))
  #"""
  bi.add(Convolution1D(nb_filter=128,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
  bi.add(GlobalMaxPooling1D())
  #"""
  pos = Sequential()
  pos.add(Embedding(max_features_pos, embedding_dim, dropout = 0.5))
  #graph = pretrain(sequence_maxlen, embedding_dim)
  #pos.add(graph) 
  pos.add(LSTM(128))
  """
  pos.add(Convolution1D(nb_filter=128,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
  pos.add(GlobalMaxPooling1D())
  """
  # bow
  bow = Sequential()
  bow.add(Dense(512, input_dim = bowlen, activation='relu'))
  bow.add(Dropout(0.5))
  # bob
  bob = Sequential()
  bob.add(Dense(512, input_dim = boblen, activation='relu'))
  bob.add(Dropout(0.5))
  # bop
  bop = Sequential()
  bop.add(Dense(512, input_dim = boplen, activation='relu'))
  bop.add(Dropout(0.5)) 
  # speaker type
  speaker_type = Sequential()
  speaker_type.add(Dense(3, input_dim = 1))
  speaker_type.add(Activation('relu'))
  # message_position
  message_position = Sequential()
  message_position.add(Dense(1, input_dim=1)) 
  # message_length
  message_length = Sequential()
  message_length.add(Dense(1, input_dim=1))
  message_length.add(Activation('relu'))   
  # merge
  #merged = Merge([bow, bob, speaker_type, message_position, message_length], mode='concat')
  merged = Merge([bow, bob, speaker_type, message_position, message_length], mode='concat')
  #merged = Merge([pob], mode='concat')    
  model = Sequential()
  model.add(merged)
  #model = bop
  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(dense_dim))
  model.add(Activation('softmax'))
  #rmsprop=keras.optimizers.RMSprop(lr=0.001*0.5, rho=0.9, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
  return model
  
def get_dummies(label):
  dummies = np.array(pd.get_dummies(label))  
  d = {np.argmax(x):label[i] for i, x in enumerate(dummies)}
  return dummies, d
  
def GloveEmbedding(embedding_dim):
  #res = PolyglotEmbedding.from_glove("/home/is/seiya-ka/embedding_vector/glove.twitter.27B."+str(embedding_dim)+"d.txt")
  res = PolyglotEmbedding.from_glove("/home/is/seiya-ka/embedding_vector/glove.6B."+str(embedding_dim)+"d.txt")
  return res

def nn_classifier(dense_dim, embedding_dim, sequence_maxlen, max_features_uni, max_features_bi, max_features_pos, bowlen, boblen, boplen, glove_embedding):
  if glove_embedding != "":
    embeddings = GloveEmbedding(embedding_dim)
    rank = glove_embedding
    vec, voc, weights = embeddings.vectors, embeddings.vocabulary, []
    for i, k in enumerate([0]+sorted(rank.keys())):
      if i == 0:
        weights.append(np.random.rand(embedding_dim)) 
        continue
      w = rank[k]
      if w in voc:
        weights.append(vec[voc[w]])
      else:
        weights.append(np.random.rand(embedding_dim)) 
    return model_read(dense_dim, embedding_dim, sequence_maxlen, max_features_uni+1, max_features_bi+1, max_features_pos+1, bowlen, boblen, boplen, [np.array(weights)])    
  else:
    return model_read(dense_dim, embedding_dim, sequence_maxlen, max_features_uni+1, max_features_bi+1, max_features_pos+1, bowlen, boblen, boplen, None)
  
def best_fit(model, verbose, train_feature, test_feature, tgt, feature_label, batch_size, nb_epoch):
  dense_dim = len(train_feature[tgt][0])
  train, test = [np.array(train_feature[fl]) for fl in  feature_label], [np.array(test_feature[fl]) for fl in  feature_label]
  train_y, test_y = train_feature[tgt], test_feature[tgt]
  es_cb = keras.callbacks.ModelCheckpoint(filepath='best.model', monitor='val_acc',  verbose=verbose, save_weights_only=True, save_best_only=True)
  model.fit(train, train_y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(test, test_y), callbacks=[es_cb])
  model.load_weights("best.model")
  return model
  
def pad_sequence(sequences, maxlen):
  sequences = [x[0] for x in sequences]
  return sequence.pad_sequences(np.array(sequences), maxlen=maxlen)
    
def predict(model, test_feature, tgt, feature_label):
  test =  [np.array(test_feature[fl]) for fl in  feature_label]
  test_y = test_feature[tgt]
  res = model.predict_classes(test)
  out_pred, out_true = [], []
  t, n = 0, 0
  for out_i, y_i in zip(res, test_y):
    out_pred.append(out_i)
    out_true.append(np.argmax(y_i))
    if out_i == np.argmax(y_i):  t += 1
    else: n += 1
  print(classification_report(out_true, out_pred, digits = 4))
