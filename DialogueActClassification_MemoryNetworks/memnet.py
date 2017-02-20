# coding: utf-8

from __future__ import print_function
import os    
os.environ["THEANO_FLAGS"] = "device=gpu0" 
os.environ["floatX"] = "float32"
os.environ["base_compiledir"] = "dir0"

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM, GRU, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# memory network
def memnet(vocab_size, story_maxlen, query_maxlen, answer_dim, embedding_dim):
  input_encoder_m = Sequential()
  input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=embedding_dim,
                              input_length=story_maxlen))
  input_encoder_m.add(Dropout(0.3))
  
  question_encoder = Sequential()
  question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=query_maxlen))
  question_encoder.add(Dropout(0.3))
  
  match = Sequential()
  match.add(Merge([input_encoder_m, question_encoder],
                mode='dot',
                dot_axes=[2, 2]))
  
  input_encoder_c = Sequential()
  input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen,
                              input_length=story_maxlen))
  input_encoder_c.add(Dropout(0.3))
  
  response = Sequential()
  response.add(Merge([match, input_encoder_c], mode='sum'))
  response.add(Permute((2, 1)))
  
  answer = Sequential()
  answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
  answer.add(LSTM(256))
  answer.add(Dropout(0.3))
  answer.add(Dense(answer_dim))
  answer.add(Activation('softmax'))
  rmsprop=keras.optimizers.RMSprop(lr=0.001*0.5, rho=0.9, epsilon=1e-08, decay=0.0)
  answer.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
  return answer
  
# best model
def best_fit(model, batch_size, nb_epoch, train_feature, test_feature, tgt):
  inputs_train, queries_train, inputs_train, answers_train = [np.array(train_feature[k]) for k in ["stories","questions","stories", tgt]]
  inputs_test, queries_test, inputs_test, answers_test = [np.array(test_feature[k]) for k in ["stories","questions","stories", tgt]]
  print( inputs_train.shape, queries_train.shape, answers_train.shape)
  print( inputs_test.shape, queries_test.shape, answers_test.shape)
  es_cb = keras.callbacks.ModelCheckpoint(filepath='best.model', monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True)
  model.fit([inputs_train, queries_train, inputs_train], answers_train,
           batch_size=batch_size,
           nb_epoch=nb_epoch,
           validation_data=([inputs_test, queries_test, inputs_test], answers_test), callbacks=[es_cb])
  model.load_weights("best.model")  
  return model

def predict(model, test_feature, tgt):
  inputs_test, queries_test, inputs_test, answers_test = [np.array(test_feature[k]) for k in ["stories","questions","stories", tgt]]
  res = model.predict_classes([inputs_test, queries_test, inputs_test])
  out_pred, out_true, t, n = [], [], 0, 0
  for out_i, y_i in zip(res, answers_test):
    out_pred.append(out_i)
    out_true.append(np.argmax(y_i))
    if out_i == np.argmax(y_i):  t += 1
    else: n += 1
  print(classification_report(out_true, out_pred, digits = 4))

