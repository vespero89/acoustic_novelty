# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:17:53 2016

@author: Emanuele
"""

from keras.callbacks import Callback, EarlyStopping, ProgbarLogger, BaseLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input, MaxoutDense, merge
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianNoise
from keras.models import model_from_json, Model
from scipy.io import netcdf
from utils import print_and_flush
import numpy as np
from time import time
import sys

def EvaluateOnBatch(original_data, generated_data, labels):

    if (original_data.shape[1] == 1):
      error_seq = np.abs(generated_data - original_data.reshape(generated_data.shape)).mean(axis=1)
    else:
      error_seq = np.abs(generated_data - original_data).mean(axis=1)
    
    error_seq = np.delete(error_seq, -1, 0) 
    
    error_median = np.median(error_seq)
    seq_test_labels = np.array(labels, copy=True)
    seq_test_labels[labels == 1] = 3
    seq_test_labels = np.delete(seq_test_labels, -1, 0)
    
    betas = np.arange(1.1, 2.1, 0.1)
    tp = np.zeros(betas.shape)
    fn = np.zeros(betas.shape)
    fp = np.zeros(betas.shape)
    window = np.ones((20,))
    idx_beta = 0
    for beta in betas:
        th = beta * error_median
        decisions = np.zeros(error_seq.shape)
        decisions[error_seq > th] = 1
        
        decisions_conv = np.zeros(error_seq.shape)
        decisions_conv = np.convolve(decisions, window, mode='full')
        decisions_conv = np.delete(decisions_conv,np.s_[(decisions_conv.size-19):decisions_conv.size])
        
        decisions_2 = np.zeros(error_seq.shape)
        decisions_2[decisions_conv > 9.5] = 1  
        delta = seq_test_labels - decisions_2
        tp[idx_beta] = np.count_nonzero(delta == 2)
        fn[idx_beta] = np.count_nonzero(delta == 3)
        fp[idx_beta] = np.count_nonzero(delta == -1) 
        idx_beta = idx_beta + 1
      
    precision_tmp = tp / (tp + fp + 0.000001)
    recall_tmp = tp / (tp + fn + 0.000001)
    fmeasure_tmp = 2 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp + 0.000001)
    maxf1 = np.argmax(fmeasure_tmp)
    f1 = fmeasure_tmp[maxf1]
    TP_tot = tp[maxf1]
    FN_tot = fn[maxf1]
    FP_tot = fp[maxf1]
 
#    print (TP_tot, FN_tot, FP_tot)
    return TP_tot, FN_tot, FP_tot

def Evaluate(original_data, generated_data, seq_lengths, labels):
  error = np.abs(generated_data - original_data).mean(axis=1)
  betas = np.arange(1.1, 2.1, 0.1)
  startSeq = 0
  endSeq = -1
  window = np.ones((20,))
  tp_tot = 0
  fn_tot = 0
  fp_tot = 0
  for seqLen in seq_lengths:
    startSeq = endSeq + 1
    endSeq = startSeq + seqLen
    error_seq = error[startSeq:endSeq]
    error_median = np.median(error_seq)
    idx_beta = 0
    seq_test_labels = np.array(labels[startSeq:endSeq], copy=True)
    seq_test_labels[labels == 1] = 3
    tp = np.zeros(betas.shape)
    fn = np.zeros(betas.shape)
    fp = np.zeros(betas.shape)
    for beta in betas:
      th = beta * error_median
      decisions = np.zeros(error_seq.shape)
      decisions[error_seq >= th] = 1
      decisions[error_seq < th] = 0
      decisions_conv = np.convolve(decisions, window, mode='same')
      decisions[decisions_conv >= 10] = 1
      decisions[decisions_conv < 10] = 0
      delta = seq_test_labels - decisions
      tp[idx_beta] = np.count_nonzero(delta == 2)
      fn[idx_beta] = np.count_nonzero(delta == 3)
      fp[idx_beta] = np.count_nonzero(delta == -1)
      idx_beta = idx_beta + 1

    precisions = tp / (tp+fp)
    recalls = tp / (tp+fn)
    fmeasures = 2*precisions*recalls/(precisions+recalls)
    idx = fmeasures.argmax()
    tp_tot += tp[idx]
    fn_tot += tp[idx]
    fp_tot += tp[idx]

  precision = tp_tot / (tp_tot+fp_tot)
  recall = tp_tot / (tp_tot+fn_tot)
  fmeasure = 2*precision*recall/(precision+recall)

  return fmeasure


class CustomEarlyStopping(Callback):
  def __init__(self, validation_data, seq_lengths, labels, patience=0):
    super(Callback, self).__init__()

    self.best = -(np.Inf)
    self.validation_data = validation_data
    self.seq_lengths = seq_lengths
    self.labels = labels
    self.patience = patience
    self.wait = 0

  def on_epoch_end(self, epoch, logs={}):
    startSeq = 0;
    endSeq = 0
    tp = np.zeros(1)
    fn = np.zeros(1)
    fp = np.zeros(1)
    for seqLen in self.seq_lengths:
      startSeq = endSeq
      endSeq = startSeq + seqLen
      generated_data = self.model.predict_on_batch(self.validation_data[startSeq:endSeq])
      #print("generated_data: {0}".format(generated_data.shape))
      #print("validation_data: {0}".format(self.validation_data[startSeq:endSeq].shape))
      curTp, curFn, curFp = EvaluateOnBatch(self.validation_data[startSeq:endSeq], generated_data, self.labels[startSeq:endSeq])
      if (tp.shape[0] == 0):
        tp = np.array(curTp, copy=True)
        fn = np.array(curFn, copy=True)
        fp = np.array(curFp, copy=True)
      else:
        tp += curTp
        fn += curFn
        fp += curFp

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    fmeasure = 2*precision*recall/(precision+recall)
    score = float(fmeasure)
    #print (score)
    if (score > self.best):
      self.best = score
      self.wait = 0
      print_and_flush("\tNew best score: {0:.2%}".format(score))
      sys.stdout.flush()
    else:
      print_and_flush("Score: {0:.2%}".format(score))
      if (self.wait >= self.patience):
        self.model.stop_training = True
      self.wait += 1
      

def load_test_labels():
  y = np.genfromtxt('data/PASCAL_testset.label', delimiter=';',skip_footer=1,usecols=range(1,2999))
  y_end = np.genfromtxt('data/PASCAL_testset.label', delimiter=';',skip_header=140,usecols=range(1,1883))
  y = np.reshape(y, (y.size, 1))
  y_end = np.reshape(y_end, (y_end.size, 1))
  y = np.concatenate((y, y_end), axis=0)

  return y.flatten()

def load_data(path):
  with netcdf.netcdf_file(path, 'r') as f:
    inputs = f.variables['inputs'][:].copy()
    seqLengths = f.variables['seqLengths'][:].copy()

  return inputs, seqLengths

start_exp_time = time()
#
# Load train data
#
train_data, trainSeqLengths = load_data('data/PASCAL_trainset_102min_netcdf3.nc')

train_data_noise = train_data + np.random.normal(loc=0.0, scale=0.25, size=train_data.shape)
train_data_noise = train_data_noise.reshape((train_data_noise.shape[0], 1, train_data_noise.shape[1]))

#
# Load test data
#
test_data, testSeqLengths = load_data('data/PASCAL_testset_netcdf3.nc')
#test_data_no_reshape = np.ndarray(test_data, copy=True)
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
test_labels = load_test_labels();

n_samples = train_data.shape[0]
feat_dim = train_data.shape[1]


model = Sequential()
model.add(LSTM(216, input_shape = (1, train_data_noise.shape[2]), return_sequences = True, activation='tanh'))
#model.add(BatchNormalization())
model.add(LSTM(216, return_sequences = True, activation='tanh'))
#model.add(BatchNormalization())
model.add(LSTM(216, activation='tanh', return_sequences = False))
#model.add(BatchNormalization())
model.add(Dense(54, activation='linear'))

print("Compiling...")
sys.stdout.flush()
autoencoder.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

print("Fitting...")
sys.stdout.flush()
#autoencoder.fit(train_data_noise, train_data, nb_epoch = 100, callbacks = [CustomEarlyStopping(validation_data = test_data, seq_lengths = testSeqLengths, labels = test_labels, patience = 20)], verbose=1)
autoencoder.fit(train_data_noise, train_data, nb_epoch = 100, verbose=True,callbacks=[BaseLogger(), ProgbarLogger()])


end_exp_time = time()-start_exp_time


print("Overall experiment time: {0:.2f} s\n".format(end_exp_time))

