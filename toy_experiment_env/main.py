#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:13:10 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from DataLoader import DataLoader
from wip_DAE import DAE

n_epochs = 5
train_batch_size = 50
val_batch_size = 50
n_val_batches = 1000 // val_batch_size
n_ids = 81616
n_track_ids = 61740
dataset = DataLoader('./toy_preprocessed/id_dicts')

training_set = dataset.get_traing_set('./toy_train',train_batch_size,123)
n_train_batches = len(training_set)
validation_sets = dataset.get_validation_sets('./toy_val',val_batch_size)
#model = DAE(n_ids,n_track_ids,BATCH_SIZE,)
opt = keras.optimizers.Adam()

model = DAE(n_ids,n_track_ids)
model.optimizer = opt
model.train(training_set,validation_sets,n_epochs,train_batch_size,val_batch_size)



        
        
    