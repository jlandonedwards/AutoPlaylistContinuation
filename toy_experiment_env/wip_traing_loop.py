#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:13:10 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from TrainingDataLoader import *
from wip_DAE import *

EPOCH = 1
BATCH_SIZE = 50
dataset = DataLoader('./toy_preprocessed/id_dicts')
training_set = dataset.get_traing_set('./toy_train',BATCH_SIZE,123)
model = DAE()
opt = keras.optimizers.Adam()

                  
for epoch in range(EPOCH):
   for (x_tracks,x_artists),(y_tracks,y_artists) in training_set:
        
        with tf.GradientTape() as tape:
            y_pred = model(tf.concat([x_tracks,x_artists],axis=1), training=False)  # Forward pass
            # Compute our own loss
            loss = model.loss(y_tracks,y_artists,y_pred)
            # Compute gradients
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
    
        # Update weights
        opt.apply_gradients(zip(gradients, trainable_vars))
        
        rec_tracks,rec_artists = model.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred)
        r_precision,ndcg,rec_clicks = model.Metrics.collect_metrics(1,loss,rec_tracks,rec_artists,y_tracks,y_artists)
        print(" loss:{0:.2f}, R-precison:{1:.2f} , NDCG: {2:.2f} , Rec-Clicks: {3:.2f}".format(loss,r_precision,ndcg,rec_clicks))
    
   model.Metrics.collect_metrics(0)
        
        
    