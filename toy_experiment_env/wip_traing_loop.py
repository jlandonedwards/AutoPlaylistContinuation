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

EPOCH = 5
BATCH_SIZE = 50
dataset = DataLoader('./toy_preprocessed/id_dicts')
training_set = dataset.get_traing_set('./toy_train',BATCH_SIZE,123)
validation_sets = dataset.get_validation_sets('./toy_val')
model = DAE(BATCH_SIZE)
opt = keras.optimizers.Adam()



# Made train a function so we can wrap @tf.function on it when doing it for larg data set and will be much faster 

# @ tf.function  (Need to change tempory lists in Metrics.training = []  to tf.TensorArray and other things
#                 to make this work.  will look up how to do this tomorrow )
def train():
    model = DAE(50)
    count = 0                   
    for epoch in range(EPOCH):
       print("EPOCH: ",epoch)
       for x_tracks,x_artists,y_tracks,y_artists in training_set:
            with tf.GradientTape() as tape:
                y_pred = model(tf.concat([x_tracks,x_artists],axis=1), training=True)  # Forward pass
                # Compute our own loss
                loss = model.loss(y_tracks,y_artists,y_pred)
                # Compute gradients
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
            
        
            # Update weights
            opt.apply_gradients(zip(gradients, trainable_vars))
            
            rec_tracks,rec_artists = model.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred)
            r_precision,ndcg,rec_clicks = model.Metrics.collect_metrics("train",loss,rec_tracks,rec_artists,y_tracks,y_artists)
            #tf.print("[Batch #{0}],loss:{1:g},R-precison:{2:g},NDCG:{3:.3f},Rec-Clicks:{4:g}".format(count,loss,r_precision,ndcg,rec_clicks))
            count +=1
            
       for val_set in validation_sets:
            x_tracks,x_artists,y_tracks,y_artists,_ = val_set
            y_pred = model(tf.concat([x_tracks,x_artists],axis=1), training=False)
            loss = model.loss(y_tracks,y_artists,y_pred)
            
            rec_tracks,rec_artists = model.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred,is_train=False)
            model.Metrics.collect_metrics("val",loss,rec_tracks,rec_artists,y_tracks,y_artists)
           
            model.Metrics.collect_metrics("collect_val")
            
       metrics_train,metrics_val = model.Metrics.collect_metrics("epoch")
       loss,r_precision,ndcg,rec_clicks = metrics_train
       #tf.print("Average Train: loss:{0:g},R-precison:{1:g},NDCG:{2:g},Rec-Clicks:{3:g}".format(loss,r_precision,ndcg,rec_clicks))
       loss,r_precision,ndcg,rec_clicks = metrics_val
       #tf.print("Average Vaidation: loss:{0:g},R-precison:{1:g},NDCG:{2:g},Rec-Clicks:{3:g}\n".format(loss,r_precision,ndcg,rec_clicks))

train()
        
        
    