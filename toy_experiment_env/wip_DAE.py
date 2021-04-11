#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:13:50 2021

@author: landon
"""
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import backend as K
from Metrics import Metrics


class Embedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        
        super(Embedding, self).__init__(**kwargs)
        
        
    def build(self,inputs):
       
        self.w = tf.Variable(
            tf.random.truncated_normal(
                shape=[81616, 32],
                stddev=1.0/math.sqrt(32)),
            name = "var_w",
            trainable=True) 

        # self.b = tf.Variable(
        #     tf.random.truncated_normal(
        #         shape=[32],
        #         stddev=1.0/math.sqrt(32)),
        #     name="var_b", 
        #     trainable=True)
        
    def call(self, inputs):
        x = tf.nn.embedding_lookup(self.w,inputs)
        x = keras.backend.sum(x,1)
        return x
    
   

def ragged_to_dense(tensor):
        idxs = tf.reshape(tensor,(-1,1))
        dense = tf.scatter_nd(indices=idxs,updates=tf.ones_like(tensor,dtype="float32"),shape=[81616])
        # sparse = tf.sparse.from_dense(dense)
        return dense      
        
    



#loss_tracker = keras.metrics.Mean(name="loss")

class DAE(tf.keras.Model):
    def __init__(self,batch_size):
        super(DAE, self).__init__()
        self.n_ids = 81616
        self.n_track_ids = 61740
        self.batch_size = batch_size
        self.val_batch_size = 1000
        self.emb = Embedding()
        self.Metrics = Metrics(self.n_ids,self.n_track_ids,self.batch_size)

    
    def call(self, ids,training=False):
        x = self.emb(ids)
        x = keras.activations.sigmoid(x)
        x = x @ K.transpose(self.emb.w)
        y_pred = keras.activations.sigmoid(x)
        return y_pred
    
    def loss(self,y_tracks,y_artists,y_pred):
        y_true = tf.cast(tf.concat([y_tracks,y_artists],1),tf.float32).to_tensor(default_value=0,shape=(y_tracks.shape[0],self.n_ids))
        return tf.reduce_mean(-K.sum(y_true*tf.math.log(y_pred+1e-10) + (1-y_true)*tf.math.log(1 -(y_pred+1e-10)),axis=1),axis=0)
        
    def get_reccomendations(self,x_tracks,y_tracks,y_artists,y_pred,is_train=True):
        cand_ids = self._zero_by_ids(y_pred,x_tracks,is_train)
        cand_track = cand_ids[:,0:self.n_track_ids]
        cand_artist = cand_ids[:,self.n_track_ids:]
        _,rec_tracks = tf.math.top_k(cand_track,k=500)
        _,rec_artists = tf.math.top_k(cand_artist,k=500)
        rec_artists += self.n_track_ids
        return rec_tracks,rec_artists
    
    def _zero_by_ids(self,tensor,ids,is_train):
        if is_train: nrows = self.batch_size
        else: nrows = self.val_batch_size 
        ids_2d = tf.stack([ids,tf.ones_like(ids)*tf.expand_dims(tf.range(nrows),1)],2)
        zeros = tf.zeros_like(ids,dtype=tf.float32)
        return tf.tensor_scatter_nd_update(tensor,ids_2d.flat_values,zeros.flat_values)
    
    
    
    @tf.function
    def train_step(self, data):
        
        x, y = data
        y,y_tracks = y
        y = keras.backend.map_fn(ragged_to_dense,y,dtype=tf.float32)
        
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            
            # Compute our own loss
            loss = self.loss(y,y_pred)
            #loss = K.mean(-K.sum(tf.sparse.to_dense(y)*K.log(y_pred+1e-10)+(1-tf.sparse.to_dense(y))*K.log(1 - (y_pred+1e-10)),axis=1))
        # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        #loss_tracker.update_state(loss)
        return {"loss": loss}

if __name__ ==  "__main__":
    from TrainingDataLoader import *
    data = TrainingDataLoader('./toy_train', 81616)
    
    
        
    
    model = DAE()
    model.compile(optimizer="adam",loss=loss_fn)
    #callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir="./tmp")
    #callback = tf.keras.callbacks.ModelCheckpoint(filepath='./tmp',monitor='loss',save_weights_only=True)
    model.fit(data.get_generator(100, 123),epochs=1)
    model.save_weights("weights",save_format="tf")
    del model
    
    reconstructed_model = DAE()
    reconstructed_model.compile(optimizer="adam",loss=loss_fn)
    reconstructed_model.load_weights("weights")
    print("Reconstruction Training")
    reconstructed_model.fit(data.get_generator(100, 123),epochs=1)


'''
To DO:
    Look at modification to this pretty simplistic network they use
    ideas: 
        -differnt activatoins
        -rather than sum embedding use some other combination of them
        - Implement Character CNN  or get an recent exisitng architecture and use it
'''


#reconstructed_model = DAE()
#reconstructed_model.compile(optimizer="adam",loss=loss_fn)
#reconstructed_model.load_weights("weights")