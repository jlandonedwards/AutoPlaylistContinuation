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
    
#loss_tracker = keras.metrics.Mean(name="loss")

class DAE(tf.keras.Model):
    def __init__(self,n_ids,n_track_ids):
        super(DAE, self).__init__()
        self.n_ids = n_ids
        self.n_track_ids = n_track_ids
        self.emb = Embedding()
        self.Metrics = Metrics(n_ids,n_track_ids)

    
    def call(self, ids,training=False):
        x = self.emb(ids)
        x = keras.activations.sigmoid(x)
        x = x @ K.transpose(self.emb.w)
        y_pred = keras.activations.sigmoid(x)
        return y_pred
    
    
    def loss(self,y_tracks,y_artists,y_pred):
        y_true = tf.cast(tf.concat([y_tracks,y_artists],1),tf.float32).to_tensor(default_value=0,shape=(y_tracks.shape[0],self.n_ids))
        return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred+1e-10) + (1-y_true)*tf.math.log(1 -(y_pred+1e-10)),axis=1),axis=0)
        
    def get_reccomendations(self,x_tracks,y_tracks,y_artists,y_pred,is_train=True):
        cand_ids = self._zero_by_ids(y_pred,x_tracks,is_train)
        cand_track = cand_ids[:,0:self.n_track_ids]
        cand_artist = cand_ids[:,self.n_track_ids:]
        _,rec_tracks = tf.math.top_k(cand_track,k=500)
        _,rec_artists = tf.math.top_k(cand_artist,k=500)
        rec_artists += self.n_track_ids
        return rec_tracks,rec_artists
    
    def _zero_by_ids(self,tensor,ids,is_train):
        if is_train: nrows = self.train_batch_size
        else: nrows = self.val_batch_size 
        ids_2d = tf.stack([tf.ones_like(ids)*tf.expand_dims(tf.range(nrows),1),ids],2)
        zeros = tf.zeros_like(ids,dtype=tf.float32)
        return tf.tensor_scatter_nd_update(tensor,ids_2d.flat_values,zeros.flat_values)
    
    
    
    @tf.function
    def train_step(self, data):
        x_tracks,x_artists,y_tracks,y_artists = data
        with tf.GradientTape() as tape:
            y_pred = self(tf.concat([x_tracks,x_artists],axis=1), training=True)  # Forward pass
            # Compute our own loss
            loss = self.loss(y_tracks,y_artists,y_pred)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        rec_tracks,rec_artists = self.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred)
        r_precision,ndcg,rec_clicks = self.Metrics.calculate_metrics(rec_tracks,rec_artists,y_tracks,y_artists)
        return loss,r_precision,ndcg,rec_clicks
    
    @tf.function
    def val_step(self,data):
        x_tracks,x_artists,y_tracks,y_artists,_ = data
        y_pred = self(tf.concat([x_tracks,x_artists],axis=1), training=False)
        loss = self.loss(y_tracks,y_artists,y_pred)
        rec_tracks,rec_artists = self.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred,is_train=False)
        r_precision,ndcg,rec_clicks = self.Metrics.calculate_metrics(rec_tracks,rec_artists,y_tracks,y_artists)
        return loss,r_precision,ndcg,rec_clicks
    
    def train(self,training_set,validation_sets,
              n_epochs,train_batch_size,val_batch_size,
              resume_path,resume=0,):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        n_batches = len(training_set)
        n_val_batches = 1000//val_batch_size
       
        training_set = iter(training_set.repeat(n_epochs))
        
        checkpoint = tf.train.Checkpoint(model=self,training_set=training_set,curr_epoch=tf.Variable(0))
        manager = tf.train.CheckpointManager(checkpoint, resume_path , max_to_keep=1)
        if resume: checkpoint.restore(manager.latest_checkpoint)
        curr_epoch = checkpoint.curr_epoch.numpy()
        
        for epoch in range(curr_epoch,n_epochs):
           print("EPOCH: ",epoch)
           for batch_step in range(n_batches):
               batch = next(training_set)
               loss,r_precision,ndcg,rec_clicks = self.train_step(batch)
               print("[Batch #{0}],loss:{1:g},R-precison:{2:g},NDCG:{3:.3f},Rec-Clicks:{4:g}".format(batch_step,loss,r_precision,ndcg,rec_clicks))
               self.Metrics.update_metrics("train_batch",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
               
           count = 0     
           for batch in validation_sets:
                loss,r_precision,ndcg,rec_clicks = self.val_step(batch)
                self.Metrics.update_metrics("val_batch",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
                count += 1
                if count == n_val_batches:
                    self.Metrics.update_metrics("val_set",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
                    count = 0
           metrics_train,metrics_val = self.Metrics.update_metrics("epoch")
           checkpoint.curr_epoch.assign_add(1)
           manager.save()
           loss,r_precision,ndcg,rec_clicks = metrics_train
           print("AVG Train: loss:{0:g},R-precison:{1:g},NDCG:{2:g},Rec-Clicks:{3:g}".format(loss,r_precision,ndcg,rec_clicks))
           loss,r_precision,ndcg,rec_clicks = metrics_val
           print("AVGe Val: loss:{0:g},R-precison:{1:g},NDCG:{2:g},Rec-Clicks:{3:g}\n".format(loss,r_precision,ndcg,rec_clicks))
        
    
    
   
        
        
    

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