#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:13:50 2021

@author: landon
"""
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import backend as K
from Metrics import Metrics
import time
import json
import csv
import tensorflow_addons as tfa


# class MultiHeadAttention(tf.keras.layers.Layer):
#     def __init__(self,head_size,num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.ff1 = keras.layers.Dense(units=head_size*num_heads,activation=tfa.activations.mish)
#         self.ff2 = keras.layers.Dense(units=head_size*num_heads,activation=tfa.activations.mish)
#         self.mha = tfa.layers.MultiHeadAttention(head_size, num_heads)
        
#     def call(self,query,key):
#         key = self.ff1(key)
#         query = self.ff2(query)
#         return self.mha([query,key])


import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import backend as K
from Metrics import Metrics
import time
import json
import csv

class Embedding(tf.keras.layers.Layer):
    def __init__(self,n_ids,embedding_dim=32):
        super(Embedding, self).__init__()
        self.n_ids = n_ids
        self.embedding_dim = embedding_dim
            
    def build(self,inputs):
        self.w = tf.Variable(
            tf.random.truncated_normal(
                shape=[self.n_ids, self.embedding_dim],
                stddev=1.0/math.sqrt(self.embedding_dim)),
            name = "var_w",
            trainable=True) 
        
    def call(self, inputs):
        x = tf.nn.embedding_lookup(self.w,inputs)
        x = keras.backend.sum(x,1)
        return x
    
class CharacterCNN(tf.keras.layers.Layer):
    def __init__(self,n_cids,n_ids,embedding_dim=32):
        super(CharacterCNN, self).__init__()
        self.n_ids = n_ids
        self.emb = Embedding(n_cids,embedding_dim)
        self.ff1 = keras.layers.Dense(1014,activation='relu')
        self.conv1 = tf.keras.layers.Conv1D(256, 7, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=3)
        self.conv2 = tf.keras.layers.Conv1D(256, 7, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=3)
        self.conv3 = tf.keras.layers.Conv1D(256, 3, activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(256, 3, activation='relu')
        self.conv5 = tf.keras.layers.Conv1D(256, 3, activation='relu')
        self.conv6 = tf.keras.layers.Conv1D(256, 3, activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPooling1D(pool_size=3)
        self.flatten = tf.keras.layers.Flatten()
        self.ff2 = keras.layers.Dense(1024,activation='relu')
        self.dropout1 = keras.layers.Dropout(0.5)
        self.ff3 = keras.layers.Dense(1024,activation='relu')
        self.dropout2 = keras.layers.Dropout(0.5)
        self.ff4 = keras.layers.Dense(32,activation='relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        
    def call(self, ids,training=False):
       x = self.emb(ids) 
       x = self.ff1(x)
       x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
       x = self.conv1(x)
       x = self.maxpool1(x)
       x = self.conv2(x)
       x = self.maxpool2(x)
       x= self.bn1(x)
       x = self.conv3(x)
       x = self.conv4(x)
       x= self.bn2(x)
       x = self.conv5(x)
       x = self.conv6(x)
       x = self.maxpool3(x)
       x = self.flatten(x)
       x= self.bn3(x)
       x = self.ff2(x)
       x = self.dropout1(x)
       x = self.ff3(x)
       x = self.dropout2(x)
       x = self.ff4(x)
       x = self.bn4(x)
       return x
            
        
   
    

class DAE(tf.keras.layers.Layer):
    def __init__(self,n_ids,embedding_dim=256):
        super(DAE, self).__init__()
        self.b0 = tf.Variable(tf.random.normal(shape=[embedding_dim],dtype=tf.float32),trainable=True)
        self.b1 = tf.Variable(tf.random.normal(shape=[n_ids],dtype=tf.float32),trainable=True)
        self.emb = Embedding(n_ids,embedding_dim)
        self.ff1 = keras.layers.Dense(32,activation='relu')
        self.bn = keras.layers.BatchNormalization()
    
    def call(self, ids,training=False):
       x = self.emb(ids) + self.b0
       x = keras.activations.relu(x)
       x = x @ K.transpose(self.emb.w) + self.b1
       y_pred = self.ff1(x)
       y_pred = self.bn(y_pred)
       #y_pred = keras.activations.softmax(x,axis=1)
       #y_pred = keras.activations.sigmoid(x)
       return y_pred
        
    
#loss_tracker = keras.metrics.Mean(name="loss")

class Model(tf.keras.Model):
    def __init__(self,n_ids,n_track_ids,n_cids):
        super(Model, self).__init__()
        self.n_ids = n_ids
        self.n_track_ids = n_track_ids
        self.Metrics = Metrics(n_ids,n_track_ids)
        self.DAE = DAE(n_ids)
        self.charCNN = CharacterCNN(n_cids,n_ids)
        self.ff = keras.layers.Dense(n_ids,activation='relu')

    
    def call(self,ids,cids,training=False):
        y_pred_DAE = self.DAE(ids)
        y_pred_CNN = self.charCNN(cids)
        y_pred = self.ff(tf.concat([y_pred_DAE,y_pred_CNN],1))
        y_pred = keras.activations.softmax(y_pred,axis=1)
        return y_pred
    
    
    def loss(self,y_tracks,y_artists,y_pred):
        y_pred_tracks = y_pred[:,0:self.n_track_ids]
        y_pred_artists = y_pred[:,self.n_track_ids:]
        y_tracks = tf.cast(y_tracks,tf.float32).to_tensor(default_value=0,shape=(y_tracks.shape[0],self.n_track_ids))
        y_artists = tf.cast(y_artists,tf.float32).to_tensor(default_value=0,shape=(y_tracks.shape[0],self.n_ids-self.n_track_ids))
        l = self.cross_entropy(y_tracks,y_pred_tracks) + .5* self.cross_entropy(y_artists,y_pred_artists)
        reg = tf.linalg.norm(tf.concat([tf.reshape(w,[-1]) for w in self.trainable_weights],0))
        
        return l + 0.01* reg
    
    def cross_entropy(self,y_true,y_pred):
         return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred+1e-10) + .55*(1-y_true)*tf.math.log(1 -y_pred+1e-10),axis=1),axis=0)
        
    def get_reccomendations(self,x_tracks,y_pred):
        cand_ids = self._zero_by_ids(y_pred,x_tracks)
        cand_track = cand_ids[:,0:self.n_track_ids]
        cand_artist = cand_ids[:,self.n_track_ids:]

        _,rec_tracks = tf.math.top_k(cand_track,k=500)
        test = tf.sets.intersection(tf.cast(rec_tracks,tf.int32),x_tracks.to_sparse())
        missed = tf.sets.size(test)
        total_missed = tf.reduce_sum(missed,0)
        if total_missed > 0:
            print("debug")
        _,rec_artists = tf.math.top_k(cand_artist,k=500)
        rec_artists += self.n_track_ids
        return rec_tracks,rec_artists
    
    def _zero_by_ids(self,tensor,ids):
        ids_2d = tf.stack([tf.ones_like(ids)*tf.expand_dims(tf.range(tensor.shape[0]),1),ids],2)
        ones = tf.ones_like(ids,dtype=tf.float32) *-1
        return tf.tensor_scatter_nd_update(tensor,ids_2d.flat_values,ones.flat_values)
    
    @tf.function
    def train_step(self, data):
        x_tracks,x_artists,x_titles,y_tracks,y_artists = data
        with tf.GradientTape() as tape:
            y_pred = self(tf.concat([x_tracks,x_artists],axis=1),x_titles,training=True)  # Forward pass
            # Compute our own loss
            loss = self.loss(y_tracks,y_artists,y_pred)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        rec_tracks,rec_artists = self.get_reccomendations(x_tracks,y_pred)
        r_precision,ndcg,rec_clicks = self.Metrics.calculate_metrics(rec_tracks,rec_artists,y_tracks,y_artists)
        return loss,r_precision,ndcg,rec_clicks
    
    #@tf.function
    def val_step(self,data):
        x_tracks,x_artists,x_titles,y_tracks,y_artists = data
        y_pred = self(tf.concat([x_tracks,x_artists],axis=1),x_titles, training=False)
        loss = self.loss(y_tracks,y_artists,y_pred)
        rec_tracks,rec_artists = self.get_reccomendations(x_tracks,y_pred)
        r_precision,ndcg,rec_clicks = self.Metrics.calculate_metrics(rec_tracks,rec_artists,y_tracks,y_artists)
        return loss,r_precision,ndcg,rec_clicks
    
    def train(self,training_set,validation_sets,
              n_epochs,train_batch_size,val_batch_size,
              save_train_path,resume_path,resume=0):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        n_batches = len(training_set)
        n_val_batches = 1000//val_batch_size
        training_set = iter(training_set.repeat(n_epochs))
        
        self.checkpoint = tf.train.Checkpoint(model=self,training_set=training_set,curr_epoch=tf.Variable(0), best_val_rp = tf.Variable(0.0))
        
        if resume: 
            self.resume_manager = tf.train.CheckpointManager(self.checkpoint, resume_path , max_to_keep=1)
            self.checkpoint.restore(self.resume_manager.latest_checkpoint)
            self.Metrics.epochs_train_metrics = np.load(resume_path + "/train_metrics.npy").tolist()
            self.Metrics.epochs_val_metrics = np.load(resume_path + "/val_metrics.npy").tolist()
        
        self.most_recent_manager = tf.train.CheckpointManager(self.checkpoint, save_train_path + "/most_recent" , max_to_keep=15)
        self.best_RP_manager = tf.train.CheckpointManager(self.checkpoint, save_train_path + "/best_RP" , max_to_keep=15)
        curr_epoch = self.checkpoint.curr_epoch.numpy()
        best_val_rp = self.checkpoint.best_val_rp.numpy()
        
        
        pb_train_metrics_names = ['batch_loss','batch_R-Prec']
       
        progress_bar = tf.keras.utils.Progbar(self.train_batch_size*n_batches, stateful_metrics= pb_train_metrics_names, width=50,unit_name="batch")
        
        for epoch in range(curr_epoch,n_epochs):
           print("\nepoch {}/{}".format(epoch+1,n_epochs))
           start_time = time.time()
           for batch_step in range(n_batches):
               batch = next(training_set)
               loss,r_precision,ndcg,rec_clicks = self.train_step(batch)
               progress_bar.update((batch_step+1)*train_batch_size,list(zip(pb_train_metrics_names,[loss,np.round(r_precision,3)])))
               self.Metrics.update_metrics("train_batch",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
               
           count = 0
           set_count = 0
           for batch in validation_sets:
                loss,r_precision,ndcg,rec_clicks = self.val_step(batch)
                self.Metrics.update_metrics("val_batch",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
                count += 1
                if count == n_val_batches:
                    self.Metrics.update_metrics("val_set",tf.stack([loss,r_precision,ndcg,rec_clicks],0))
                    count = 0
                    set_count +=1
            
           metrics_train,metrics_val = self.Metrics.update_metrics("epoch")
          
   
           loss,r_precision,ndcg,rec_clicks = metrics_train
           print("\nAVG Train:\n   loss:{0:g}\n   R-precison:{1:g}\n   NDCG:{2:g}\n   Rec-Clicks:{3:g}".format(loss,r_precision,ndcg,rec_clicks))
           loss,r_precision,ndcg,rec_clicks  = metrics_val
           print("AVG Val:\n   loss:{0:g}\n   R-precison:{1:g}\n   NDCG:{2:g}\n   Rec-Clicks:{3:g}\n".format(loss,r_precision,ndcg,rec_clicks))
           
           
           
           self.checkpoint.curr_epoch.assign_add(1)
           if r_precision > best_val_rp:
               best_val_rp = r_precision
               self.checkpoint.best_val_rp.assign(best_val_rp)
               self.best_RP_manager.save()
               np.save(save_train_path + "/best_RP/train_metrics",self.Metrics.epochs_train_metrics)
               np.save(save_train_path + "/best_RP/val_metrics",self.Metrics.epochs_val_metrics)
           # Save most recent checkpoint
           self.most_recent_manager.save()
           np.save(save_train_path + "/most_recent/train_metrics",self.Metrics.epochs_train_metrics)
           np.save(save_train_path + "/most_recent/val_metrics",self.Metrics.epochs_val_metrics)
           
           print("-----Epoch {0:} completed in {1:.2f} minutes-----".format(epoch,(time.time() - start_time)/60))
          
    def generate_challenge_submissions(self,challenge_sets,save_path,team_name,email):
        with open('./utils/tid_2_uri') as file:
            tid_2_uri = tf.constant(list(json.load(file).items()))
            file.close()
        
        tid_2_uri = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            tf.strings.to_number(tid_2_uri[:,0],out_type=tf.int32),tid_2_uri[:,1]),"-")
        submissions= []
        for  x_tracks,x_artists,x_titles,(pid) in challenge_sets:
            y_pred = self(tf.concat([x_tracks,x_artists],axis=1),x_titles,training=False)
        
            rec_tracks,_ = self.get_reccomendations(x_tracks,y_pred)
            uris = "spotify:track:" + tid_2_uri[rec_tracks]
            uris = tf.concat([tf.strings.as_string(pid),uris],1)
            submissions += uris.numpy().astype(str).tolist()
    
        with open(save_path, 'w', newline = '') as outFile:
            wr = csv.writer(outFile, quoting = csv.QUOTE_NONE)
            wr.writerow(['team_info',team_name,email])
            wr.writerows(submissions)
            outFile.close()
        print("---- Submission saved to {0:} ------".format(save_path))
