#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tuesday April 6th 2021

@author: landon,ylalwani
"""

import tensorflow as tf

class Metrics():
    def __init__(self,n_ids,n_track_ids,):
        self.n_ids = n_ids
        self.n_track_ids = n_track_ids    
        self.train_batch_metrics = []
        self.epochs_train_metrics = []
        self.val_batch_metrics = []
        self.val_sets_metrics = []
        self.epochs_val_metrics = []
        
        
    def update_metrics(self,mode='train',metrics=None):
        
        if mode == 'epoch':
            epoch_train = tf.reduce_mean(tf.stack(self.train_batch_metrics,0),0)
            self.epochs_train_metrics.append(epoch_train)
            self.train_batch_idx = 0
            val_sets_metrics = tf.stack(self.val_sets_metrics,0)
            self.epochs_val_metrics.append(val_sets_metrics)
            epoch_val = tf.reduce_mean(val_sets_metrics,0)
            self.train_batch_metrics = []
            self.val_sets_metrics = []
            return epoch_train,epoch_val
        
        elif mode == 'val_set':
                self.val_sets_metrics.append(tf.reduce_mean(tf.stack(self.val_batch_metrics,0),0))
                self.val_batch_metrics = []
                
        else:
            
            if mode == 'train_batch':
                self.train_batch_metrics.append(metrics)
                
            
            elif mode == 'val_batch':
                self.val_batch_metrics.append(metrics)
                
                
            
    def calculate_metrics(self,rec_tracks,rec_artists,y_tracks,y_artists):
        
        # R-precision
        correct_tracks = tf.sets.intersection(rec_tracks,y_tracks.to_sparse())
        correct_artists = tf.sets.intersection(rec_artists,y_artists.to_sparse())
        r_precision = self.r_precision(correct_tracks,y_tracks,correct_artists,y_artists)
        
        # Normalized Discounted Cumulative Gain
        idxs_3d = tf.where(tf.equal(tf.expand_dims(tf.sparse.to_dense(correct_tracks,default_value=-1),2),tf.expand_dims(rec_tracks,1)))
        ndcg = tf.cond(tf.greater(tf.size(idxs_3d), tf.constant(0,dtype=tf.int32)),lambda: self.NDCG(idxs_3d,y_tracks.shape[0]),lambda :  tf.constant(0,dtype=tf.float32))

        # Reccomended Song Clicks
        
        rec_clicks = tf.reduce_mean(tf.cast((tf.reduce_min(tf.RaggedTensor.from_value_rowids(idxs_3d[:,2],idxs_3d[:,0]).to_tensor(511),1)-1) //10,tf.float32),0)
        
        return r_precision,ndcg,tf.cast(rec_clicks,tf.float32)
    
    
    
    def r_precision(self,correct_tracks,y_tracks,correct_artists=None,y_artists=None):
        
        n_correct_tracks = tf.cast(tf.sets.size(correct_tracks),tf.float32)
        n_correct_artists = tf.cast(tf.sets.size(correct_artists),tf.float32)
        n_total_targets = tf.cast(tf.sets.size(y_tracks.to_sparse()),tf.float32)
        return tf.math.reduce_mean((n_correct_tracks + .25*n_correct_artists)/n_total_targets)
        
    
    def NDCG(self,idxs_3d,nrows):
        log2 = tf.math.log(2.0)
        obs_pos = tf.cast(tf.RaggedTensor.from_value_rowids(idxs_3d[:,2],idxs_3d[:,0],nrows=nrows),tf.float32)
        log2_pos =  tf.math.log(obs_pos + 2) / log2
        dcg_inv = tf.math.reduce_sum(log2_pos,1)
        best_pos = tf.cast(tf.RaggedTensor.from_value_rowids(idxs_3d[:,1],idxs_3d[:,0],nrows=nrows),tf.float32)
        log2_best_pos = tf.math.log(best_pos + 2) / log2
        idcg_inv = tf.math.reduce_sum(log2_best_pos,1)
        mask = tf.not_equal(dcg_inv,0)
        dcg_inv = tf.boolean_mask(dcg_inv,mask)
        idcg_inv =  tf.boolean_mask(idcg_inv,mask)
        return tf.math.reduce_mean(idcg_inv/dcg_inv)
    
    
    
   







