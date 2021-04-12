#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:13:53 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import  json
 
class DataLoader():
    
    def __init__(self,dict_path):
        self.dict_path = dict_path
        self.tid_2_aid = None
        
        
        
        
    def get_traing_set(self,train_dir,batch_size,seed):
        training_set = tf.data.experimental.load(train_dir,tf.RaggedTensorSpec(tf.TensorShape([3, None]), tf.int32, 1, tf.int64))
        if self.tid_2_aid is None:
            with open(self.dict_path) as file:
                id_dicts = json.load(file)
                file.close()  
            tid_2_aid = tf.constant(id_dicts['tid_2_aid'])
            self.tid_2_aid = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tid_2_aid[:,0],tid_2_aid[:,1]),default_value=-1)
            del id_dicts,tid_2_aid
        tf.random.set_seed(seed)
        np.random.seed(seed)
        return training_set.map(lambda x: self.corrupt(x)).shuffle(1000,seed,True).apply(tf.data.experimental.dense_to_ragged_batch(batch_size,drop_remainder=True))
    
    def get_validation_sets(self,val_dir,batch_size):
        validation_sets = tf.data.experimental.load(val_dir,
                                                    (tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
                                                     tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
                                                     tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
                                                     tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
                                                     tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)))
        return validation_sets.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    
    def get_challenge_sets(self,challenge_dir):
        with open(challenge_dir) as file:
                data = json.load(file)
                file.close()
        challenge_data = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(data['challenge_playlists']))
        return challenge_data.batch(1000)
    
    
    

    @tf.autograph.experimental.do_not_convert
    def corrupt(self,x):
        
        
        p1 = np.random.uniform(size=1)
        if p1 > 0.5: 
            corrupt_track = True
        else: 
            corrupt_track = False
        
        input_tracks,input_artists,target_tracks,target_artists = tf.cond(tf.greater(x[2][0],25),
                                                lambda: self.gt_n(x,corrupt_track),
                                                lambda: self.le_n(x,corrupt_track))
        return (input_tracks,input_artists,x[1],target_tracks,target_artists)
    

    @tf.autograph.experimental.do_not_convert
    def gt_n(self,x,corrupt_track):
        p2 = np.random.uniform(size=1)
        if p2 > 0.5:
            keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
            n_inputs = tf.cast(tf.cast(x[2][0],tf.float32) * keep_rate,tf.int32)
            return self.random_id_corrupt(x,corrupt_track,n_inputs)
        
        else:
             return self.le_n(x,corrupt_track)
    

    @tf.autograph.experimental.do_not_convert
    def le_n(self,x,corrupt_track):
        keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
        n_inputs = tf.cast(tf.cast(x[2][0],tf.float32) * keep_rate,tf.int32)
        return self.seq_id_corrput(x,corrupt_track,n_inputs)
        
    
    # @tf.autograph.experimental.do_not_convert
    # def to_sparse(self,tensor):
    #     idxs = tf.reshape(tensor,(-1,1))
    #     dense = tf.scatter_nd(indices=idxs,updates=tf.ones_like(tensor,dtype="float32"),shape=[self.num_ids])
    #     sparse = tf.sparse.from_dense(dense)
    #     return sparse
    
    @tf.autograph.experimental.do_not_convert
    def delete_tensor_by_indices(self,tensor,indices,n_tracks):
        idxs = tf.reshape(indices,(-1,1))
        mask = ~tf.scatter_nd(indices=idxs,updates=tf.ones_like(indices,dtype=tf.bool),shape=[n_tracks])
        return tf.boolean_mask(tensor,mask)

    @tf.autograph.experimental.do_not_convert
    def random_id_corrupt(self,x,corrupt_track,n_inputs,):
        n_tracks = x[2][0]
        if corrupt_track:
            feat = x[0]
        else: 
            feat = self.tid_2_aid[x[0]]
        idxs = tf.random.shuffle(tf.range(n_tracks))[:n_inputs]
        input_ids = tf.gather(feat,idxs)
        removed_elements = self.delete_tensor_by_indices(feat,idxs,n_tracks)
        if corrupt_track :
            input_tracks = input_ids
            input_artists = tf.constant([],dtype=tf.int32)
            target_tracks = removed_elements
            target_artists = self.get_artists(target_tracks)
        else:
             input_tracks  = tf.constant([],dtype=tf.int32)
             input_mask = ~tf.equal(input_ids,-1)
             input_artists = tf.boolean_mask(input_ids,input_mask)
             target_tracks = x[0]
             target_mask = ~tf.equal(removed_elements,-1)
             target_artists = tf.boolean_mask(removed_elements,target_mask)
        return input_tracks,input_artists,target_tracks,target_artists
    
    @tf.autograph.experimental.do_not_convert
    def seq_id_corrput(self,x,corrupt_track,n_inputs):
         n_tracks = x[2][0]
         if corrupt_track:
            feat = x[0]
         else: 
            feat = self.tid_2_aid[x[0]]
        
         input_ids = feat[0:n_inputs]
         removed_elements  = feat[n_inputs:]
         if corrupt_track :
            input_tracks = input_ids
            input_artists = tf.constant([],dtype=tf.int32)
            target_tracks = removed_elements
            target_artists = self.get_artists(target_tracks)
         else:
            input_tracks  = tf.constant([],dtype=tf.int32)
            input_mask = ~tf.equal(input_ids,-1)
            input_artists = tf.boolean_mask(input_ids,input_mask)
            target_tracks = x[0]
            target_mask = ~tf.equal(removed_elements,-1)
            target_artists = tf.boolean_mask(removed_elements,target_mask)
         return input_tracks,input_artists,target_tracks,target_artists
    
    @tf.autograph.experimental.do_not_convert
    def get_artists(self,track_ids):
        artist_ids = self.tid_2_aid[track_ids]
        mask = ~tf.equal(artist_ids,-1)
        return tf.boolean_mask(artist_ids,mask)
    

if __name__ == '__main__':
    
    dataset = DataLoader('./toy_preprocessed/id_dicts')
    training_set = dataset.get_traing_set('./toy_train',50,123)
    x_tracks,x_artists,x_titles,y_tracks,y_artists = next(iter(training_set))
    validation_sets = dataset.get_validation_sets('./toy_val',50)
    challenge_sets = dataset.get_challenge_sets('./toy_preprocessed/challenge_data')
    
    
    
    
    
    