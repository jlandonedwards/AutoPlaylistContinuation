#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:13:52 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import  json




def corrupt(x):
       
       
       p1 = np.random.uniform(size=1)
       if p1 > 0.5: 
           corrupt_track = True
       else: 
           corrupt_track = False
       
       input_tracks,input_artists,target_tracks,target_artists = tf.cond(tf.greater(x[2][0],25),
                                               lambda: gt_n(x,corrupt_track),
                                               lambda: le_n(x,corrupt_track))
       return ((input_tracks,input_artists),(target_tracks,target_artists))
   
@tf.autograph.experimental.do_not_convert
def gt_n(x,corrupt_track):
    p2 = np.random.uniform(size=1)
    if p2 > 0.5:
        keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
        n_inputs = tf.cast(tf.cast(x[2][0],tf.float32) * keep_rate,tf.int32)
        return random_id_corrupt(x,corrupt_track,n_inputs)
    
    else:
         return le_n(x,corrupt_track)


@tf.autograph.experimental.do_not_convert
def le_n(x,corrupt_track):
    keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
    n_inputs = tf.cast(tf.cast(x[2][0],tf.float32) * keep_rate,tf.int32)
    return seq_id_corrput(x,corrupt_track,n_inputs)
    

# @tf.autograph.experimental.do_not_convert
# def to_sparse(self,tensor):
#     idxs = tf.reshape(tensor,(-1,1))
#     dense = tf.scatter_nd(indices=idxs,updates=tf.ones_like(tensor,dtype="float32"),shape=[self.num_ids])
#     sparse = tf.sparse.from_dense(dense)
#     return sparse

@tf.autograph.experimental.do_not_convert
def delete_tensor_by_indices(tensor,indices,n_tracks):
    idxs = tf.reshape(indices,(-1,1))
    mask = ~tf.scatter_nd(indices=idxs,updates=tf.ones_like(indices,dtype=tf.bool),shape=[n_tracks])
    return tf.boolean_mask(tensor,mask)

@tf.autograph.experimental.do_not_convert
def random_id_corrupt(x,corrupt_track,n_inputs,):
    n_tracks = x[2][0]
    if corrupt_track:
        feat = x[0]
    else: 
        feat = tid_2_aid[x[0]]
    idxs = tf.random.shuffle(tf.range(n_tracks))[:n_inputs]
    input_ids = tf.gather(feat,idxs)
    removed_elements = delete_tensor_by_indices(feat,idxs,n_tracks)
    if corrupt_track :
        input_tracks = input_ids
        input_artists = tf.constant([],dtype=tf.int32)
        target_tracks = removed_elements
        target_artists = get_artists(target_tracks)
    else:
         input_tracks  = tf.constant([],dtype=tf.int32)
         input_mask = ~tf.equal(input_ids,-1)
         input_artists = tf.boolean_mask(input_ids,input_mask)
         target_tracks = x[0]
         target_mask = ~tf.equal(removed_elements,-1)
         target_artists = tf.boolean_mask(removed_elements,target_mask)
    return input_tracks,input_artists,target_tracks,target_artists

@tf.autograph.experimental.do_not_convert
def seq_id_corrput(x,corrupt_track,n_inputs):
     n_tracks = x[2][0]
     if corrupt_track:
        feat = x[0]
     else: 
        feat = tid_2_aid[x[0]]
    
     input_ids = feat[0:n_inputs]
     removed_elements  = feat[n_inputs:]
     if corrupt_track :
        input_tracks = input_ids
        input_artists = tf.constant([],dtype=tf.int32)
        target_tracks = removed_elements
        target_artists = get_artists(target_tracks)
     else:
        input_tracks  = tf.constant([],dtype=tf.int32)
        input_mask = ~tf.equal(input_ids,-1)
        input_artists = tf.boolean_mask(input_ids,input_mask)
        target_tracks = x[0]
        target_mask = ~tf.equal(removed_elements,-1)
        target_artists = tf.boolean_mask(removed_elements,target_mask)
     return input_tracks,input_artists,target_tracks,target_artists

@tf.autograph.experimental.do_not_convert
def get_artists(track_ids):
    artist_ids = tid_2_aid[track_ids]
    mask = ~tf.equal(artist_ids,-1)
    return tf.boolean_mask(artist_ids,mask)

training_set = tf.data.experimental.load('./toy_train',tf.RaggedTensorSpec(tf.TensorShape([3, None]), tf.int32, 1, tf.int64))

with open('./toy_preprocessed/id_dicts') as file:
                id_dicts = json.load(file)
                file.close()  
tid_2_aid = tf.constant(id_dicts['tid_2_aid'])
tid_2_aid = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tid_2_aid[:,0],tid_2_aid[:,1]),default_value=-1)

b20 = training_set.batch(1).skip(451)
b20 = next(iter(b20))
#track_ids = b20[46,0]
#artist_ids = tid_2_aid[track_ids]
(x_tracks,x_artists),(y_artists,y_tracks) = corrupt(b20)
print(y_tracks)
print(y_artists)


