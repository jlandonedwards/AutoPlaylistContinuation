#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:09:40 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import json
import argparse
import os

HAS_TITLE = [1,1,0,1,1,1,1,1,1]
N_TRACKS = [0,5,5,10,25,25,100,100,1]
IS_RANDOM = [0,0,0,0,0,1,0,1,0]

TENSOR_SPEC = {'output_track_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'artist_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'title_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'n_tracks': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
      'n_output_artists': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
      'input_track_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)}



def delete_tensor_by_indices(tensor,indices,n_tracks):
    idxs = tf.reshape(indices,(-1,1))
    mask = ~tf.scatter_nd(indices=idxs,updates=tf.ones_like(indices,dtype=tf.bool),shape=[n_tracks])
    return tf.boolean_mask(tensor,mask)

@tf.autograph.experimental.do_not_convert
def map_func(x):
    return {
        'output_track_ids':x[0],
        'artist_ids':x[1],
        'title_ids':x[2],
        'n_tracks': x[3][0],
        'n_output_artists':x[4][0],
        }
@tf.autograph.experimental.do_not_convert
def no_title(x):
    x['title_ids'] = tf.constant((),dtype=tf.int32)
    return x

@tf.autograph.experimental.do_not_convert
def no_track(x):
    x['input_track_ids'] = tf.constant((),dtype=tf.int32)
    return x

@tf.autograph.experimental.do_not_convert
def random_tracks(x,n_inputs):
    n_tracks = x['n_tracks']
    idxs = tf.random.shuffle(tf.range(n_tracks))[:n_inputs]
    x['input_track_ids'] = tf.gather(x['output_track_ids'],idxs)
    x['output_track_ids'] = delete_tensor_by_indices(x['output_track_ids'],idxs,n_tracks)
    return x

@tf.autograph.experimental.do_not_convert
def seq_tracks(x,n_inputs):
     x['input_track_ids'] = x['output_track_ids'][0:n_inputs]
     x['output_track_ids'] = x['output_track_ids'][n_inputs:]
     return x
 
@tf.autograph.experimental.do_not_convert     
def le(x,n_input_tracks):
    return x['n_tracks'] <= n_input_tracks

@tf.autograph.experimental.do_not_convert 
def gt(x,n_input_tracks):
     return  x['n_tracks'] > n_input_tracks
     

def create_val_set(tf_dataset,n_playlists,n_input_tracks,title=1,rand=0):

    if n_input_tracks > 1:
        sample_excl = tf_dataset.filter(lambda x: le(x,n_input_tracks))
        sample_cand = tf_dataset.filter(lambda x: gt(x,n_input_tracks)).shuffle(n_playlists,seed=2021,reshuffle_each_iteration=False)
        selections = sample_cand.take(1000)
        unselected = sample_cand.skip(1000)
        dataset = sample_excl.concatenate(unselected)
    else:
        sample_cand = tf_dataset.shuffle(n_playlists,reshuffle_each_iteration=False)
        selections = sample_cand.take(1000)
        dataset = sample_cand.skip(1000)
    
    if not title:
        selections = selections.map(no_title)
    if n_input_tracks > 0:
        if rand:
            selections = selections.map(lambda x: random_tracks(x,n_input_tracks))
        else:
            selections = selections.map(lambda x: seq_tracks(x,n_input_tracks))
    else:
        selections = selections.map(no_track)

    return selections,dataset,n_playlists - 1000
    
  
        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--source_dir', type=str, default='./toy_preprocessed/data', help="directory where preprocessed data is stored")
    args.add_argument('--val_dir', type=str, default='./toy_val', help="directory where to witre validation sets to")
    args.add_argument('--train_dir', type=str, default='./toy_train', help="directory where to write training sets to")
    args.add_argument('--test_load', type=int, default=0, help='test loading each validation set and making sure they are equivalent to the saved')
    args = args.parse_args()
    
    if not os.path.isdir(args.val_dir):
      os.mkdir(args.val_dir)
    if not os.path.isdir(args.train_dir):
      os.mkdir(args.train_dir)
    
    
    tf.random.set_seed(2021)
    np.random.seed(2021)
    with open(args.source_dir) as file:
        data = json.load(file)
        file.close()
        
    playlists = data['playlists']    
    del data
    n_playlists = len(playlists)
    
    dataset = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(playlists))
    dataset = dataset.map(map_func)
    n_val_sets = 0
    for has_title,n_track,is_random in zip(HAS_TITLE,N_TRACKS,IS_RANDOM):
        val_set,train_set,n_playlists = create_val_set(dataset,n_playlists,n_track,has_title,is_random)
        n_val_sets+=1
        path = args.val_dir + "/" + str(n_val_sets)
        os.mkdir(path)
        tf.data.experimental.save(val_set, path)
        if args.test_load:
            val_load = tf.data.experimental.load(path,TENSOR_SPEC)
            for save,load in zip(val_set.take(5).as_numpy_iterator(),val_load.take(5).as_numpy_iterator()):
                assert np.all(save['output_track_ids'] == load['output_track_ids'])                 
    
    
    
    tf.data.experimental.save(train_set, args.train_dir)
    
    
    
    
    