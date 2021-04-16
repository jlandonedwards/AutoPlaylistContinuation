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
import time

HAS_TITLE = [1,1,0,1,1,1,1,1,1]
N_TRACKS = [0,5,5,10,25,25,100,100,1]
IS_RANDOM = [0,0,0,0,0,1,0,1,0]

TENSOR_SPEC = tf.RaggedTensorSpec(tf.TensorShape([4, None]), tf.int32, 1, tf.int64)



def delete_tensor_by_indices(tensor,indices,n_tracks):
    idxs = tf.reshape(indices,(-1,1))
    mask = ~tf.scatter_nd(indices=idxs,updates=tf.ones_like(indices,dtype=tf.bool),shape=[n_tracks])
    return tf.boolean_mask(tensor,mask)

@tf.autograph.experimental.do_not_convert
def map_func(x):
    return {
        'track_ids':x[0],
        'title_ids':x[1],
        'n_tracks':x[2][0],
        }
@tf.autograph.experimental.do_not_convert
def challenge_map_func(x):
    return (x[0],x[1],x[2],x[3])
         
@tf.autograph.experimental.do_not_convert
def get_artists(track_ids):
    artist_ids = tid_2_aid[track_ids]
    mask = ~tf.equal(artist_ids,-1)
    return tf.boolean_mask(artist_ids,mask)

@tf.autograph.experimental.do_not_convert
def no_title(x):
    x['title_ids'] = tf.constant([],dtype=tf.int32)
    return x

@tf.autograph.experimental.do_not_convert
def no_track(x):
    x['input_track_ids'] = tf.constant([],dtype=tf.int32)
    x['input_artist_ids'] = tf.constant([],dtype=tf.int32)
    x['target_track_ids'] = x['track_ids']
    x['target_artist_ids'] = get_artists(x['track_ids'])
    del x['track_ids']
    return x

@tf.autograph.experimental.do_not_convert
def random_id_corrupt(x,n_inputs):
    n_tracks = x['n_tracks']
    idxs = tf.random.shuffle(tf.range(n_tracks))[:n_inputs]
    x['input_track_ids'] = tf.gather(x['track_ids'],idxs)
    x['input_artist_ids'] = get_artists(x['input_track_ids'])
    x['target_track_ids'] = delete_tensor_by_indices(x['track_ids'],idxs,n_tracks)
    x['target_artist_ids'] =  get_artists(x['target_track_ids'])
    del x['track_ids']
    return x

@tf.autograph.experimental.do_not_convert
def seq_id_corrput(x,n_inputs):
     x['input_track_ids'] = x['track_ids'][0:n_inputs]
     x['input_artist_ids'] = get_artists(x['input_track_ids'])
     x['target_track_ids'] = x['track_ids'][n_inputs:]
     x['target_artist_ids'] =  get_artists(x['target_track_ids'])
     del x['track_ids']
     return x
 
@tf.autograph.experimental.do_not_convert     
def le(x,n_input_tracks):
    return x['n_tracks'] <= n_input_tracks

@tf.autograph.experimental.do_not_convert 
def gt(x,n_input_tracks):
     return  x['n_tracks'] > n_input_tracks

@tf.autograph.experimental.do_not_convert      
def val_to_list(x):
    return (x["input_track_ids"],x['input_artist_ids'],x["title_ids"],x["target_track_ids"],x['target_artist_ids'])

@tf.autograph.experimental.do_not_convert      
def train_to_ragged(x):
    tensors = [x["track_ids"],x["title_ids"],[x["n_tracks"]]]
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)
     

def create_val_set(tf_dataset,n_playlists,n_input_tracks,title=1,rand=0):

    if n_input_tracks > 0:
        sample_excl = tf_dataset.filter(lambda x: le(x,n_input_tracks))
        sample_cand = tf_dataset.filter(lambda x: gt(x,n_input_tracks)).shuffle(n_playlists,seed=2021,reshuffle_each_iteration=False)
        selections = sample_cand.take(1000)
        unselected = sample_cand.skip(1000)
        dataset = sample_excl.concatenate(unselected)
    else:
        sample_cand = tf_dataset.shuffle(1000,reshuffle_each_iteration=False)
        selections = sample_cand.take(1000)
        dataset = sample_cand.skip(1000)
    
    if not title:
        selections = selections.map(no_title)
    if n_input_tracks > 0:
        if rand:
            selections = selections.map(lambda x: random_id_corrupt(x,n_input_tracks))
        else:
            selections = selections.map(lambda x: seq_id_corrput(x,n_input_tracks))
    else:
        selections = selections.map(no_track)
    
    selections = selections.map(val_to_list)

    return selections,dataset,n_playlists - 1000


  
        
if __name__ == '__main__':
    start_time = time.time()
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--data_dir', type=str, default='./preprocessed_data', help="directory where preprocessed data is stored")
    args.add_argument('--utils_dir', type=str, default='./utils', help="directory where to write training sets to")
    args = args.parse_args()
    
    if not os.path.isdir("./val"):
        os.mkdir("./val")
    
        
    if not os.path.isdir("./train"):
        os.mkdir("./train")
    
    if not os.path.isdir("./challenge"):
        os.mkdir("./challenge")
    
    tf.random.set_seed(2021)
    np.random.seed(2021)
    with open(args.data_dir + "/" + "data") as file:
        data = json.load(file)
        file.close()  
    playlists = data['playlists']    
    del data
    with open(args.utils_dir + "/" + "tid_2_aid") as file:
        tid_2_aid = tf.constant(json.load(file))
        file.close()  
    tid_2_aid = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tid_2_aid[:,0],tid_2_aid[:,1]),default_value=-1)
    n_playlists = len(playlists)
    
    dataset = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(playlists))
    dataset = dataset.map(map_func)
    first = 1

    for has_title,n_track,is_random in zip(HAS_TITLE,N_TRACKS,IS_RANDOM):
        val_set,train_set,n_playlists = create_val_set(dataset,n_playlists,n_track,has_title,is_random)
        if  first:
            validation_sets = val_set
            first = 0
        else:
            validation_sets = validation_sets.concatenate(dataset=val_set)
        
    
    # Dataset where each element is it own validation set
    tf.data.experimental.save(validation_sets, "./val")
    
    # Full Uncorrupted Training Dataset
    train_set = train_set.map(train_to_ragged)
    tf.data.experimental.save(train_set, "./train")
    
    # Load Challenge Set
    with open(args.data_dir + "/challenge_data") as file:
        data = json.load(file)
        file.close()  
    challenge_sets = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(data['challenge_playlists']))
    challenge_sets = challenge_sets.map(challenge_map_func)
    del data
    tf.data.experimental.save(challenge_sets, "./challenge")
    
    print("---completed in %s minutes ---" % round((time.time() - start_time)/60,2))
    
'''
To Do:
    Figure out how to feed Validation sets 1 to 9 sequentially into model.fit(val=validation_data)
    So that it gets called and reports the meterics at the end of each epoch
'''    
    
    
    
    
    