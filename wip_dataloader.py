#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 22:40:55 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import argparse

TENSOR_SPEC = {'output_track_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'artist_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'title_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
      'n_tracks': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
      'n_output_artists': tf.TensorSpec(shape=(), dtype=tf.int32, name=None)}



if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--train_dir', type=str, default='./toy_train', help="directory where to write training sets to")
    args = args.parse_args()
    
    dataset = tf.data.experimental.load(args.train_dir,TENSOR_SPEC)
    
    for x in dataset.take(4):
        print(x)
                      
    """ 
    To Do:
        - Decide On corruption Process
        - Decide On best format to feed  to model in batches
    """  
    
        