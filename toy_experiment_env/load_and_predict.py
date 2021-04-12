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
import argparse
import time
import sys

if __name__=='__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--challenge_data_dir', type=str, default='./toy_preprocessed/challenge_data', help="directory where challenge challange data_loader is stored")
    args.add_argument('--load_dir', type=str, default='./trained_models/model_v1/post_train', help="directory where to load a trained model")
    args.add_argument('--save_dir', type=str, default='./submissions/v1', help="path where to save challenge submissions")
   
    args = args.parse_args()
    
    if args.mode not in ["train","resume","load"]:
        print("Please specify --mode arg as one of 'train' ,'resume', or 'load'")
        sys.exit()
    resume = 0
    
    # All the hyperparamters that will be passed in by config object
    n_epochs = 2
    train_batch_size = 50
    val_batch_size = 50
    n_val_batches = 1000 // val_batch_size
    n_ids = 81616
    n_track_ids = 61740
    
    dataset = DataLoader('./toy_preprocessed/id_dicts')
    challenge_set = dataset.get_challenge_sets(args.challenge_data_dir,train_batch_size,123)
   
    opt = keras.optimizers.Adam()
    model = DAE(n_ids,n_track_ids)
    model.optimizer = opt
    
    checkpoint = tf.train.Checkpoint(model=model)
    load_manager = tf.train.CheckpointManager(checkpoint, args.load_dir, max_to_keep=3)
    checkpoint.restore(load_manager.latest_checkpoint)
    
    
    ## To Be Completed
    
    #model.generate_submissions(sav_dir,challenge_data)
    
    
    
    
        
    
    
    
        
        
    