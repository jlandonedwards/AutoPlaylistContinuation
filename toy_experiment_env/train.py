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
from Model import Model
import argparse
import time
import sys

if __name__=='__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--train_dir', type=str, default='./toy_data', help="directory where training data loader is stored")
    args.add_argument('--val_dir', type=str, default='./toy_data', help="directory where training data loader is stored")
    args.add_argument('--challenge_data_dir', type=str, default='./toy_preprocessed/challenge_data', help="directory where challenge challange data_loader is stored")
    args.add_argument('--models_dir', type=str, default='./trained_models', help="directory where to save model checkpoints")
    args.add_argument('--load_dir', type=str, default='./trained_models/model_v1/post_train', help="directory where to save model checkpoints")
    args.add_argument('--model_name', type=str, default='combined_test', help="Unique Name to Save Model")
    args.add_argument('--mode', type=str, default='train', help="whether to train,resume training, or load model and start new training session")
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
    n_cids = 41
    dataset = DataLoader('./toy_preprocessed/id_dicts')
    training_set = dataset.get_traing_set('./toy_train',train_batch_size,123)
    n_train_batches = len(training_set)
    validation_sets = dataset.get_validation_sets('./toy_val',val_batch_size)
    opt = keras.optimizers.Adam()
    model = Model(n_ids,n_track_ids,n_cids)
    model.optimizer = opt
    resume_path = args.models_dir + "/" + args.model_name +"/" +"resume"
    
    
    if args.mode == 'load':
        checkpoint = tf.train.Checkpoint(model=model)
        load_manager = tf.train.CheckpointManager(checkpoint, args.load_dir, max_to_keep=3)
        checkpoint.restore(load_manager.latest_checkpoint)
    
    elif args.mode == 'resume':
        resume=1
    
    model.train(training_set,validation_sets,n_epochs,train_batch_size,val_batch_size,resume_path,resume)
    # Save the Model Upon Completeing Traing
    checkpoint = tf.train.Checkpoint(model=model)
    save_manager = tf.train.CheckpointManager(checkpoint, args.models_dir +"/" + args.model_name +"/" +"post_train" , max_to_keep=3)
    save_manager.save()
    
    
        
    
    
    
        
        
    