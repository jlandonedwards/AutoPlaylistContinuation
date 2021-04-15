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


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



if __name__=='__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--train_dir', type=str, default='./train', help="directory where training data loader is stored")
    args.add_argument('--val_dir', type=str, default='./val', help="directory where training data loader is stored")
    args.add_argument('--challenge_data_dir', type=str, default='./data_preprocessed/challenge_data', help="directory where challenge challange data_loader is stored")
    args.add_argument('--models_dir', type=str, default='./trained_models', help="directory where to save model checkpoints")
    args.add_argument('--load_dir', type=str, default='./trained_models/junk/post_train', help="directory where to load completed trained model")
    args.add_argument('--resume_dir', type=str, default='./trained_models/junk/resume/most_recent', help="directory where to save model checkpoints to resumed to in training")
    args.add_argument('--model_name', type=str, default='junk', help="Unique Name to Save Model")
    args.add_argument('--mode', type=str, default='train', help="whether to train,resume training, or load model and start new training session")
    args = args.parse_args()
    
    if args.mode not in ["train","resume","load"]:
        print("Please specify --mode arg as one of 'train' ,'resume', or 'load'")
        sys.exit()
    resume = 0
    resume_path=""
    # All the hyperparamters that will be passed in by config object
    n_epochs = 2
    train_batch_size = 128
    val_batch_size = 100
    n_val_batches = 1000 // val_batch_size
    n_ids = 1380913
    n_track_ids = 1187154
    n_cids = 41
    dataset = DataLoader('./data_preprocessed/id_dicts')
    training_set = dataset.get_traing_set(args.train_dir,train_batch_size,123)
    n_train_batches = len(training_set)
    validation_sets = dataset.get_validation_sets(args.val_dir,val_batch_size)
    opt = keras.optimizers.Adam()
    model = Model(n_ids,n_track_ids,n_cids)
    model.optimizer = opt
    save_train_path =  args.models_dir +"/" + args.model_name +"/" +"resume/"
    
    
    if args.mode == 'load':
        checkpoint = tf.train.Checkpoint(model=model)
        load_manager = tf.train.CheckpointManager(checkpoint, args.load_dir, max_to_keep=3)
        checkpoint.restore(load_manager.latest_checkpoint)
        model.Metrics.epochs_train_metrics = np.load(args.load_dir + "/train_metrics.npy").tolist()
        model.Metrics.epochs_val_metrics = np.load(args.load_dir + "/val_metrics.npy").tolist()
    elif args.mode == 'resume':
        resume_path = args.resume_dir
        resume=1
    
         
    
    model.train(training_set,validation_sets,n_epochs,train_batch_size,val_batch_size,save_train_path,resume_path,resume)
    # Save the Model Upon Completeing Traing
    checkpoint = tf.train.Checkpoint(model=model,epochs_train_metrics=tf.Variable(model.Metrics.epochs_train_metrics),
                                              epoch_val_metrics=tf.Variable(model.Metrics.epochs_val_metrics))
    save_manager = tf.train.CheckpointManager(checkpoint, args.models_dir +"/" + args.model_name +"/" +"post_train" , max_to_keep=3)
    save_manager.save()
    np.save(args.models_dir +"/" + args.model_name +"/" +"post_train" + "/train_metrics",model.Metrics.epochs_train_metrics)
    np.save(args.models_dir +"/" + args.model_name +"/" +"post_train" + "/val_metrics",model.Metrics.epochs_val_metrics)
    
    
        
    
    
    
        
        
    