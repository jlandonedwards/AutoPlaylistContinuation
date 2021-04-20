#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:13:10 2021

@author: landon
"""


import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras as keras
from DataLoader import DataLoader
from Model import Model
import argparse
import time
import sys
import json
import os

tf.random.set_seed(2021)
np.random.seed(2021)



if __name__=='__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--models_dir', type=str, default='./trained_models', help="directory where to save model checkpoints")
    args.add_argument('--load_dir', type=str, default='./trained_models/final_sub/post_train', help="directory where to save model checkpoints")
    args.add_argument('--resume_dir', type=str, default='./trained_models/final_sub/resume/best_RP', help="directory where to save model checkpoints to resumed to in training")
    args.add_argument('--model_name', type=str, default='final_sub', help="Unique Name to Save Model")
    args.add_argument('--mode', type=str, default='resume', help="whether to train,resume training, or load model and start new training session")
    args = args.parse_args()
    
    if args.mode not in ["train","resume","load"]:
        print("Please specify --mode arg as one of 'train' ,'resume', or 'load'")
        sys.exit()
    
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)
   
    resume = 0
    resume_path=""
    # All the hyperparamters that will be passed in by config object
    n_epochs = 30
    train_batch_size = 128
    val_batch_size = 100
    n_val_batches = 1000 // val_batch_size
    with open("./utils/data_properties") as file:
        data_prop = json.load(file)
        file.close()
    n_ids = data_prop["n_tracks_artists"]
    n_track_ids = data_prop["n_tracks"]
    n_cids = data_prop["n_chars"]
    dataset = DataLoader('./utils')
    training_set = dataset.get_traing_set(train_batch_size,2020)
    n_train_batches = len(training_set)
    validation_sets = dataset.get_validation_sets(val_batch_size)
    opt = keras.optimizers.Adam(learning_rate=0.005)
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
    
    
        
    
    
    
        
        
    