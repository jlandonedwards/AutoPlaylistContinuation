#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:13:10 2021

@author: landon
"""

import logging

import tensorflow as tf
import tensorflow.keras as keras
from DataLoader import DataLoader
from Model import Model
import argparse
import json
import os




if __name__=='__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--load_dir', type=str, default='./trained_models/final_sub/resume/best_RP', help="directory where to load a trained model")
    args.add_argument('--save_name', type=str, default='sub.csv', help="name of submission file")
    args.add_argument('--team_name', type=str, default='JEYL', help="challange team name")
    args.add_argument('--email', type=str, default='j32edwar@uwaterloo.ca', help="team contact email")
    args = args.parse_args()
    
    if not os.path.isdir("./submissions"):
            os.mkdir("./submissions")
    save_path =  "./submissions/" + args.save_name
    
    with open("./utils/data_properties") as file:
        data_prop = json.load(file)
        file.close()
    n_ids = data_prop["n_tracks_artists"]
    n_track_ids = data_prop["n_tracks"]
    n_cids = data_prop["n_chars"]
    
    
    opt = keras.optimizers.Adam()
    model = Model(n_ids,n_track_ids,n_cids)
    model.optimizer = opt
    
    checkpoint = tf.train.Checkpoint(model=model,curr_epoch=tf.Variable(0), best_val_rp = tf.Variable(0.0))
    load_manager = tf.train.CheckpointManager(checkpoint, args.load_dir,max_to_keep=3)
    checkpoint.restore(load_manager.latest_checkpoint)
    
    dataset = DataLoader('./utils')
    challenge_batch_size = 50
    challenge_sets = dataset.get_challenge_sets(challenge_batch_size)
    
    model.generate_challenge_submissions(challenge_sets,save_path,args.team_name,args.email)
    

    
    
    
    
        
    
    
    
        
        
    