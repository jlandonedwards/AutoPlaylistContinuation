#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 01:26:20 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from TrainingDataLoader import *
from wip_DAE import DAE

# Call back allows the training to be inturrupted and resume in exact same state, results in the same loss I checked
data = TrainingDataLoader('./toy_train', 81616)
model = DAE()
callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir="./tmp")
model.compile(optimizer="adam")
model.fit(data.get_generator(50, 123),epochs=5,callbacks=callback)

'''To DO:
    - Imlpment Model Save checkpoint based on metrics
    - Implment Load Model Frome Completed Traing (Differnt Call back than one above)
        - This might go in a differnt script as it will be used to predict on challenge 
'''
    


