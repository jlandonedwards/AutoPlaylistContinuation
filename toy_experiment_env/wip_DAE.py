#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:13:50 2021

@author: landon
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class PlaylistEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PlaylistEmbedding, self).__init__(**kwargs)
        
        
    def build(self,inputs):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            name = "var_w",
            initial_value=w_init(shape=(81616,32), dtype="float32"),
            trainable=True,   
            
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name="var_b",
            initial_value=b_init(shape=(32), dtype="float32"), 
            trainable=True,
        )
        
    def call(self, inputs):
        return tf.linalg.matmul(tf.sparse.to_dense(inputs),self.w,a_is_sparse=True) + self.b
    



#loss_tracker = keras.metrics.Mean(name="loss")

class DAE(tf.keras.Model):
    def __init__(self,config=None):
        super(DAE, self).__init__()
        self.emb = PlaylistEmbedding()
    
    def call(self, inputs,training=False):
        x = self.emb(inputs)
        x = keras.activations.sigmoid(x)
        x = x @ K.transpose(self.emb.w)
        y_pred = keras.activations.sigmoid(x)
        return y_pred
    
    @tf.function
    def train_step(self, data):
        
        x, y = data
        y = tf.sparse.to_dense(y)
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.loss(y,y_pred)
            #loss = K.mean(-K.sum(tf.sparse.to_dense(y)*K.log(y_pred+1e-10)+(1-tf.sparse.to_dense(y))*K.log(1 - (y_pred+1e-10)),axis=1))
        # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        #loss_tracker.update_state(loss)
        return {"loss": loss}

if __name__ ==  "__main__":
    from TrainingDataLoader import *
    data = TrainingDataLoader('./toy_train', 81616)
    @tf.function
    def loss_fn(y_true, y_pred, sample_weight=None):
        return K.mean(
            -K.sum(
                y_true*K.log(y_pred+1e-10)
                + (1-y_true) * K.log(1 - (y_pred+1e-10)),
                   axis=1)
            )
        
    
    model = DAE()
    #callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir="./tmp")
    model.compile(optimizer="adam",loss=loss_fn)
    model.fit(data.get_generator(50, 123),epochs=1)
    model.save("./my_model.tf",save_format='tf')
    reconstructed_model = keras.models.load_model("./my_model.tf",custom_objects={"loss_fn":loss_fn})
    #reconstructed_model.compile(optimizer="adam",loss=loss_fn)
    print("Reconstruction Training")
    reconstructed_model.fit(data.get_generator(50, 123),epochs=1)


'''
To DO:
    Look at modification to this pretty simplistic network they use
    ideas: 
        -differnt activatoins
        -rather than sum embedding use some other combination of them
        - Implement Character CNN  or get an recent exisitng architecture and use it
'''

    