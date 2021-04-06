#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:13:53 2021

@author: landon
"""

import tensorflow as tf
import numpy as np
 
class TrainingDataLoader():
    
    def __init__(self,data_dir,num_ids):
        self.TENSOR_SPEC =  tf.RaggedTensorSpec(tf.TensorShape([5, None]), tf.int32, 1, tf.int64)
        self.num_ids = num_ids
        self.training_set = tf.data.experimental.load(data_dir,self.TENSOR_SPEC)
    
    def get_generator(self,batch_size,seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        return self.training_set.shuffle(10000,seed,True).map(lambda x: self.corrupt(x)).batch(batch_size)
    

    @tf.autograph.experimental.do_not_convert
    def corrupt(self,x): 
        p1 = np.random.uniform(size=1)
        if p1 > 0.5: 
            feat,n_feat = 0,3
            target_feat,n_target_feat = 1,4 
        else: 
            feat,n_feat = 1,4
            target_feat,n_target_feat = 0,3
        
        n_feat = x[n_feat][0]
        n_target_feat = x[n_target_feat][0]
        
        input_ids,target_ids,n_inputs = tf.cond(tf.greater(n_feat,25),
                                                lambda: self.gt_n(x,feat,n_feat,target_feat),
                                                lambda: self.le_n(x,feat,n_feat,target_feat))
        return (self.to_sparse(input_ids),self.to_sparse(target_ids))
    

    @tf.autograph.experimental.do_not_convert
    def gt_n(self,x,feat,n_feat,target_feat):
        p2 = np.random.uniform(size=1)
        if p2 > 0.5:
            keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
            n_inputs = tf.cast(tf.cast(n_feat,tf.float32) * keep_rate,tf.int32)
            input_ids,target_ids = self.random_id_corrupt(x,feat,n_inputs,n_feat,target_feat)
            return input_ids, target_ids, n_inputs
        else:
             return self.le_n(x,feat,n_feat,target_feat)
    

    @tf.autograph.experimental.do_not_convert
    def le_n(self,x,feat,n_feat,target_feat):
        keep_rate = tf.random.uniform(minval=.2,maxval=.5,shape=())
        n_inputs = tf.cast(tf.cast(n_feat,tf.float32) * keep_rate,tf.int32)
        input_ids,target_ids = self.seq_id_corrput(x,feat,n_inputs,n_feat,target_feat)
        return input_ids,target_ids,n_inputs
    
    @tf.autograph.experimental.do_not_convert
    def to_sparse(self,tensor):
        idxs = tf.reshape(tensor,(-1,1))
        dense = tf.scatter_nd(indices=idxs,updates=tf.ones_like(tensor,dtype="float32"),shape=[self.num_ids])
        sparse = tf.sparse.from_dense(dense)
        return sparse
    
    @tf.autograph.experimental.do_not_convert
    def delete_tensor_by_indices(self,tensor,indices,n_tracks):
        idxs = tf.reshape(indices,(-1,1))
        mask = ~tf.scatter_nd(indices=idxs,updates=tf.ones_like(indices,dtype=tf.bool),shape=[n_tracks])
        return tf.boolean_mask(tensor,mask)

    @tf.autograph.experimental.do_not_convert
    def random_id_corrupt(self,x,feat,n_inputs,n_feat,target_feat):
        idxs = tf.random.shuffle(tf.range(n_feat))[:n_inputs]
        input_ids = tf.gather(x[feat],idxs)
        removed_elements = self.delete_tensor_by_indices(x[feat],idxs,n_feat)
        target_ids = tf.concat([removed_elements,x[target_feat]],axis=0)
        return input_ids,target_ids
    
    @tf.autograph.experimental.do_not_convert
    def seq_id_corrput(self,x,feat,n_inputs,n_feat,target_feat):
         input_ids = x[feat][0:n_inputs]
         removed_elements  = x[feat][n_inputs:]
         target_ids = tf.concat([removed_elements,x[target_feat]],axis=0) 
         return input_ids,target_ids
    
    

if __name__ == '__main__':
    dataset = TrainingDataLoader('./toy_train',81616)
    gen = dataset.get_generator(100,123)
    x,y = next(iter(gen))
    print("X shape:",x.shape)
    print("Y shape",y.shape)
        
    
    
    