#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tuesday April 6th 2021

@author: ylalwani
"""

import tensorflow as tf
import numpy as np

# TODO for challenge, will have to modify to pick top 500 from recommended, and only intersect recommended of same length as ground

def get_rprecision(recommended, ground):
    if (str(type(recommended)) != "<class 'tensorflow.python.framework.sparse_tensor.SparseTensor'>"):
        recommended = tf.sparse.from_dense(recommended)
    if (str(type(ground)) != "<class 'tensorflow.python.framework.sparse_tensor.SparseTensor'>"):
        ground = tf.sparse.from_dense(ground)

    recommended_ind_set = set([tuple(x) for x in recommended.indices.numpy()])
    ground_ind_set = set([tuple(x) for x in ground.indices.numpy()])

    intersection = np.array([x for x in recommended_ind_set & ground_ind_set])

    ground_size = tf.size(ground)

    return(intersection.shape[0] / ground_size.numpy())