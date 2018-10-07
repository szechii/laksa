'''
author:         szechi
date created:   2 Oct 2018
description:    cost functions
'''

import tensorflow as tf
import numpy as np
import scipy
import os
import sys

def pixel_wise_accuracy(ytrue, ypred, axis=-1):
    # axis: Specify axis containing one-hots
    L = tf.equal(tf.argmax(ypred,axis=axis), tf.argmax(ytrue,axis=axis))
    return tf.reduce_mean(tf.to_float(L) , name='accuracy')

def fg_accuracy(ytrue, ypred, axis=-1):
    # axis: Specify axis containing one-hots
    intersect_fore = tf.reduce_sum(tf.multiply(tf.to_float(tf.equal(tf.argmax(ypred,axis=axis),1)), tf.to_float(ytrue[...,1])))
    union_for = tf.reduce_sum(tf.to_float(tf.equal(tf.argmax(ypred,axis=axis),1))) + tf.reduce_sum(tf.to_float(ytrue[...,1]))
    A = ((2.0*intersect_fore)+1.0)/(union_for+1.0)
    return A

def eval_metrics(heatList, actualList, threshold):
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(len(heatList)):
        target_heat = heatList[i]
        target_act = actualList[i]
        pred = np.array(target_heat >= threshold, np.int32)
        union = 3.0*pred + 5.0*target_act
        label_im, nb_labels = scipy.ndimage.label(union)
        for n in range(nb_labels):
            if np.max(union*np.array(label_im == n+1, np.int32)) == 3.0:
                fp += 1.0
            if np.max(union[label_im == n+1]) == 8.0:
                tp += 1.0
        if np.sum(pred) > 30000: #arbitrary threshold for aneurysm project only
            fp += 1.0

    return tp, fp, fn

def multiclassDiceScore(ytrue, ypred, hard=True):
    nclass = ytrue.shape[-1].value
    assert nclass == ypred.shape[-1].value, \
        "Error: ypred nclass /= ytrue nclass"

    # Avoids 1/0 if ytrue and ypred == 0
    eta = 1e-5
    # Must be binary 0/1
    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    # Softmax
    tf_ypred_softmax = tf.nn.softmax(tf.reshape(ypred, (-1, nclass)),dim=-1)
    if hard:
        flat_index = tf.argmax(tf_ypred_softmax,axis=-1)
        flat_ypred = tf.one_hot(flat_index,depth=nclass)
    else:
        flat_ypred = tf.pow(tf_ypred_softmax,gamma)

    tf_prod = tf.matmul(tf.transpose(flat_ytrue), flat_ypred) # (nclass,nclass)
    flat_ytrue_area = tf.reduce_sum(flat_ytrue, axis=0, keep_dims=True)
    flat_ypred_area = tf.reduce_sum(flat_ypred, axis=0, keep_dims=True)
    tf_sum = tf.transpose(flat_ytrue_area) + flat_ypred_area
    tf_sum = tf.clip_by_value(tf_sum, eta, tf.reduce_max(tf_sum))

    return 2*tf_prod/tf_sum

def weightedFocalCrossEntropyLoss(ytrue, ypred, \
    weighting=False, weights=[1.0], focal=False, gamma=2):
    nclass = ytrue.shape[-1].value
    weights = np.reshape(np.asarray(weights),(-1,))
    if weighting:
        assert weights.size == nclass, "len(weights) != nclass"
    assert nclass == ypred.shape[-1].value, \
        "Error: ypred nclass /= ytrue nclass"

    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    flat_ypred = tf.reshape(ypred, (-1, nclass))

    if weighting:
        # Weight vector dim = (N) after x and reduce_sum with one-hot labels
        weights_v = tf.convert_to_tensor(weights,dtype=tf.float32)
        tf_weights = tf.reduce_sum(weights_v*tf.cast(flat_ytrue,tf.float32),axis=1)
    else:
        tf_weights = 1.0

    if focal:
        # 1-softmax vector
        tf_csoftmax = tf.nn.softmax(flat_ypred,dim=-1)
        tf_dsoftmax = tf.reduce_sum((1.0-tf_csoftmax)*tf.cast(flat_ytrue, tf.float32),axis=-1)
        # Focal loss vector
        tf_focal = tf.pow(tf_dsoftmax,gamma)
    else:
        tf_focal = 1.0

    # dim = (N) where N = # pixels (flattened from 2D array)
    L = tf.nn.softmax_cross_entropy_with_logits( \
        logits=flat_ypred, labels=flat_ytrue)

    mean_L = tf.reduce_mean(L*tf_weights*tf_focal)

    return mean_L

def crossEntropyLoss(ytrue, ypred):
    nclass = ytrue.shape[-1].value
    assert nclass == ypred.shape[-1].value, \
        "Error: ypred nclass /= ytrue nclass"

    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    flat_ypred = tf.reshape(ypred, (-1, nclass))

    L = tf.nn.softmax_cross_entropy_with_logits( \
        logits=flat_ypred,labels=flat_ytrue)

    mean_L = tf.reduce_mean(L)

    return mean_L

def weightedDiceLoss(ytrue, ypred, invarea=False, \
    weighting=False, weights=[1.0], hard=True, gamma=1.0):
    # weighting - whether to use custom weights for each class
    # hard      - if false, ypred = softmax prob, else ypred = one-hot from
    #             ypred's softmax
    # Note: Loss is weighted over classes as 1/(area^2) to compensate for
    #       naturally large scores for big regions. Loss is also summed over
    #       all classes before taking quotient. Therefore, normally loss will
    #       almost always be < 1.0. To get normal dice loss for binary classes,
    #       set weighting=True, weights=[0,1.0] (assuming class #1 is of
    #       interest), hard=True (so one-hot overlap is computed)
    nclass = ytrue.shape[-1].value
    weights = np.reshape(np.asarray(weights),(-1,))
    assert nclass == ypred.shape[-1].value, \
        "Error: ypred nclass /= ytrue nclass"
    if weighting:
        assert weights.size == nclass, "len(weights) != nclass"
    # Avoids 1/0 if ytrue and ypred == 0
    eta = 1e-5
    # Must be binary 0/1
    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    # Softmax
    tf_ypred_softmax = tf.nn.softmax(tf.reshape(ypred, (-1, nclass)),dim=-1)
    if hard:
        flat_index = tf.argmax(tf_ypred_softmax,axis=-1)
        flat_ypred = tf.one_hot(flat_index,depth=nclass)
    else:
        flat_ypred = tf.pow(tf_ypred_softmax,gamma)
    #flat_ypred = tf.reshape(ypred, (-1, nclass))
    tf_prod = tf.reduce_sum(tf.cast(flat_ytrue,tf.float32)*flat_ypred,axis=0)
    tf_sum = tf.reduce_sum(tf.cast(flat_ytrue, tf.float32) + flat_ypred,axis=0)

    # Weight to compensate for large dice score for large areas
    if invarea:
        tf_area = tf.reduce_sum(flat_ytrue,axis=0)
        # If area is too small, set its inverse weight to 0
        tf_mask = tf.greater_equal(tf_area,eta)
        tf_area = tf.maximum(tf_area,eta)
        tf_areaw = tf.cast(tf_mask,dtype=tf.float32)/(tf_area*tf_area)
    else:
        tf_areaw = tf.constant(1.0,dtype=tf.float32)

    if weighting:
        # Weight to compensate for class imbalances
        tf_weights = tf.convert_to_tensor(weights,dtype=tf.float32)
    else:
        tf_weights = tf.constant(1.0,dtype=tf.float32)

    # Multiclass Dice loss
    tf_w = tf_areaw*tf_weights
    tf_dice = tf.reduce_sum(tf_w*tf_prod)/ \
        tf.maximum(tf.reduce_sum(tf_w*tf_sum),eta*eta)

    return 1 - 2*tf_dice

def weightedCrossDiceLoss(ytrue, ypred, invarea=False, \
    weighting=False, weights=[1.0], hard=True, gamma=1.0):
    # weighting - whether to use custom weights for each class
    # hard      - if false, ypred = softmax prob, else ypred = one-hot from
    #             ypred's softmax
    # Note: Loss is weighted over classes as 1/(area^2) to compensate for
    #       naturally large scores for big regions. Loss is also summed over
    #       all classes before taking quotient. Therefore, normally loss will
    #       almost always be < 1.0. To get normal dice loss for binary classes,
    #       set weighting=True, weights=[0,1.0] (assuming class #1 is of
    #       interest), hard=True (so one-hot overlap is computed)
    nclass = ytrue.shape[-1].value
    weights = np.reshape(np.asarray(weights),(-1,))
    assert nclass == ypred.shape[-1].value, \
        "Error: ypred nclass /= ytrue nclass"
    if weighting:
        assert weights.size == nclass, "len(weights) != nclass"
    # Avoids 1/0 if ytrue and ypred == 0
    eta = 1e-5
    # Must be binary 0/1
    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    # Softmax
    tf_ypred_softmax = tf.nn.softmax(tf.reshape(ypred, (-1, nclass)),dim=-1)
    if hard:
        flat_index = tf.argmax(tf_ypred_softmax,axis=-1)
        flat_ypred = tf.one_hot(flat_index,depth=nclass)
    else:
        flat_ypred = tf.pow(tf_ypred_softmax,gamma)

    # Mat prod to get cross-terms
    # row = ytrue index, col = ypred index
    tf_prod = tf.matmul(tf.transpose(flat_ytrue),flat_ypred) # (nclass,nclass)
    '''
    tf_sum_ytrue = tf.expand_dims(tf.reduce_sum(flat_ytrue,axis=0),1) # (nclass,1)
    tf_sum_ytrue = tf.tile(tf_sum_ytrue,[1,nclass]) # (nclass,nclass)
    tf_sum_ypred = tf.expand_dims(tf.reduce_sum(flat_ypred,axis=0),0) # (1,nclass)
    tf_sum_ypred = tf.tile(tf_sum_ypred,[nclass,1]) # (nclass,nclass)
    tf_sum = tf_sum_ytrue + tf_sum_ypred
    '''
    # No need to compute cross-sum score for denom
    tf_sum = tf.reduce_sum(flat_ytrue + flat_ypred,axis=0) # (nclass,1)

    # Weight to compensate for large dice score for large areas
    if invarea:
        '''
        tf_area = tf_sum_ytrue[:,0]
        tf_mask = tf.greater_equal(tf_area,eta)
        tf_area = tf.maximum(tf_area,eta)
        tf_areaw = tf.cast(tf_mask,dtype=tf.float32)/(tf_area*tf_area)
        '''
        tf_area = tf.reduce_sum(flat_ytrue,axis=0)
        # If area is too small, set its inverse weight to 0
        tf_mask = tf.greater_equal(tf_area,eta)
        tf_area = tf.maximum(tf_area,eta)
        tf_areaw = tf.cast(tf_mask,dtype=tf.float32)/(tf_area*tf_area)
    else:
        tf_areaw = tf.constant(1.0,dtype=tf.float32)

    if weighting:
        # Weight to compensate for class imbalances
        tf_weights = tf.convert_to_tensor(weights,dtype=tf.float32)
    else:
        tf_weights = tf.constant(1.0,dtype=tf.float32)

    # Negative matrix
    tf_neg = 2*tf.eye(nclass,dtype=tf.int32) - tf.ones(nclass,dtype=tf.int32)
    tf_neg = tf.cast(tf_neg,tf.float32)
    # Multiclass Dice loss
    tf_w = tf.expand_dims(tf_areaw*tf_weights,1)*tf_neg
    tf_dice = tf.reduce_sum(tf_w*tf_prod)/ \
        tf.reduce_sum(tf.maximum(tf_w*tf_sum,eta*eta))

    return 1 - 2*tf_dice

def weightedWassersteinLoss(ytrue, ypred, M=None, exclude_background=False, \
    weighting=False, weights=[1.0], hard=True, gamma=1):

    eta = 1e-10

    nclass = ytrue.shape[-1].value
    assert nclass == ypred.shape[-1].value, "Error: nclass unequal"
    if M is None:
        M = tf.ones((nclass, nclass), dtype=tf.int32) \
            - tf.eye(nclass,  dtype=tf.int32)
        M = tf.cast(M, dtype=tf.float32)

    # Flatten data
    flat_ytrue = tf.reshape(ytrue, (-1, nclass))
    flat_ypred = tf.reshape(ypred, (-1, nclass))
    # Softmax
    tf_ypred_softmax = tf.nn.softmax(tf.reshape(ypred, (-1, nclass)), dim=-1)
    if hard:
        flat_index = tf.argmax(tf_ypred_softmax,axis=-1)
        flat_ypred = tf.one_hot(flat_index,depth=nclass)
    else:
        flat_ypred = tf.pow(tf_ypred_softmax, gamma)

    # WM
    wm_vec = tf.matmul(M, tf.transpose(flat_ytrue)) # dim (npx, nclass)
    wm_vec = tf.reduce_sum(flat_ypred*tf.transpose(wm_vec), \
        axis=-1, keep_dims=True) # dim (npx, 1)

    if weighting:
        # Weight to compensate for class imbalances
        assert len(weights) == nclass, "Error: len(weights) != nclass"
        tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        tf_weights = tf.expand_dims(tf_weights, axis=0) # dim (1, nclass)
    else:
        tf_weights = tf.constant(1.0, dtype=tf.float32)

    if exclude_background:
        wm_lb = M[0:1,:] # dim (1, nclass)
    else:
        wm_lb = tf.constant(1.0, dtype=tf.float32)

    tf_w = wm_lb*tf_weights
    tf_tp = flat_ytrue*(wm_lb - wm_vec) # dim (npx, nclass)
    tf_tp = 2*tf.reduce_sum(tf_w*tf_tp) # scalar

    denom = tf_tp + tf.reduce_sum(wm_vec)
    denom = tf.sign(denom)*tf.maximum(tf.abs(denom), eta)
    wm_wasserstein = tf_tp/denom

    return 1 - wm_wasserstein
