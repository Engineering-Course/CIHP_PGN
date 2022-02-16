from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
import argparse
from glob import glob
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from PIL import Image
from utils.image_reade_inf import *
from utils.ops import  *
from utils.utils import *
from utils.model_pgn import *

argp = argparse.ArgumentParser(description="Inference pipeline")
argp.add_argument('-i',
                  '--directory',
                  type=str, help='Path of the input dir',
                  default='./datasets/images')
argp.add_argument('-o',
                  '--output',
                  type=str, help='Path of the input dir',
                  default='./datasets/output')

args = argp.parse_args()

image_list_inp = []
for i in glob(os.path.join(args.directory, '**'), recursive=True):
    if os.path.isfile(i):
        image_list_inp.append(i)
# print(image_list)
image_list_inp = image_list_inp[:5]
# sys.exit(2)
N_CLASSES = 20
NUM_STEPS = len(image_list_inp)
RESTORE_FROM = './checkpoint/CIHP_pgn'


def main():
    """Create the model and start the evaluation process."""
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(image_list_inp, None, False,
                             False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
    image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
    image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
    image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))
         
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
    # parsing net

    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    # edge net
    edge_out2_100 = net_100.layers['edge_rf_fc']
    edge_out2_125 = net_125.layers['edge_rf_fc']
    edge_out2_150 = net_150.layers['edge_rf_fc']
    edge_out2_175 = net_175.layers['edge_rf_fc']


    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


    edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
    edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
    edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
    edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
    edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)
                                           
    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.


    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # prepare ground truth 
    # preds = tf.reshape(pred_all, [-1,])
    # weights = tf.cast(tf.less_equal(gt, N_CLASSES - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # mIoU, update_op_iou = tf.contrib.metrics.streaming_mean_iou(preds, gt, num_classes=N_CLASSES, weights=weights)
    # macc, update_op_acc = tf.contrib.metrics.streaming_accuracy(preds, gt, weights=weights)
    #
    # # precision and recall
    # recall, update_op_recall = tf.contrib.metrics.streaming_recall(res_edge, edge_gt_batch)
    # precision, update_op_precision = tf.contrib.metrics.streaming_precision(res_edge, edge_gt_batch)

    # update_op = tf.group(update_op_iou, update_op_acc, update_op_recall, update_op_precision)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.

    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # evaluate prosessing
    parsing_dir = os.path.join(args.output, 'cihp_parsing_maps')
    if not os.path.exists(parsing_dir):
        os.makedirs(parsing_dir)
    edge_dir = os.path.join(args.output, 'cihp_edge_maps')
    if not os.path.exists(edge_dir):
        os.makedirs(edge_dir)
    # Iterate over training steps.
    for step in range(NUM_STEPS):
        # if step > 100:
        #     break
        print(step)
        parsing_, scores, edge_ = sess.run([pred_all, pred_scores, pred_edge])
        if step % 1 == 0:
            print('step {:d}'.format(step))
            print (image_list[step])
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]
        
        msk = decode_labels(parsing_, num_classes=N_CLASSES)

        parsing_im = Image.fromarray(msk[0])
        # print("here")
        parsing_im.save('{}/{}_vis.png'.format(parsing_dir, img_id))
        cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])
        # sio.savemat('{}/{}.mat'.format(parsing_dir, img_id), {'data': scores[0,:,:]})
        
        cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0,:,:,0] * 255)
        print("here")

    # res_mIou = mIoU.eval(session=sess)
    # res_macc = macc.eval(session=sess)
    # res_recall = recall.eval(session=sess)
    # res_precision = precision.eval(session=sess)
    # f1 = 2 * res_precision * res_recall / (res_precision + res_recall)
    # print('Mean IoU: {:.4f}, Mean Acc: {:.4f}'.format(res_mIou, res_macc))
    # print('Recall: {:.4f}, Precision: {:.4f}, F1 score: {:.4f}'.format(res_recall, res_precision, f1))

    coord.request_stop()
    coord.join(threads)
    


if __name__ == '__main__':
    main()


##############################################################333
