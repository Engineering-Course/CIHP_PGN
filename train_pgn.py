from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from utils import *

# Set gpus
gpus = [0]
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use

### parameters setting
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/train_rev.txt'
DATA_ID_LIST = './datasets/CIHP/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/CIHP_pgn'
LOG_DIR = './logs/CIHP_pgn'

N_CLASSES = 20
INPUT_SIZE = (512, 512)
BATCH_I = 1
BATCH_SIZE = BATCH_I * len(gpus)
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
POWER = 0.9
p_Weight = 50
e_Weight = 0.005
Edge_Pos_W = 2
with open(DATA_ID_LIST, 'r') as f:
    TRAIN_SET = len(f.readlines())
SAVE_PRED_EVERY = TRAIN_SET / BATCH_SIZE + 1   # save model per epoch  (number of training set / batch)
NUM_STEPS = int(SAVE_PRED_EVERY) * 100 + 1  # 100 epoch



def main():
    RANDOM_SEED = random.randint(1000, 9999)
    tf.set_random_seed(RANDOM_SEED)

    ## Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE

    ## Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReaderPGN(DATA_DIR, LIST_PATH, DATA_ID_LIST, INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
        image_batch, label_batch, edge_batch = reader.dequeue(BATCH_SIZE)

    tower_grads = []
    reuse1 = False
    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))
    optim = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)

    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:
                if i == 0:
                    reuse1 = False
                else:
                    reuse1 = True
                next_image = image_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_label = label_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_edge = edge_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                # Create network.
                with tf.variable_scope('', reuse=reuse1):
                    net = PGNModel({'data': next_image}, is_training=False, n_classes=N_CLASSES, keep_prob=0.9)

                # parsing net
                parsing_out1 = net.layers['parsing_fc']
                parsing_out2 = net.layers['parsing_rf_fc']

                # edge net
                edge_out1_final = net.layers['edge_fc']
                edge_out1_res5 = net.layers['fc1_edge_res5']
                edge_out1_res4 = net.layers['fc1_edge_res4']
                edge_out1_res3 = net.layers['fc1_edge_res3']
                edge_out2_final = net.layers['edge_rf_fc']

                # combine resize
                edge_out1 = tf.image.resize_images(edge_out1_final, tf.shape(next_image)[1:3,])
                edge_out2 = tf.image.resize_images(edge_out2_final, tf.shape(next_image)[1:3,])
                edge_out1_res5 = tf.image.resize_images(edge_out1_res5, tf.shape(next_image)[1:3,])
                edge_out1_res4 = tf.image.resize_images(edge_out1_res4, tf.shape(next_image)[1:3,])
                edge_out1_res3 = tf.image.resize_images(edge_out1_res3, tf.shape(next_image)[1:3,])

                ### Predictions: ignoring all predictions with labels greater or equal than n_classes
                raw_prediction_p1 = tf.reshape(parsing_out1, [-1, N_CLASSES])
                raw_prediction_p2 = tf.reshape(parsing_out2, [-1, N_CLASSES])
                label_proc = prepare_label(next_label, tf.stack(parsing_out1.get_shape()[1:3]), one_hot=False) # [batch_size, h, w]
                raw_gt = tf.reshape(label_proc, [-1,])
                indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, N_CLASSES - 1)), 1)
                gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                prediction_p1 = tf.gather(raw_prediction_p1, indices)
                prediction_p2 = tf.gather(raw_prediction_p2, indices)

                raw_edge = tf.reshape(tf.sigmoid(edge_out2_final), [-1,])
                edge_cond = tf.multiply(tf.cast(tf.greater(raw_edge, 0.1), tf.int32), tf.cast(tf.less_equal(raw_gt, N_CLASSES - 1), tf.int32))
                edge_mask = tf.squeeze(tf.where(tf.equal(edge_cond, 1)), 1)
                gt_edge = tf.cast(tf.gather(raw_gt, edge_mask), tf.int32)
                p1_lc = tf.gather(raw_prediction_p1, edge_mask)
                p2_lc = tf.gather(raw_prediction_p2, edge_mask)

                ### Pixel-wise softmax loss.
                loss_p1_gb = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1, labels=gt))
                loss_p2_gb = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p2, labels=gt))
                loss_p1_lc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p1_lc, labels=gt_edge))
                loss_p2_lc = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p2_lc, labels=gt_edge))
                loss_p1 = loss_p1_lc + loss_p1_gb
                loss_p2 = loss_p2_lc + loss_p2_gb

                ### Sigmoid cross entropy
                edge_pos_mask = tf.equal(next_edge, 1)
                edge_neg_mask = tf.logical_not(edge_pos_mask)
                edge_pos_mask = tf.cast(edge_pos_mask, tf.float32)
                edge_neg_mask = tf.cast(edge_neg_mask, tf.float32)

                total_pixels = tf.cast(tf.shape(next_edge)[1] * tf.shape(next_edge)[2], tf.int32)
                pos_pixels = tf.reduce_sum(tf.to_int32(next_edge))
                neg_pixels = tf.subtract(total_pixels, pos_pixels)
                pos_weight = tf.cast(tf.divide(neg_pixels, total_pixels), tf.float32)
                neg_weight = tf.cast(tf.divide(pos_pixels, total_pixels), tf.float32)

                parsing_mask = tf.cast(tf.greater(next_label, 0), tf.float32)
                edge_gt = tf.cast(next_edge, tf.float32)

                t_loss_e1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_out1, labels=edge_gt)
                loss_e1_pos_gb = tf.reduce_sum(tf.multiply(t_loss_e1, edge_pos_mask), [1, 2])
                loss_e1_neg_gb = tf.reduce_sum(tf.multiply(t_loss_e1, edge_neg_mask), [1, 2])
                loss_e1_pos_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1, parsing_mask), edge_pos_mask), [1, 2])
                loss_e1_neg_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1, parsing_mask), edge_neg_mask), [1, 2])
                loss_e1_pos = (loss_e1_pos_gb + loss_e1_pos_lc)* pos_weight
                loss_e1_neg = (loss_e1_neg_gb + loss_e1_neg_lc) * neg_weight
                loss_e1 = tf.reduce_mean(loss_e1_pos * Edge_Pos_W + loss_e1_neg)

                t_loss_e2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_out2, labels=edge_gt)
                loss_e2_pos_gb = tf.reduce_sum(tf.multiply(t_loss_e2, edge_pos_mask), [1, 2])
                loss_e2_neg_gb = tf.reduce_sum(tf.multiply(t_loss_e2, edge_neg_mask), [1, 2])
                loss_e2_pos_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e2, parsing_mask), edge_pos_mask), [1, 2])
                loss_e2_neg_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e2, parsing_mask), edge_neg_mask), [1, 2])
                loss_e2_pos = (loss_e2_pos_gb + loss_e2_pos_lc)* pos_weight
                loss_e2_neg = (loss_e2_neg_gb + loss_e2_neg_lc) * neg_weight
                loss_e2 = tf.reduce_mean(loss_e2_pos * Edge_Pos_W + loss_e2_neg)

                t_loss_e1_res5 = tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_out1_res5, labels=edge_gt)
                loss_e1_res5_pos_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res5, edge_pos_mask), [1, 2])
                loss_e1_res5_neg_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res5, edge_neg_mask), [1, 2])
                loss_e1_res5_pos_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res5, parsing_mask), edge_pos_mask), [1, 2])
                loss_e1_res5_neg_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res5, parsing_mask), edge_neg_mask), [1, 2])
                loss_e1_res5_pos = (loss_e1_res5_pos_gb + loss_e1_res5_pos_lc)* pos_weight
                loss_e1_res5_neg = (loss_e1_res5_neg_gb + loss_e1_res5_neg_lc) * neg_weight
                loss_e1_res5 = tf.reduce_mean(loss_e1_res5_pos * Edge_Pos_W + loss_e1_res5_neg)

                t_loss_e1_res4 = tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_out1_res4, labels=edge_gt)
                loss_e1_res4_pos_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res4, edge_pos_mask), [1, 2])
                loss_e1_res4_neg_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res4, edge_neg_mask), [1, 2])
                loss_e1_res4_pos_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res4, parsing_mask), edge_pos_mask), [1, 2])
                loss_e1_res4_neg_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res4, parsing_mask), edge_neg_mask), [1, 2])
                loss_e1_res4_pos = (loss_e1_res4_pos_gb + loss_e1_res4_pos_lc)* pos_weight
                loss_e1_res4_neg = (loss_e1_res4_neg_gb + loss_e1_res4_neg_lc) * neg_weight
                loss_e1_res4 = tf.reduce_mean(loss_e1_res4_pos * Edge_Pos_W + loss_e1_res4_neg)

                t_loss_e1_res3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_out1_res3, labels=edge_gt)
                loss_e1_res3_pos_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res3, edge_pos_mask), [1, 2])
                loss_e1_res3_neg_gb = tf.reduce_sum(tf.multiply(t_loss_e1_res3, edge_neg_mask), [1, 2])
                loss_e1_res3_pos_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res3, parsing_mask), edge_pos_mask), [1, 2])
                loss_e1_res3_neg_lc = tf.reduce_sum(tf.multiply(tf.multiply(t_loss_e1_res3, parsing_mask), edge_neg_mask), [1, 2])
                loss_e1_res3_pos = (loss_e1_res3_pos_gb + loss_e1_res3_pos_lc)* pos_weight
                loss_e1_res3_neg = (loss_e1_res3_neg_gb + loss_e1_res3_neg_lc) * neg_weight
                loss_e1_res3 = tf.reduce_mean(loss_e1_res3_pos * Edge_Pos_W + loss_e1_res3_neg)

                loss_parsing = loss_p1 + loss_p2
                loss_edge = loss_e1 + loss_e2 + loss_e1_res5 + loss_e1_res4 + loss_e1_res3
                reduced_loss = loss_parsing * p_Weight + loss_edge * e_Weight

                trainable_variable = tf.trainable_variables()
                grads = optim.compute_gradients(reduced_loss, var_list=trainable_variable)

                tower_grads.append(grads)

                tf.add_to_collection('loss_p', loss_parsing)
                tf.add_to_collection('loss_e', loss_edge)
                tf.add_to_collection('reduced_loss', reduced_loss)

    # Average the gradients
    grads_ave = average_gradients(tower_grads)
    # apply the gradients with our optimizers
    train_op = optim.apply_gradients(grads_ave)

    loss_p_ave = tf.reduce_mean(tf.get_collection('loss_p'))
    loss_e_ave = tf.reduce_mean(tf.get_collection('loss_e'))
    loss_ave = tf.reduce_mean(tf.get_collection('reduced_loss'))

    loss_summary_p = tf.summary.scalar("loss_p_ave", loss_p_ave)
    loss_summary_e = tf.summary.scalar("loss_e_ave", loss_e_ave)
    loss_summary_ave = tf.summary.scalar("loss_ave", loss_ave)
    loss_summary = tf.summary.merge([loss_summary_ave, loss_summary_p, loss_summary_e])
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())


    # Saver for storing checkpoints of the model.
    all_saver_var = tf.global_variables()
    restore_var = [v for v in all_saver_var if 'parsing' not in v.name and 'edge' not in v.name and 'Momentum' not in v.name]
    saver = tf.train.Saver(var_list=all_saver_var, max_to_keep=100)
    loader = tf.train.Saver(var_list=restore_var)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        loss_value = 0
        feed_dict = { step_ph : step }

        # Apply gradients.
        summary, loss_value, par_loss, edge_loss, _ = sess.run([loss_summary, reduced_loss, loss_parsing, loss_edge, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        if step % SAVE_PRED_EVERY == 0:
            save(saver, sess, SNAPSHOT_DIR, step)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, parsing_loss = {:.3f}, edge_loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, par_loss, edge_loss, duration))



def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

if __name__ == '__main__':
    main()


##########################################
