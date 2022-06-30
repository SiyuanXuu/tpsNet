# Code for
# Reconstructing Dynamic Soft Tissue with Stereo Endoscope Based on a Single-layer Network
# Bo Yang, Siyuan Xu
#
# parts of the code from https://github.com/SiyuanXuu/tpsNet

from __future__ import absolute_import, division, print_function

import argparse

from utils.create_T0 import *
from utils.linear_sample import *
from utils.read_images import *

# only keep warnings and errors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import time
import tensorflow as tf

# set parameters of TPS
parser = argparse.ArgumentParser(description='train and test of standard TPS and alternative TPS')

parser.add_argument('--model', type=str, help='TPS or OTPS', default='OTPS')
parser.add_argument('--test_num', type=int, help='total test images, use for Alter', default=100)
parser.add_argument('--data_size', type=int, help='num of total images, use for TPS', default=300)
parser.add_argument('--batch_size', type=int, help='num of minimum batch size', default=10)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=3e-1)
parser.add_argument('--epoch_num', type=int, help='num of epochs', default=100)
parser.add_argument('--input_height', type=int, help='input height', default=288)
parser.add_argument('--input_width', type=int, help='input width', default=360)
parser.add_argument('--input_channel', type=int, help='input channel', default=3)
parser.add_argument('--crop_top', type=int, help='crop pos[top,bottom,left,right]', default=54)
parser.add_argument('--crop_bottom', type=int, help='crop pos[top,bottom,left,right]', default=34)
parser.add_argument('--crop_left', type=int, help='crop pos[top,bottom,left,right]', default=42)
parser.add_argument('--crop_right', type=int, help='crop pos[top,bottom,left,right]', default=118)
parser.add_argument('--cpts_row', type=int, help='row of control points', default=4)
parser.add_argument('--cpts_col', type=int, help='col of control points', default=4)
parser.add_argument('--output_directory', type=str, help='output directory for test disparities', default='output/')
parser.add_argument('--source_directory', type=str, help='directory to load source images',
                    default='dataset/invivo1_rect/')
parser.add_argument('--T_directory', type=str, help='directory to load T matrix',
                    default='output/T_init.txt')

params = parser.parse_args()

print("load model {}".format(params.model))
# inintial T matrix
if params.model == 'TPS':
    T_init = get_T_init(params).astype(np.float32)
    print("load T matrix from {}".format(params.T_directory))
    theta_out = 83.2 * np.ones([params.data_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    batch_num = int(params.data_size / params.batch_size)
    ids = [x for x in range(params.data_size)]
else:
    print("load T matrix from {}".format(os.path.join(params.output_directory, 'T_trained.npy')))
    T_init = np.load(os.path.join(params.output_directory, 'T_trained.npy'))
    theta_out = 83.2 * np.ones([params.test_num, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    params.batch_size = 1
    batch_num = int(params.test_num / params.batch_size)
    ids = [x + params.data_size for x in range(params.test_num)]
    # special OTPS test params
    params.epoch_num = 1

""" train of theta """
# an appropriate initial value of theta_init
theta_batch = 83.2 * np.ones([params.batch_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
theta_val = np.copy(theta_batch)

""" build graph of training theta"""
with tf.Graph().as_default():
    left = tf.placeholder(tf.float32,
                          shape=[None, params.input_height, params.input_width,
                                 params.input_channel],
                          name='left_in')
    right = tf.placeholder(tf.float32,
                           shape=[None, params.input_height, params.input_width,
                                  params.input_channel],
                           name='right_in')
    theta_input = tf.Variable(tf.constant(theta_batch), dtype=tf.float32, name='theta_in')
    T_weight = tf.constant(T_init, dtype=tf.float32, name='T_const')  # pin T weight
    compensateI = tf.constant(4.3, dtype=tf.float32, name='light_comp')  # light compensation
    disp = decoder_forward(theta_input, T_weight, params)
    linear_interpolator = LinearInterpolator(params)
    right_est = linear_interpolator.interpolate(left, disp)
    loss_rec, compa_sum, loss_rec_sum = compute_rec_loss(right_est, right, compensateI, params, 180.0)
    learning_rate_init = np.float32(params.learning_rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate_init)
    train_op = optimize_op.minimize(loss_rec, var_list=theta_input)  # optimize theta

    """ run graph of training theta """
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(params.epoch_num):
            if params.model == 'TPS':
                print("--------- Epoch {} ---------".format(i))
            else:
                print("---------- Test ----------")
            T1 = time.time()
            left_ims, right_ims = read_stereo_images(params.source_directory, ids)
            left_ims_f = np.array(left_ims, dtype=np.float32)
            right_ims_f = np.array(right_ims, dtype=np.float32)
            theta_pre = np.copy(theta_val)
            # update theta
            for batch_idx in range(0, batch_num):
                left_ims_b = left_ims_f[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                right_ims_b = right_ims_f[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                theta_val, T_val, loss_rec_val, compa_sum_val, loss_rec_sum_val, disp_val, est_right_val, _ = sess.run(
                    [theta_input, T_weight, loss_rec, compa_sum, loss_rec_sum, disp, right_est, train_op],
                    feed_dict={left: left_ims_b, right: right_ims_b})
                theta_out[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size] = theta_val
                if 0 == batch_idx % 10:
                    theta_var_mean = np.mean(theta_val - theta_pre)
                    print(
                        'iter{:4} | Recons loss={:4} | theta_var_mean={:4}'
                            .format(batch_idx, loss_rec_val, theta_var_mean, )
                    )
                    distance = np.mean(np.square(theta_val - theta_pre))
            T2 = time.time()
            print('cost time:%s s' % (T2 - T1))
            disps = np.squeeze(disp_val, axis=3)
            np.save(os.path.join(params.output_directory, 'tps_theta_{}.npy'.format(params.model)), theta_out)
            np.save(os.path.join(params.output_directory, 'tps_disp_{}.npy'.format(params.model)), disps)
            if params.model == 'TPS':
                print("Epoch %d training finished" % i)
            else:
                print("test finished")
            print("save theta and disparity at {}".format(params.output_directory))
