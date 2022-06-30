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

parser = argparse.ArgumentParser(description='alternative train OTPS and test')

parser.add_argument('--max_step', type=int, help='num of sub-train step', default=20)
parser.add_argument('--pretrained', type=bool, help='use last trained theta and T', default=False)
parser.add_argument('--data_size', type=int, help='num of total images', default=200)
parser.add_argument('--batch_size', type=int, help='num of minimum batch size', default=100)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--epoch_num', type=int, help='num of epochs', default=2)
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
print_str = "Step{:4d} | recons loss={:.8f} | smooth loss={:.8f} | norm loss={:.8f} | total loss={:.8f}"

# initial T and theta
T_init = get_T_init(params).astype(np.float32)
theta_init = 83.2 * np.ones([params.batch_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)

# load history trained disp, theta and T if existed
if params.pretrained:
    if os.path.exists(os.path.join(params.output_directory, 'pretrained_invivo_disp.npy')):
        trained_disp = np.load(
            os.path.join(params.output_directory, 'pretrained_invivo_disp.npy'))  # disp of trained 200 images
    else:
        trained_disp = 83.2 * np.ones([params.data_size, params.input_height, params.input_width, 1], dtype=np.float32)
    if os.path.exists(os.path.join(params.output_directory, 'pretrained_invivo_theta.npy')):
        trained_theta = np.load(
            os.path.join(params.output_directory, 'pretrained_invivo_theta.npy'))  # theta of trained 200 images
    else:
        trained_theta = 83.2 * np.ones([params.data_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    theta_baseline = np.mean(trained_theta, axis=0)
    disp_baseline = np.mean(trained_disp, axis=0)
else:
    trained_disp = 83.2 * np.ones([params.data_size, params.input_height, params.input_width, 1], dtype=np.float32)
    trained_theta = 83.2 * np.ones([params.data_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    theta_baseline = 83.2 * np.ones([1, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    disp_baseline = 83.2 * np.ones([1, params.input_height, params.input_width, 1], dtype=np.float32)

# repeat to batch_size
theta_baseline = np.tile(theta_baseline, (params.batch_size, 1, 1))  # mean of theta
disp_baseline = np.tile(disp_baseline, (params.batch_size, 1, 1, 1))  # mean of disp
# output theta, T and disp initialize
out_disp = np.copy(trained_disp)
tps_disp = np.copy(trained_disp)
out_theta = np.copy(trained_theta)

out_T = np.copy(T_init)
theta_batch = 83.2 * np.ones([params.batch_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)

batch_num = int(params.data_size / params.batch_size)

""" train of OTPS """
learning_rate_init = np.float32(params.learning_rate)
optimize_op = tf.train.AdamOptimizer(learning_rate_init)  # std TPS
optimize_op1 = tf.train.AdamOptimizer(learning_rate_init)  # update theta alternatively
optimize_op2 = tf.train.AdamOptimizer(2e-3)  # update T alternatively

# with tf.Graph().as_default(), tf.device('/gpu: 1'):
with tf.Graph().as_default():
    """ graph of training alternative theta and T"""
    left = tf.placeholder(tf.float32,
                          shape=[None, params.input_height, params.input_width,
                                 params.input_channel],
                          name='left_in')
    right = tf.placeholder(tf.float32,
                           shape=[None, params.input_height, params.input_width,
                                  params.input_channel],
                           name='right_in')
    theta_input_f = tf.constant(theta_init, name='theta_in')
    T_weight_true = tf.constant(T_init, name='T_val_base')
    T_weight_f = tf.constant(T_init, name='T_val')
    compensateI = tf.constant(4.3, dtype=tf.float32, name='contr_val')  # light compensation
    theta_in_base = tf.placeholder(tf.float32, shape=(params.batch_size, params.cpts_row * params.cpts_col, 1),
                                   name='theta_base')
    disp_base = tf.placeholder(tf.float32, shape=(params.batch_size, params.input_height, params.input_width, 1),
                               name='disp_base')
    theta_input = tf.Variable(theta_input_f, dtype=tf.float32)
    T_weight = tf.Variable(T_weight_f, dtype=tf.float32)
    update1 = tf.assign(theta_input, theta_input_f)  # bind theta_input and theta_batch
    update2 = tf.assign(T_weight, T_weight_f)  # bind T_weight and T_init
    linear_interpolator = LinearInterpolator(params)
    disp1 = decoder_forward(theta_input, T_weight, linear_interpolator.sz_params)  # calculate std TPS first time
    disp2 = decoder_forward2(theta_input, theta_in_base, T_weight, disp_base,
                             linear_interpolator.sz_params)  # calculate disparity of alternative training
    right_est1 = linear_interpolator.interpolate(left, disp1)  # generate right image of std TPS
    right_est2 = linear_interpolator.interpolate(left, disp2)  # generate right image of alternative training

    # loss for training T
    loss_rec1, loss_smooth1, compa_sum1 = compute_rec_smooth_loss1(right_est1, right, compensateI,
                                                                   linear_interpolator.sz_params, T_weight_true,
                                                                   T_weight)
    # calculate batch loss and single image loss
    loss_rec2, loss_smooth2, compa_sum2 = compute_rec_smooth_loss1(right_est2, right, compensateI,
                                                                   linear_interpolator.sz_params, T_weight_true,
                                                                   T_weight)

    # loss for training theta
    loss_rec3, loss_rec3_per, compa_sum3, loss_smooth3 = compute_rec_smooth_loss2(right_est2, right, compensateI,
                                                                                  linear_interpolator.sz_params,
                                                                                  T_weight_true, T_weight)

    # make sure whether points far from excepted region
    disp_size = get_tps_size(params)
    disp_True = tf.zeros([params.batch_size, params.input_height, params.input_width, 1], dtype=tf.float32)
    disp_vec1 = tf.map_fn(lambda x: tf.matmul(T_weight, x), theta_input)
    disp_vec2 = tf.slice(disp_True, [0, params.crop_top, params.crop_left, 0],
                         [-1, disp_size[0], disp_size[1], -1])
    disp_vec2 = tf.reshape(disp_vec2, [params.batch_size, disp_size[0] * disp_size[1], 1])
    loss_wt_norm = 0.01 * tf.reduce_mean(tf.reduce_mean(tf.square(disp_vec1 - disp_vec2), axis=[1, 2]))
    loss_T = loss_rec2 + loss_smooth2 + loss_wt_norm  # final loss of training T
    loss_theta = loss_rec3  # final loss of training theta, same to TPS

    # set optimizer to special iterative params
    train_op1 = optimize_op1.minimize(loss_theta, var_list=theta_input)
    train_op2 = optimize_op2.minimize(loss_T, var_list=T_weight)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # variable initialize
        loss_rec_temp = 0.
        disp_pre = np.zeros([params.batch_size, params.input_height, params.input_width, 1])
        est_right_pre = np.zeros([params.batch_size, params.input_height, params.input_width, params.input_channel])
        for i in range(params.epoch_num):  # period of entire train T and theta
            print("---------Epoch {}---------".format(i + 1))
            T1 = time.time()
            # update trained_theta and out_T alternatively
            """ train T start"""
            print("[%d / %d] start train T" % (i + 1, params.epoch_num))
            for step in range(0, params.max_step):

                """shuffle training samples"""
                ids = [i for i in range(params.data_size)]
                np.random.shuffle(ids)
                left_ims, right_ims = read_stereo_images(params.source_directory, ids)
                trained_disp = trained_disp[ids]
                trained_theta = trained_theta[ids]
                left_ims = left_ims[ids]
                right_ims = right_ims[ids]
                out_left = np.copy(left_ims)
                out_right = np.copy(right_ims)

                tms_overfit = 0  # record time of over fit, more than 3 times to stop
                loss_rec_val_mean = 0
                loss_smooth_val_mean = 0
                loss_norm_val_mean = 0
                compa_sum = 0

                for batch_idx in range(batch_num):
                    cur_theta = trained_theta[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_disp = trained_disp[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_left = left_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_right = right_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]

                    _, theta_val, T_val, loss_rec_val, loss_smooth_val, loss_norm_val, compa_sum2_val, loss_val, disp_val, est_right_val = sess.run(
                        [train_op2, theta_input, T_weight, loss_rec2, loss_smooth2, loss_wt_norm, compa_sum2,
                         loss_T, disp2, right_est2],
                        feed_dict={theta_input_f: cur_theta, T_weight_f: out_T, left: cur_left, right: cur_right,
                                   theta_in_base: theta_baseline, disp_base: disp_baseline, disp_True: cur_disp})
                    out_T = np.copy(T_val)  # update T each batch
                # update other variable
                for batch_idx in range(batch_num):
                    cur_theta = trained_theta[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_disp = trained_disp[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_left = left_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_right = right_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]

                    theta_val, T_val, loss_rec_val, loss_smooth_val, loss_norm_val, compa_sum3_val, disp_val, est_right_val = sess.run(
                        [theta_input, T_weight, loss_rec3_per, loss_smooth3, loss_wt_norm, compa_sum3, disp2,
                         right_est2],
                        feed_dict={theta_input_f: cur_theta, T_weight_f: out_T, left: cur_left, right: cur_right,
                                   theta_in_base: theta_baseline, disp_base: disp_baseline, disp_True: cur_disp})
                    loss_rec_val_mean += loss_rec_val * params.batch_size
                    loss_smooth_val_mean += loss_smooth_val / batch_num
                    loss_norm_val_mean += loss_norm_val / batch_num
                    compa_sum += compa_sum3_val
                # batch mean loss
                loss_rec_val_mean = loss_rec_val_mean / compa_sum
                loss_val_mean = loss_rec_val_mean + loss_smooth_val_mean + loss_norm_val_mean
                if loss_val_mean - loss_rec_temp > 0:
                    tms_overfit += 1
                else:
                    tms_overfit = 0
                if tms_overfit > 4:
                    break
                loss_rec_var = np.abs(loss_val_mean - loss_rec_temp)
                loss_rec_temp = loss_val_mean
                if step > 0 and (0 == step % 10 or step + 1 == params.max_step):
                    print(
                        print_str.format(step, loss_rec_val_mean, loss_smooth_val_mean, loss_norm_val_mean,
                                         loss_val_mean))  # loss message of current patch
                    out_T = np.copy(T_val)  # update T each batch
                    theta_batch = theta_val
                    if step >= 100:
                        break
            # finished all step update theta and T_val
            theta_val, T_val = sess.run(
                [theta_input, T_weight],
                feed_dict={theta_input_f: theta_batch, T_weight_f: out_T, left: cur_left, right: cur_right,
                           theta_in_base: theta_baseline, disp_base: disp_baseline, disp_True: cur_disp})
            out_T = T_val
            # judge whether theta changed while training T
            print("During training T, is theta keep constant?", theta_val.any() == theta_batch.any())
            print("[%d / %d] finished training of T" % (i + 1, params.epoch_num))
            """ train T close """

            """ train theta start """
            print("[%d / %d] start train theta" % (i + 1, params.epoch_num))
            for batch_idx in range(batch_num):
                cur_theta = out_theta[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                cur_disp = tps_disp[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                cur_left = out_left[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                cur_right = out_right[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                sess.run(update1, feed_dict={theta_input_f: cur_theta})
                sess.run(update2, feed_dict={T_weight_f: out_T})

                for step in range(0, params.max_step):
                    _, theta_val, T_val, loss_rec_val, loss_smooth_val, loss_norm_val, compa_sum3_val, loss_val, disp_val, est_right_val = sess.run(
                        [train_op1, theta_input, T_weight, loss_rec3_per, loss_smooth3, loss_wt_norm, compa_sum3,
                         loss_theta, disp2, right_est2],
                        feed_dict={T_weight_f: out_T, left: cur_left, right: cur_right,
                                   theta_in_base: theta_baseline, disp_base: disp_baseline, disp_True: cur_disp})
                    theta_batch = np.copy(theta_val)
                    if 0 == step % 10 or step + 1 == params.max_step:
                        print(
                            print_str.format(step, params.batch_size * loss_rec_val / compa_sum3_val,
                                             loss_smooth_val, loss_norm_val,
                                             loss_val))
                        # update current step's theta and T
                        theta_batch = np.copy(theta_val)
                        out_T = np.copy(T_val)
                        loss_rec_var = np.abs(loss_rec_val - loss_rec_temp)
                        loss_rec_temp = loss_rec_val

                        if step >= 50:
                            theta_val, T_val, loss_rec_val, loss_smooth_val, loss_norm_val, loss_val, disp_val, est_right_val = sess.run(
                                [theta_input, T_weight, loss_rec3_per, loss_smooth3, loss_wt_norm, loss_theta, disp2,
                                 right_est2],
                                feed_dict={theta_input_f: theta_batch, T_weight_f: out_T, left: cur_left,
                                           right: cur_right,
                                           theta_in_base: theta_baseline, disp_base: disp_baseline,
                                           disp_True: cur_disp})
                            theta_batch = np.copy(theta_val)
                            loss_rec_temp = np.copy(loss_rec_val)
                            out_theta[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size] = theta_val
                            out_disp[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size] = disp_val
                            break

                theta_batch = np.copy(theta_val)  # update theta each step
                print("During training theta, is T keep constant?", out_T.any() == T_val.any())
                print("[%d / %d] finished training of theta" % (i + 1, params.epoch_num))
                # update iterative value each step
                trained_disp = np.copy(out_disp)
                trained_theta = np.copy(out_theta)
                left_ims = out_left.copy()
                right_ims = out_right.copy()

                """ train theta stop """
            T2 = time.time()
            print('Epoch %d cost time:%s s' % (i + 1, (T2 - T1)))

        np.save(os.path.join(params.output_directory, 'disp_trained.npy'), out_disp)
        np.save(os.path.join(params.output_directory, 'theta_trained.npy'), out_theta)
        np.save(os.path.join(params.output_directory, 'T_trained.npy'), out_T)
        print("trained theta, T and disparity have been saved at {}".format(os.path.join(params.output_directory)))
