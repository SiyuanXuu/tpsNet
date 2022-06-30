# Code for
# Reconstructing Dynamic Soft Tissue with Stereo Endoscope Based on a Single-layer Network
# Bo Yang, Siyuan Xu
#
# parts of the code from https://github.com/SiyuanXuu/tpsNet

import numpy as np
import tensorflow as tf


def decoder_forward(theta_in, T_weight, sz_params):
    """
    Do matrix multiplication between the T matrix and the control points vector, theta
    to solve disparity of TPS interpolation area.
    @param theta_in:
    @param T_weight:
    @param sz_params:
    @return:
    """
    # Before interpolation, crop the upper and
    # right edges of the image. The range of cut is determined by image situation.

    disp_vec = tf.map_fn(lambda x: tf.matmul(T_weight, x), theta_in)  # D = T*theta
    d_height, d_width = get_tps_size(sz_params)  # size of cropped image, expect to be 200x200
    disp_tensor = tf.reshape(disp_vec, [tf.shape(theta_in)[0], d_height, d_width, 1])  # shape = [b h w 1]
    paddings = tf.constant(
        [[0, 0], [sz_params.crop_top, sz_params.crop_bottom], [sz_params.crop_left, sz_params.crop_right],
         [0, 0]])  # padding by zeros
    disp_ex = tf.pad(disp_tensor, paddings, "CONSTANT")

    return disp_ex


def decoder_forward2(feature_in, feature_in_base, tps_weight, disp_base, sz_params):
    z_delta = feature_in - feature_in_base
    disp_vec = tf.map_fn(lambda x: tf.matmul(tps_weight, x), z_delta)
    d_height, d_width = get_tps_size(sz_params)
    disp_tensor = tf.reshape(disp_vec, [tf.shape(feature_in)[0], d_height, d_width, 1])
    paddings = tf.constant(
        [[0, 0], [sz_params.crop_top, sz_params.crop_bottom], [sz_params.crop_left, sz_params.crop_right],
         [0, 0]])  # padding by zeros
    disp_ex = tf.pad(disp_tensor, paddings, "CONSTANT")
    disp_ex = tf.add(disp_ex, disp_base)

    return disp_ex


def get_tps_size(sz_params):
    """
    sz_params = size_params(batch=50,
                    height=288,
                    width=360,
                    channel=3,
                    crop_pos)
    """
    # Get the height and width of the TPS interpolation area
    tps_height = np.int32(sz_params.input_height - sz_params.crop_top - sz_params.crop_bottom)
    tps_width = np.int32(sz_params.input_width - sz_params.crop_left - sz_params.crop_right)

    return tps_height, tps_width
