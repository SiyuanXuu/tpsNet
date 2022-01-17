""""
解码器部分：
    decoder_forward函数：就是一个全连接层，将TPS矩阵与编码器生成的控制点做矩阵乘法，生成的就是视差图（这个视差图的是插值区域的视差图，插值区域
                        不是整个图像区域，在做插值之前我们将图像做了一个切割，将图像上边缘和右边缘切掉了，切掉的范围自行确定）；
    get_tps_size()函数：获取做TPS插值区域的高和宽；
    extend_cutted_disp()函数：将插值区域的视差图在上边缘和右边缘填充0，使它恢复成整张图像的大小，方便后面的线性插值重建右图
"""
import tensorflow as tf
import numpy as np


def decoder_forward1(feature_in, tps_weight, sz_params):
    #  weight = tf.Variable(tps_param, dtype=tf.float32, name='fc_weight')
    #  feature_in: shape = [50, 20, 1]
    disp_vec = tf.map_fn(lambda x: tf.matmul(tps_weight, x), feature_in)#tf.matmul表示矩阵相乘，做一个映射 d = Ax
    print(disp_vec)
    d_height, d_width = get_tps_size(sz_params)#就是剪裁之后图片的大小
    disp_tensor = tf.reshape(disp_vec, [tf.shape(feature_in)[0], d_height, d_width, 1])  # shape = [b h w 1]
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)
    # print('decoder_forward|decoder_forward|decoder_forward')
    return disp_ex
def decoder_forward2(feature_in,feature_in_base, tps_weight, disp_base, sz_params):
    
    z_derta = feature_in - feature_in_base
    print(z_derta)
    disp_vec = tf.map_fn(lambda x: tf.matmul(tps_weight, x), z_derta)#tf.matmul表示矩阵相乘，做一个映射，但是要做中心化处理
    #disp_vec = tf.map_fn(lambda x: tf.matmul(tps_weight, x), feature_in)
    print(disp_vec)
    
    
    d_height, d_width = get_tps_size(sz_params)#就是剪裁之后图片的大小
    disp_tensor = tf.reshape(disp_vec, [tf.shape(feature_in)[0], d_height, d_width, 1])  # shape = [b h w 1]
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)
    # print('decoder_forward|decoder_forward|decoder_forward')
    
    disp_ex = tf.add(disp_ex,disp_base)
    print(disp_ex)
    
    return disp_ex


def get_tps_size(sz_params):
    """
    sz_params = size_params(batch=50,
                            height=288,
                            width=360,
                            channel=3,
                            cutTop=20,
                            cutBottom=0,
                            cutLeft=0,
                            cutRight=50)
    """
    # tps_height = tf.constant(sz_params.height - sz_params.cutTop - sz_params.cutBottom, dtype=tf.int32)
    # tps_width = tf.constant(sz_params.width - sz_params.cutLeft - sz_params.cutRight, dtype=tf.int32)
    tps_height = np.int32(sz_params.height - sz_params.cutTop - sz_params.cutBottom)
    tps_width = np.int32(sz_params.width - sz_params.cutLeft - sz_params.cutRight)
    return tps_height, tps_width


def extend_cutted_disp(disp, sz_params):
    d_height, d_width = get_tps_size(sz_params)
    ex_up = tf.zeros([tf.shape(disp)[0], sz_params.cutTop, d_width, tf.shape(disp)[3]])
    ex_bottom = tf.zeros([tf.shape(disp)[0], sz_params.cutBottom, d_width, tf.shape(disp)[3]])
    ex_left = tf.zeros(
        [tf.shape(disp)[0], sz_params.height, sz_params.cutLeft, tf.shape(disp)[3]])
    ex_right = tf.zeros(
        [tf.shape(disp)[0], sz_params.height, sz_params.cutRight, tf.shape(disp)[3]])

    ex_disp = tf.concat([ex_up, disp, ex_bottom], axis=1)
    ex_disp = tf.concat([ex_left, ex_disp, ex_right], axis=2)
    # print('extend_cutted_im|extend_cutted_im|extend_cutted_im')
    return ex_disp
