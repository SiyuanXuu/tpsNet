from decoder import *
import keras.backend as K

class LinearInterpolator:
    def __init__(self, sz_params):
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
        self.sz_params = sz_params###参数
        self.batch = sz_params.mini_batch_size###batch数
        self.height = sz_params.height##高
        self.width = sz_params.width##宽
        self.height_f = tf.cast(self.height, tf.float32)
        self.width_f  = tf.cast(self.width,  tf.float32)

        x_t, y_t = tf.meshgrid(tf.linspace(0.0,   self.width_f - 1.0,  self.width),
                                   tf.linspace(0.0 , self.height_f - 1.0 , self.height))

        print(x_t,y_t)#288 360 288 360  xt是列的集合
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        print(x_t_flat,y_t_flat)##(1, 103680)
        x_t_flat = tf.tile(x_t_flat, tf.stack([self.batch, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([self.batch, 1]))
        print(x_t_flat,y_t_flat)##(50, 103680)
        self.x_t_flat = tf.reshape(x_t_flat, [-1])
        self.y_t_flat = tf.reshape(y_t_flat, [-1])
        print(self.x_t_flat,self.y_t_flat)###(5184000,)
        
    def _repeat(self, x, n_repeats):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])
    def interpolate(self, img_source0, disp):
        """
        :param img_source: shape=[b, h_s, w_s, c],, dtype=tf.float32
        :param disp: shape=[b, h_s, w_s, 1]
        :return: shape=[b, h_s, w_s, c]
        """

        x = self.x_t_flat + tf.reshape(disp, [-1])###加上视差
        _edge_size = 1
        img_source = tf.pad(img_source0, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        #img_source = img_source0
        x = x + _edge_size
        y = self.y_t_flat + _edge_size
        x = tf.clip_by_value(x, 0.0,  self.width_f - 1 + 2 * _edge_size)###限制上下限

        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f,  self.width_f - 1 + 2 * _edge_size), tf.int32)##上下界

        dim2 = (self.width + 2 * _edge_size)
        dim1 = (self.width + 2 * _edge_size) * (self.height + 2 * _edge_size)
        base = self._repeat(tf.range(self.batch) * dim1, self.height * self.width)
        base_y0 = base + y0 * dim2

        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(img_source, tf.stack([-1, tf.shape(img_source)[3]]))

        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)
        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)

        img_recon = weight_l * pix_l + weight_r * pix_r
        img_recon = tf.reshape(img_recon, tf.shape(img_source0))
        return img_recon

def compute_rec_loss160(est_im, real_im, compensateI, sz_params, A, cu_A, point_meshgrid, yuzhi, belta):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    
    real_clip_gray = tf.image.rgb_to_grayscale(real_clip)
    est_im_gray = tf.image.rgb_to_grayscale(est_clip)
    print(real_clip_gray)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_gray)
    jm = tf.fill(shape,yuzhi)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    #jm = tf.fill(shape,160.0)
    #shape2 = tf.shape(left_im_sum)
    #jm3 = tf.fill(shape2,680.0)

    compa2=tf.less(real_clip_gray,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    compa3=tf.less(est_im_gray,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    
    
    
    compa = tf.to_float(compa2&compa3)
    #compa = tf.to_float(compa3)
    
    #compa = tf.expand_dims(compa, -1)
    #kernel = tf.zeros([3,3,1],dtype = tf.float32)
    kernel = np.array([
    [-1,-1,0,0,0,-1,-1],
    [-1,0,0,0,0,0,-1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [-1,0,0,0,0,0,-1],
    [-1,-1,0,0,0,-1,-1]]).astype(np.float32).reshape(7,7,1)
    compa_erosion = tf.nn.erosion2d(compa, kernel, strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding="SAME")###compa是整个图片计算损失函数的区域
    compa_rgb = tf.tile(compa_erosion,[1,1,1,3])
    
    
    ###batch误差和初除总点数
    #loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb))
    #compa_sum = 3*tf.reduce_sum(compa_erosion)
    #loss_rec = tf.math.divide(loss_rec_sum,compa_sum) 
    
    ###batch每张图片的像素误差的平均
    loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb), axis = [1,2,3])
    print(loss_rec_sum)
    compa_sum = tf.reduce_sum(compa_rgb,axis = [1,2,3])
    print(compa_sum)
    loss_rec = tf.reduce_mean(tf.divide(loss_rec_sum,compa_sum))
    
    
    ####batc每张图片误差和平均
    #loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb), axis = [1,2,3])
    #print(loss_rec_sum)
    #loss_rec = tf.reduce_mean(loss_rec_sum)
    
    #compa_sum = tf.reduce_sum(compa_rgb)###这个batch的mask总点数
    
    #####计算拉普拉斯平滑项
    A_loss = cu_A - A
    A_loss = tf.reshape(A_loss, [1,200,200,-1])
    A_loss_ = tf.transpose(A_loss,[3,1,2,0])
    filter_ = np.array([[1,1,1],[1,-8,1],[1,1,1]]).reshape(3,3,1,1)###3*3,输入3通道，输出3通道
    filter_1 = tf.constant(filter_,dtype = 'float32')
    Laplace_img = tf.nn.conv2d(input = A_loss_,filter = filter_1,strides=[1,1,1,1],padding='SAME')
    print(Laplace_img)####1, 248, 270, 20
    
    loss_A_smooth = belta*tf.reduce_sum(tf.square(Laplace_img))#+0.01*tf.reduce_sum(tf.abs(Laplace_img))##0.8
    
    
    
    
    #####计算每个点的影响是否越界，每张控制图上的16个控制点位置，当前A和初始A之间的差值的平方256
    
    #使用gather函数直接提取对应位置的值进行计算  point_meshgrid
    
    #point_all = tf.gather_nd(A_loss, point_meshgrid)
    #point_all = tf.gather_nd(tf.reshape(A, [1,240,240,16]), point_meshgrid)
    #print(point_all)
    
    #loss_point = 0.1*tf.reduce_sum(tf.square(point_all)) 
    
    return loss_rec, loss_A_smooth, compa_sum

def compute_rec_loss161(est_im, real_im, compensateI, sz_params, A, cu_A, point_meshgrid, yuzhi, belta):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    
    real_clip_gray = tf.image.rgb_to_grayscale(real_clip)
    est_im_gray = tf.image.rgb_to_grayscale(est_clip)
    print(real_clip_gray)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_gray)
    jm = tf.fill(shape,yuzhi)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    
    #shape2 = tf.shape(left_im_sum)
    #jm3 = tf.fill(shape2,680.0)

    compa2=tf.less(real_clip_gray,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    compa3=tf.less(est_im_gray,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    
    
    
    compa = tf.to_float(compa2&compa3)
    #compa = tf.to_float(compa3)
    
    #compa = tf.expand_dims(compa, -1)
    #kernel = tf.zeros([3,3,1],dtype = tf.float32)
    kernel = np.array([
    [-1,-1,0,0,0,-1,-1],
    [-1,0,0,0,0,0,-1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [-1,0,0,0,0,0,-1],
    [-1,-1,0,0,0,-1,-1]]).astype(np.float32).reshape(7,7,1)
    compa_erosion = tf.nn.erosion2d(compa, kernel, strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding="SAME")###compa是整个图片计算损失函数的区域
    compa_rgb = tf.tile(compa_erosion,[1,1,1,3])
    
    
    ###batch误差和初除总点数
    #loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb))
    #compa_sum = 3*tf.reduce_sum(compa_erosion)
    #loss_rec = tf.math.divide(loss_rec_sum,compa_sum) 
    
    ###batch每张图片的像素误差的平均
    loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb), axis = [1,2,3])
    #print(loss_rec_sum)
    compa_sum = tf.reduce_sum(compa_rgb,axis = [1,2,3])
    #print(compa_sum)
    loss_rec = tf.reduce_mean(tf.divide(loss_rec_sum,compa_sum))

    loss_rec_perimg = tf.reduce_mean(loss_rec_sum)
    compa_sum_batch = tf.reduce_sum(compa_sum)
    
    
    ####batc每张图片误差和平均
    #loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI),compa_rgb), axis = [1,2,3])
    #print(loss_rec_sum)
    #loss_rec = tf.reduce_mean(loss_rec_sum)
    
    #compa_sum = tf.reduce_sum(compa_rgb)###这个batch的mask总点数
    
    #####计算拉普拉斯平滑项
    A_loss = cu_A - A
    A_loss = tf.reshape(A_loss, [1,200,200,-1])
    A_loss_ = tf.transpose(A_loss,[3,1,2,0])
    filter_ = np.array([[1,1,1],[1,-8,1],[1,1,1]]).reshape(3,3,1,1)###3*3,输入3通道，输出3通道
    filter_1 = tf.constant(filter_,dtype = 'float32')
    Laplace_img = tf.nn.conv2d(input = A_loss_,filter = filter_1,strides=[1,1,1,1],padding='SAME')
    print(Laplace_img)####1, 248, 270, 20
    
    loss_A_smooth = belta*tf.reduce_sum(tf.square(Laplace_img))#+0.01*tf.reduce_sum(tf.abs(Laplace_img))##0.8
    
    
    
    
    #####计算每个点的影响是否越界，每张控制图上的16个控制点位置，当前A和初始A之间的差值的平方256
    
    #使用gather函数直接提取对应位置的值进行计算  point_meshgrid
    
    #point_all = tf.gather_nd(A_loss, point_meshgrid)
    #point_all = tf.gather_nd(tf.reshape(A, [1,240,240,16]), point_meshgrid)
    #print(point_all)
    
    #loss_point = 0.1*tf.reduce_sum(tf.square(point_all)) 
    
    return loss_rec, loss_A_smooth, loss_rec_perimg, compa_sum_batch

def compute_rec_loss(est_im, real_im, sz_params, A, cu_A, point_meshgrid):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    #est_clip = tf.reshape(est_clip,[-1])
    #real_clip = tf.reshape(real_clip,[-1])
    
    real_clip_sum = tf.reduce_sum(real_clip,axis = 3)
    #left_im_sum = tf.reduce_sum(left_im,axis = 3)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_sum)
    jm = tf.fill(shape,650.0)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    jm2 = tf.fill(shape,0.0)
    
    #shape2 = tf.shape(left_im_sum)
    #jm3 = tf.fill(shape2,680.0)
    
    
    #compa1=tf.less(est_clip,jm)
    #compa1=tf.greater(real_clip_sum,jm2)###全1 这样写0并不会变为1
    compa1=tf.fill(shape,True)###全1
    compa2=tf.less(real_clip_sum,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    #compa3=tf.greater(left_im_sum,jm3)####左图镜面反射区域，所有大于650的点，增加一个固定的视差
    
    
    
    
    
    compa = tf.to_float(compa1&compa2)
    #compa = tf.to_float(compa2)
    
    compa = tf.expand_dims(compa, -1)
    compa_rgb = tf.tile(compa,[1,1,1,3])
    
    
    
    loss_rec = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa_rgb), [3]),[1,2]), name='loss_rec')
    
    
    #####计算拉普拉斯平滑项
    A_loss = tf.subtract(cu_A , A )
    A_loss = tf.reshape(A_loss, [1,240,240,16])
    A_loss_ = tf.transpose(A_loss,[3,1,2,0])
    filter_ = np.array([[1,1,1],[1,-8,1],[1,1,1]]).reshape(3,3,1,1)###3*3,输入3通道，输出3通道
    filter_1 = tf.constant(filter_,dtype = 'float32')
    Laplace_img = tf.nn.conv2d(input = A_loss_,filter = filter_1,strides=[1,1,1,1],padding='SAME')
    print(Laplace_img)####1, 248, 270, 20
    
    loss_A_smooth = 3*tf.reduce_sum(tf.square(Laplace_img))#+0.01*tf.reduce_sum(tf.abs(Laplace_img))
    
    
    
    
    #####计算每个点的影响是否越界，每张控制图上的16个控制点位置，当前A和初始A之间的差值的平方256
    
    #使用gather函数直接提取对应位置的值进行计算  point_meshgrid
    
    point_all = tf.gather_nd(A_loss, point_meshgrid)
    #point_all = tf.gather_nd(tf.reshape(A, [1,240,240,16]), point_meshgrid)
    print(point_all)
    
    loss_point = 1.0*tf.reduce_sum(tf.square(point_all))
    
    
    
    
    
    
    return loss_rec, loss_A_smooth, loss_point
def compute_rec_loss1(est_im, real_im, sz_params):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    #est_clip = tf.reshape(est_clip,[-1])
    #real_clip = tf.reshape(real_clip,[-1])
    
    real_clip_sum = tf.reduce_sum(real_clip,axis = 3)
    #left_im_sum = tf.reduce_sum(left_im,axis = 3)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_sum)
    jm = tf.fill(shape,650.0)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    jm2 = tf.fill(shape,0.0)
    
    #shape2 = tf.shape(left_im_sum)
    #jm3 = tf.fill(shape2,680.0)
    
    
    #compa1=tf.less(est_clip,jm)
    #compa1=tf.greater(real_clip_sum,jm2)###全1 这样写0并不会变为1
    compa1=tf.fill(shape,True)###全1
    compa2=tf.less(real_clip_sum,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    #compa3=tf.greater(left_im_sum,jm3)####左图镜面反射区域，所有大于650的点，增加一个固定的视差
    
    
    compa = tf.to_float(compa1&compa2)
    #compa = tf.to_float(compa2)
    
    compa = tf.expand_dims(compa, -1)
    compa_rgb = tf.tile(compa,[1,1,1,3])
    

    loss_rec = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa_rgb), [3]),[1,2]), name='loss_rec')
    
  
    return loss_rec

def compute_rec_loss2(est_im, real_im, left_im, sz_params):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    ####切割生成的右图和真实的右图，左图并没有进行切割
    
    #est_clip = tf.reshape(est_clip,[-1])
    #real_clip = tf.reshape(real_clip,[-1])

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(est_clip)
    jm = tf.fill(shape,235.0)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    jm2 = tf.fill(shape,0.0)
    
    shape2 = tf.shape(left_im)
    jm3 = tf.fill(shape2,225.0)
    
    
    #compa1=tf.less(est_clip,jm)
    compa1=tf.greater(real_clip,jm2)###全1
    compa2=tf.less(real_clip,jm)###右图没有镜面反射的区域 所有小于240的为1 50,256,256,3
    compa3=tf.greater(left_im,jm3)####左图镜面反射区域，所有大于235的点，增加一个固定的视差
    ##将RGB层压缩为1层，只要有一层为235就为1
    compa1 = tf.reduce_all(compa1,axis = 3)##50,256,256
    compa2 = tf.reduce_any(compa2,axis = 3)##50,256,256 或操作，rgb任意小于235才能为1,为非镜面区域
    compa3 = tf.reduce_all(compa3,axis = 3)###与操作，rgb全部大于235就为镜面区域##############################
    
    
    #####如何使镜面区域对应的一部分可能区域全为0
    
    
    pad1 = np.array([[0, 0], [0, 0], [0, 50]])
    # tf.pad进行填充
    compa3_pad = tf.pad(compa3, pad1, name='pad_1')
    

    
    #计算非镜面反射区域
    compa3_pad2 = tf.logical_not(compa3_pad, name=None)
    
    
    ###通过切割靠右的一部分实现镜面点向左的移位，估计右图对应的区域
    #compa3_1 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 70], [-1, d_height, d_width])
    compa3_2 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 75], [-1, d_height, d_width])
    compa3_3 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 80], [-1, d_height, d_width])
    compa3_4 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 85], [-1, d_height, d_width])
    compa3_5 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 90], [-1, d_height, d_width])
    #compa3_6 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 95], [-1, d_height, d_width])
    
    
    compa = tf.to_float(compa1&compa2&compa3_2&compa3_3&compa3_4&compa3_5)
    #compa = tf.to_float(compa2)
    compa = tf.expand_dims(compa, -1)
    compa = tf.tile(compa,[1,1,1,3])
    #print(compa.shape)
    
    
    loss_rec = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa), [3]),[1,2]), name='loss_rec')
    
    #loss_rec = tf.reduce_mean(tf.square(real_clip - est_clip), name='loss_rec')
    return loss_rec

def compute_rec_loss3(est_im, real_im, left_im, sz_params):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    ####切割生成的右图和真实的右图，左图并没有进行切割
    
    #est_clip = tf.reshape(est_clip,[-1])
    #real_clip = tf.reshape(real_clip,[-1])
    
    
    real_clip_sum = tf.reduce_sum(real_clip,axis = 3)
    left_im_sum = tf.reduce_sum(left_im,axis = 3)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_sum)
    jm = tf.fill(shape,650.0)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    jm2 = tf.fill(shape,0.0)
    
    shape2 = tf.shape(left_im_sum)
    jm3 = tf.fill(shape2,680.0)
    
    
    #compa1=tf.less(est_clip,jm)
    compa1=tf.greater(real_clip_sum,jm2)###全1
    compa2=tf.less(real_clip_sum,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    compa3=tf.greater(left_im_sum,jm3)####左图镜面反射区域，所有大于650的点，增加一个固定的视差
      
    #####如何使镜面区域对应的一部分可能区域全为0
    
    
    pad1 = np.array([[0, 0], [0, 0], [0, 50]])
    # tf.pad进行填充
    compa3_pad = tf.pad(compa3, pad1, name='pad_1')
    

    
    #计算左图非镜面反射区域
    compa3_pad2 = tf.logical_not(compa3_pad, name=None)
    
    
    ###通过切割靠右的一部分实现镜面点向左的移位，估计右图对应的区域
    compa3_1 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 67], [-1, d_height, d_width])
    compa3_2 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 79], [-1, d_height, d_width])
    compa3_3 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 71], [-1, d_height, d_width])
    compa3_4 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 73], [-1, d_height, d_width])
    compa3_5 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 75], [-1, d_height, d_width])
    compa3_6 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 77], [-1, d_height, d_width])
    compa3_7 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 79], [-1, d_height, d_width])
    compa3_8 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 81], [-1, d_height, d_width])
    compa3_9 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 83], [-1, d_height, d_width])
    compa3_10 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 85], [-1, d_height, d_width])
    compa3_11 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 87], [-1, d_height, d_width])
    compa3_12 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 89], [-1, d_height, d_width])
    compa3_13 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 91], [-1, d_height, d_width])
    compa3_14 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 93], [-1, d_height, d_width])
    compa3_15 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 95], [-1, d_height, d_width])
    
    
    compa = tf.to_float(compa1&compa2&compa3_1&compa3_2&compa3_3&compa3_4&compa3_5&compa3_6&compa3_7&compa3_8&compa3_9&compa3_10&compa3_10&compa3_11&compa3_12&compa3_13&compa3_14&compa3_15)
    #compa = tf.to_float(compa1&compa2)
    #compa = tf.to_float(compa2)
    compa = tf.expand_dims(compa, -1)
    
    
    compa = tf.tile(compa,[1,1,1,3])
    #print(compa.shape)
    
    
    loss_rec = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa), [3]),[1,2]), name='loss_rec')
    
    #loss_rec = tf.reduce_mean(tf.square(real_clip - est_clip), name='loss_rec')
    return loss_rec



def compute_rec_loss4(est_im, real_im, left_im, disp, sz_params):
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
    d_height, d_width = get_tps_size(sz_params)
    # d_height = sz_params.height - sz_params.cutTop
    # d_width = sz_params.width - sz_params.cutRight
    est_clip = tf.slice(est_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0,  sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    
    
    
    ####切割生成的右图和真实的右图，左图并没有进行切割
    
    #est_clip = tf.reshape(est_clip,[-1])
    #real_clip = tf.reshape(real_clip,[-1])
    
    
    real_clip_sum = tf.reduce_sum(real_clip,axis = 3)
    left_im_sum = tf.reduce_sum(left_im,axis = 3)

    #平铺   镜面反射：左图右图均有，左图映射过来，视差图为右图，所以只能消去右图的，消去左图会导致全局向镜面反射点收敛
    ##左图的镜面反射点，应当对x小的右图点
    shape = tf.shape(real_clip_sum)
    jm = tf.fill(shape,650.0)#####如何筛选出正确的镜面反射区域,使用全部rgb都大于235来判断，在白机械臂干扰，还是有问题
    jm2 = tf.fill(shape,0.0)
    
    shape2 = tf.shape(left_im_sum)
    jm3 = tf.fill(shape2,680.0)
    
    
    #compa1=tf.less(est_clip,jm)
    #compa1=tf.greater(real_clip_sum,jm2)###全1 这样写0并不会变为1
    compa1=tf.fill(shape,True)###全1
    compa2=tf.less(real_clip_sum,jm)###右图没有镜面反射的区域 所有小于650的为1 50,256,256,3
    compa3=tf.greater(left_im_sum,jm3)####左图镜面反射区域，所有大于650的点，增加一个固定的视差
    
    #####如何使镜面区域对应的一部分可能区域全为0
    
    
    pad1 = np.array([[0, 0], [0, 0], [0, 50]])
    # tf.pad进行填充
    compa3_pad = tf.pad(compa3, pad1, name='pad_1')
    

    
    #计算非镜面反射区域
    compa3_pad2 = tf.logical_not(compa3_pad, name=None)
    
    
    ###通过切割靠右的一部分实现镜面点向左的移位，估计右图对应的区域
    compa3_1 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 67], [-1, d_height, d_width])
    compa3_2 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 79], [-1, d_height, d_width])
    compa3_3 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 71], [-1, d_height, d_width])
    compa3_4 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 73], [-1, d_height, d_width])
    compa3_5 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 75], [-1, d_height, d_width])
    compa3_6 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 77], [-1, d_height, d_width])
    compa3_7 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 79], [-1, d_height, d_width])
    compa3_8 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 81], [-1, d_height, d_width])
    compa3_9 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 83], [-1, d_height, d_width])
    compa3_10 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 85], [-1, d_height, d_width])
    compa3_11 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 87], [-1, d_height, d_width])
    compa3_12 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 89], [-1, d_height, d_width])
    compa3_13 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 91], [-1, d_height, d_width])
    compa3_14 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 93], [-1, d_height, d_width])
    compa3_15 = tf.slice(compa3_pad2, [0,  sz_params.cutTop, sz_params.cutLeft + 95], [-1, d_height, d_width])
    
    compa_bijiao = tf.to_float(compa1&compa2)
    compa_bijiao = tf.expand_dims(compa_bijiao, -1)
    
    
    compa = tf.to_float(compa1&compa2&compa3_1&compa3_2&compa3_3&compa3_4&compa3_5&compa3_6&compa3_7&compa3_8&compa3_9&compa3_10&compa3_10&compa3_11&compa3_12&compa3_13&compa3_14&compa3_15)
    #compa = tf.to_float(compa2)
    compa = tf.expand_dims(compa, -1)
    
    kernel = tf.zeros([2,2,1],dtype = tf.float32)
    compa_erosion = tf.nn.erosion2d(compa, kernel, strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding="SAME")###compa是整个图片计算损失函数的区域
    ###腐蚀，将计算损失的区域减少
    
    compa_rgb = tf.tile(compa_erosion,[1,1,1,3])
    #print(compa.shape)
    compa_rgb_bijiao = tf.tile(compa_bijiao,[1,1,1,3])
    
    loss_rec_batch = tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa_rgb), [3]),[1,2])
    loss_rec = tf.reduce_mean(loss_rec_batch, name='loss_rec')
    #loss_rec = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa_rgb), [3]),[1,2]), name='loss_rec')
    
    
    
    loss_rec_bijiao = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.square(real_clip - est_clip),compa_rgb_bijiao), [3]),[1,2]), name='loss_rec_bijiao')
    
    #loss_rec = tf.reduce_mean(tf.square(real_clip - est_clip), name='loss_rec')
    
    disp_i1 = tf.slice(disp, [0,  32, 14, 0], [-1, 256, 255, -1],
                         name='r_clip')
    disp_i2 = tf.slice(disp, [0,  32, 14+1, 0], [-1, 256, 255, -1],
                         name='r_clip')
    disp_i3 = tf.slice(disp, [0,  32, 14, 0], [-1, 255, 256, -1],
                         name='r_clip')
    disp_i4 = tf.slice(disp, [0,  32+1, 14, 0], [-1, 255, 256, -1],
                         name='r_clip')
        
    loss_wt_norm = tf.square(disp_i2 - disp_i1)##横向
    loss_wt_norm2 = tf.square(disp_i4 - disp_i3)##纵向
    print(compa.shape)
    
    #kernel = K.random_normal(shape = (3, 3, 1))
    
    
    kernel = tf.zeros([21,21,1],dtype = tf.float32)
    #compa_smooth = tf.nn.dilation2d(compa, kernel, strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding="SAME")###需要在反射区域，计算平滑项
    compa_smooth = tf.nn.erosion2d(compa, kernel, strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding="SAME")##腐蚀然后取反
    jm5 = tf.fill(shape,0.5)
    jm5 = tf.expand_dims(jm5, -1)  
    compa_smooth = tf.to_float(tf.less(compa_smooth,jm5))
    
    
    loss_norm_smooth = tf.multiply(0.04, tf.reduce_sum(tf.multiply(loss_wt_norm,compa_smooth[:,:,0:255,:])), name='punishment')+tf.multiply(0.04, tf.reduce_sum(tf.multiply(loss_wt_norm2,compa_smooth[:,0:255,:,:])), name='punishment1')
    #loss_norm_smooth = tf.multiply(0.001, tf.reduce_sum(tf.multiply(loss_wt_norm,compa_smooth[:,:,0:255,:])[:,20:-20,20:-20,:]), name='punishment')+tf.multiply(0.002, tf.reduce_sum(tf.multiply(loss_wt_norm2,compa_smooth[:,0:255,:,:])[:,20:-20,20:-20,:]), name='punishment1')
    
    
    
    return loss_rec, loss_norm_smooth, loss_rec_bijiao, loss_rec_batch