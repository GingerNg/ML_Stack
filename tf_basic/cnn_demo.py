"""
input [batch, in_height, in_width, in_channels]

filter [filter_height, filter_width, in_channels, out_channels]
filter_shape = [filter_size, embedding_size, 1, num_filters]

output(feature map) [batch, out_height, out_width, out_channels]

一个 epoch（代）是指整个数据集正向反向训练一次。

"""
import tensorflow as tf
input = tf.Variable(tf.random_normal([10,6,6,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")

conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# 一个叫bias的向量加到一个叫value的矩阵上，是向量与矩阵的每一行进行相加，得到的结果和value矩阵大小相同。
# bias的维度必须与value的最后一维相同
h = tf.nn.relu(tf.nn.bias_add(value=conv, bias=b), name="relu")  # 激活函数
# Maxpooling over the outputs
pooled = tf.nn.max_pool(
    h,
    ksize=[1, 2, 2, 1],
    strides=[1, 1, 1, 1],
    padding='VALID',
    name="pool")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(pooled).shape)  # 10,4,4,1