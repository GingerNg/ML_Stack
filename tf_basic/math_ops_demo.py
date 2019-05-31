
import tensorflow as tf
import numpy as np

x = tf.constant([[1., 1.], [2., 3.]])
x1 = tf.constant([[1., 1.], [2., 3.]])
# x = tf.placeholder(tf.float32, shape=(1024, 1024))
y= tf.reduce_mean(x)  # 1.5
print(x)
print(y)
print("---")

softmax_y = tf.nn.softmax(x)
sigmoid_y = tf.nn.sigmoid(x)

"""乘法"""
multiply_res = tf.multiply(x,x1) # 两个矩阵中对应元素各自相乘
matmul_res = tf.matmul(x,x1) # 将矩阵a乘以矩阵b，生成a * b。


with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: 此处x还没有赋值.

    # rand_array = np.random.rand(1024, 1024)
    print(sess.run(y))  # Will succeed.
    print(sess.run(softmax_y))
    print(sess.run(sigmoid_y))

    print(multiply_res)
    print(matmul_res)