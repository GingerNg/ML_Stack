
import tensorflow as tf
import numpy as np

x = tf.constant([[1., 1.], [2., 3.]])
# x = tf.placeholder(tf.float32, shape=(1024, 1024))
y= tf.reduce_mean(x)  # 1.5
print(x)
print(y)
print("---")

softmax_y = tf.nn.softmax(x)
sigmoid_y = tf.nn.sigmoid(x)


with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: 此处x还没有赋值.

    # rand_array = np.random.rand(1024, 1024)
    print(sess.run(y))  # Will succeed.
    print(sess.run(softmax_y))
    print(sess.run(sigmoid_y))