"""
tf.Session.run()
run(    fetches,   feed_dict=None,    options=None,    run_metadata=None)
tf.Session.run()函数返回值为fetches的执行结果。
如果fetches是一个元素就返回一个值；
若fetches是一个list，则返回list的值，
若fetches是一个字典类型，则返回和fetches同keys的字典。

feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值。
"""
import numpy as np
import tensorflow as tf

# tf.placeholder
# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: 此处x还没有赋值.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
