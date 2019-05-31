
import tensorflow as tf
import numpy as np

# 返回10*4的矩阵，产生于low和high之间，产生的值是均匀分布的。
w = tf.random_uniform([10, 4], -1.0, 1.0)
print(w)   # 10*4

print(tf.reduce_mean(w))

# tf.reduce_mean 可跨越维度的计算张量各元素的平均值


#
# embedded_chars = tf.nn.embedding_lookup(w, input_x)

data = np.array([[[2],[1]],[[3],[4]],[[6],[7]]])
data = tf.convert_to_tensor(data)
print(data.shape)  # 3*2*1
lk = [[0,1],[1,0],[0,0]]
lookup_data = tf.nn.embedding_lookup(data,lk)

sess = tf.Session()
print(sess.run(lookup_data))

"""
https://www.jianshu.com/p/abea0d9d2436
tf.nn.embedding_lookup的作用就是找到要寻找的embedding data中的对应的行下的vector。
lk[0]也就是[0,1]对应着下面sess.run(lookup_data)的结果恰好是把data中的[[2],[1]],[[3],[4]]
[
    [
        [[2][1]]
        [[3][4]]
    ]


    [
        [[3][4]]
        [[2][1]]
    ]


    [
        [[2][1]]
        [[2][1]]
    ]
]

"""