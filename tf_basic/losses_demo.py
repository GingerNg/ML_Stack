import numpy as np
import tensorflow as tf

y_true = np.array([[0.,1.,1.],[1.,1.,0.]])
y_pred = np.array([[2.,1.,1.],[1.,1.,1.]])

# tf输入Numpy数据时会自动转换为Tensor来处理
# 显式处理
# y_true= tf.convert_to_tensor(y_true)
# y_pred= tf.convert_to_tensor(y_pred)
"""logistic regression loss"""
lll_loss = tf.nn.log_poisson_loss(targets=y_true,
                                  log_input=tf.log(y_pred))

"""sigmoid loss"""
"""
https://www.jianshu.com/p/cf235861311b
z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
sigmoid_cross_entropy_with_logits:
对输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得的结果不至于溢出。
"""
sigmoid_ce_logits = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                        logits=y_pred)
sigmoid_ce_logits_loss = tf.reduce_mean(sigmoid_ce_logits)

y_hat_sigmoid = tf.nn.sigmoid(y_pred)
sigmoid_total_loss = tf.reduce_mean(-y_true*tf.log(y_hat_sigmoid)-(1-y_true)*tf.log(1-y_hat_sigmoid))
# sigmoid_total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_sigmoid), [1]))

"""
https://www.jianshu.com/p/6c9b0cc6978b
If you interpret the scores in y_hat as unnormalized log probabilities, then they are logits.
"""
"""softmax loss"""
softmax_ce_logits = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                        logits=y_pred)
softmax_ce_logits_loss = tf.reduce_mean(softmax_ce_logits)

y_hat_softmax = tf.nn.softmax(y_pred)
softmax_total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), [1]))



with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: 此处x还没有赋值.

    # rand_array = np.random.rand(1024, 1024)
    print(sess.run(lll_loss))

    """sigmoid"""
    print(sess.run(sigmoid_ce_logits_loss))
    print(sess.run(sigmoid_total_loss))
    """softmax"""
    print(sess.run(softmax_ce_logits_loss))
    print(sess.run(softmax_total_loss))

