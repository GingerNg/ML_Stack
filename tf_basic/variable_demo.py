import tensorflow as tf
x = tf.Variable(0)
xr = x.read_value()
y = x.assign(3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
    print(sess.run(xr))
    print(sess.run(y))