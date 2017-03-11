import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#
# data
#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.placeholder(tf.float32, [None, 10])

#
# model
#
logits = tf.matmul(x, W) + b

#
# loss function
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y, logits=logits))

#
# training
#
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

#
# evaluation
#
p = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y: mnist.test.labels}))
