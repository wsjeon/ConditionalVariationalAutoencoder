import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.examples.tutorials.mnist import input_data
import os

FLAGS = tf.app.flags.FLAGS

class Agent(object):
  def __init__(self):
    """ Fix random sees for reproducibility """
#    tf.set_random_seed(FLAGS.SEED)

    """ TensorFlow graph construction """
    self.build_net()
    self.build_loss()
    self.build_optimizer()
    self.build_summary()
    self.build_saver()

    """ Open TensorFlow session and initialize variables. """
    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())


  def build_net(self):
    self.x_ = tf.placeholder(tf.float32, [None, 784])
    self.y_ = tf.placeholder(tf.float32, [None, 784])

    def net(x_, y_):
      init_w = tf.truncated_normal_initializer(stddev = 0.001)
      init_b = tf.constant_initializer()

      with slim.arg_scope([fc], activation_fn = None, weights_initializer = init_w,
          biases_initializer = init_b):
        h1 = fc(tf.concat([x_, y_], 1), 400 * 2, activation_fn = tf.nn.relu, scope = "fc0")
        mean = fc(h1, 20 * 2, scope = "fc1")
        logvar = fc(h1, 20 * 2, scope = "fc2")
        eps = tf.random_normal(tf.shape(logvar))
        z = mean + tf.exp(0.5 * logvar) * eps # latent variable
        h2 = fc(tf.concat([y_, z], 1), 400 * 2, activation_fn = tf.nn.relu, scope = "fc3")
        x = fc(h2, 784, scope = "fc4")

      return mean, logvar, x

    self.mean, self.logvar, self.x = net(self.x_, self.y_)

  def build_loss(self):
    KLD = - 0.5 * tf.reduce_sum(1 + self.logvar - self.mean ** 2 - tf.exp(self.logvar), axis = 1)
    BCE = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x, labels=self.x_), axis = 1)
    self.loss = tf.reduce_mean(KLD + BCE)

  def build_optimizer(self):
    optimizer = tf.train.AdamOptimizer(FLAGS.LR)
    self.optimizer = optimizer.minimize(self.loss)

  def build_summary(self):
    run_count = 0
    while True:
      tmp_dir = os.path.join(FLAGS.LOGDIR, 'run%d' % run_count)
      if os.path.exists(tmp_dir):
        run_count += 1
      else:
        break
    self.log_dir = os.path.join(tmp_dir, str(FLAGS.LR))

    self.summary_op = tf.summary.scalar('lower bound (train)', - self.loss)
    self.summary_op_test = tf.summary.scalar('lower bound (test)', - self.loss)
    self.summary_writer = tf.summary.FileWriter(self.log_dir)

  def build_saver(self):
    self.saver = tf.train.Saver()

  def learn(self):
    mnist = input_data.read_data_sets('MNIST')
    train_images = mnist.train.images; train_labels = mnist.train.labels
    test_images = mnist.test.images; test_labels = mnist.test.labels
    idx_train = []; idx_test = []
    for i in range(10):
      idx_train.append(np.where(train_labels == i)[0][:4987])
      idx_test.append(np.where(test_labels == i)[0][:892])
    idx_train = np.array(idx_train); idx_test = np.array(idx_test)

    def get_batch(batch_size, idx, images):
      labels_y = np.random.randint(10, size = batch_size)
      labels_x = (labels_y + 1) % 10
      idx_y = idx[(labels_y, np.random.randint(idx.shape[1], size = batch_size))]
      idx_x = idx[(labels_x, np.random.randint(idx.shape[1], size = batch_size))]
      images_y = images[idx_y, :]
      images_x = images[idx_x, :]
      return [images_y, images_x], [labels_y, labels_x]

    for step in range(FLAGS.TRAINING_STEP):
      batch = get_batch(FLAGS.BATCH_SIZE, idx_train, train_images)
      _, loss, summary_str = self.sess.run([self.optimizer, self.loss, self.summary_op],
          feed_dict = {self.y_: batch[0][0], self.x_: batch[0][1]})
      self.summary_writer.add_summary(summary_str, (step + 1) * FLAGS.BATCH_SIZE)

      if step % 50 == 0:
        self.saver.save(self.sess, self.log_dir)
        batch = get_batch(FLAGS.BATCH_SIZE, idx_test, test_images)
        loss_test, summary_str = self.sess.run([self.loss, self.summary_op_test],
            feed_dict = {self.y_: batch[0][0], self.x_:batch[0][1]})
        self.summary_writer.add_summary(summary_str, (step + 1) * FLAGS.BATCH_SIZE)
        print 'step: {0}\t|\
               training samples: {1}\t|\
               lower bound (train): {2}\t|\
               lower bound (test): {3}'\
               .format(step, (step + 1) * FLAGS.BATCH_SIZE, -loss, -loss_test)
