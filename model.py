import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import fully_connected as fc
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

def get_idx(labels, idx_len):
  idx = []
  for i in range(10):
    idx.append(np.where(labels == i)[0][:idx_len])
  return np.array(idx)

def get_batch(batch_size, idx, images, random = True):
  labels_y = np.random.randint(10, size = batch_size)
  if random:
    noise = np.random.randint(2, size = batch_size) * 2 - 1
  else:
    noise = 1
  labels_x = (labels_y + noise) % 10
  idx_y = idx[(labels_y, np.random.randint(idx.shape[1], size = batch_size))]
  idx_x = idx[(labels_x, np.random.randint(idx.shape[1], size = batch_size))]
  images_y = images[idx_y, :]
  images_x = images[idx_x, :]
  return [images_y, images_x], [labels_y, labels_x]

#def plot_data(image, label):


class Agent(object):
  def __init__(self, Dataset):
    """ Load datasets """
    self.Dataset = Dataset
    if FLAGS.IS_TRAINING:
      self.train_images = Dataset.train.images
      self.train_labels = Dataset.train.labels
      self.val_images = Dataset.validation.images
      self.val_labels = Dataset.validation.labels
    if FLAGS.IS_TESTING:
      self.test_images = Dataset.test.images
      self.test_labels = Dataset.test.labels

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
        mean = fc(h1, FLAGS.LATENT_DIM, scope = "fc1")
        logvar = fc(h1, FLAGS.LATENT_DIM, scope = "fc2")
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
    self.log_dir = os.path.join(*(FLAGS.LOGDIR, str(FLAGS.LR), str(FLAGS.LATENT_DIM)))
    run_count = 0
    while True:
      tmp_dir = os.path.join(self.log_dir, 'run%d' % run_count)
      if os.path.exists(tmp_dir):
        run_count += 1
      else:
        break
    self.log_dir = tmp_dir

    self.summary_op = tf.summary.scalar('lower bound (train)', - self.loss)
    self.summary_op_val = tf.summary.scalar('lower bound (val)', - self.loss)
    self.summary_writer = tf.summary.FileWriter(self.log_dir)

  def build_saver(self):
    self.saver = tf.train.Saver(max_to_keep = 1)

  def learn(self):
    idx_train = get_idx(self.train_labels, 4987)
    idx_val = get_idx(self.val_labels, 434)

    for step in range(FLAGS.TRAINING_STEP):
      batch = get_batch(FLAGS.BATCH_SIZE, idx_train, self.train_images)
      _, loss, summary_str = self.sess.run([self.optimizer, self.loss, self.summary_op],
          feed_dict = {self.y_: batch[0][0], self.x_: batch[0][1]})
      self.summary_writer.add_summary(summary_str, (step + 1) * FLAGS.BATCH_SIZE)

      if step % 50 == 0:
        self.saver.save(self.sess, os.path.join(self.log_dir, 'model'), global_step = step)
        print self.log_dir
        batch = get_batch(FLAGS.BATCH_SIZE, idx_val, self.val_images)
        loss_val, summary_str = self.sess.run([self.loss, self.summary_op_val],
            feed_dict = {self.y_: batch[0][0], self.x_:batch[0][1]})
        self.summary_writer.add_summary(summary_str, (step + 1) * FLAGS.BATCH_SIZE)
        print 'step: {0}\t|\
               training samples: {1}\t|\
               lower bound (train): {2}\t|\
               lower bound (val): {3}'\
               .format(step, (step + 1) * FLAGS.BATCH_SIZE, -loss, -loss_val)

  def test(self):
    idx_test = get_idx(self.test_labels, 892)
    self.saver.restore(self.sess, os.path.join(FLAGS.TEST_LOGDIR, 'model-131950'))

    self.y_ = tf.placeholder(tf.float32, [None, 784])

    def net(y_):
      with slim.arg_scope([fc], activation_fn = None, weights_initializer = None,
          biases_initializer = None, reuse = True):
        z = tf.random_normal(tf.stack([tf.shape(y_)[0], 20 * 2]))
        h2 = fc(tf.concat([y_, z], 1), 400 * 2, activation_fn = tf.nn.relu, scope = "fc3")
        x = fc(h2, 784, scope = "fc4")
      return x

    x = net(self.y_)
    print x

    for step in range(100):
      batch = get_batch(1, idx_test, self.test_images)
      output = self.sess.run(x, feed_dict = {self.y_: batch[0][1]})
      print output
#      plt.title('label: {0}'.format(output[1][1]))
      plt.imshow(output.reshape(28, 28), cmap='gray')
      plt.show()
