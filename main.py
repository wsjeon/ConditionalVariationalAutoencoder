import numpy as np
import tensorflow as tf
from model import Agent
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags

flags.DEFINE_string('GPU_ID', '0', 'GPU seletion')
flags.DEFINE_float('LR', 0.00025, 'learning rate')
flags.DEFINE_integer('BATCH_SIZE', 100, 'batch size')
flags.DEFINE_integer('SEED', 0, 'random seed for reproducibility')
flags.DEFINE_string('LOGDIR', './tmp', 'path to save event files')
flags.DEFINE_integer('TRAINING_STEP', 500000, 'number of training steps')
flags.DEFINE_boolean('IS_TRAINING', True, 'whether training or not')
flags.DEFINE_boolean('IS_TESTING', False, 'whether testing or not')
flags.DEFINE_string('TEST_LOGDIR', './tmp/run4/0.00025', 'restore the checkpoint file in test phase')
flags.DEFINE_integer('LATENT_DIM', 40, 'dimension of latent variables')

FLAGS = flags.FLAGS

def main():
  """ Load dataset (MNIST) """
  Dataset = input_data.read_data_sets('/home/wsjeon/webdav/datasets/MNIST/',
      validation_size = 5000)

  """ Agent setting """
  import os; os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.GPU_ID
  agent = Agent(Dataset)

  """ Learning """
  if FLAGS.IS_TRAINING:
    agent.learn()

  """ Test """
  if FLAGS.IS_TESTING:
    agent.test()

  agent.sess.close()

if __name__ == "__main__":
  main()
