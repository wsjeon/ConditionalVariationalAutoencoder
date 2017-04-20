import numpy as np
import tensorflow as tf
from model import Agent

flags = tf.app.flags

flags.DEFINE_string('GPU_ID', '0', 'GPU seletion')
flags.DEFINE_float('LR', 0.0005, 'learning rate')
flags.DEFINE_integer('BATCH_SIZE', 100, 'batch size')
flags.DEFINE_integer('SEED', 0, 'random seed for reproducibility')
flags.DEFINE_string('LOGDIR', './tmp', 'path to save event files')
flags.DEFINE_integer('TRAINING_STEP', 1000000, 'number of training steps')

FLAGS = flags.FLAGS

def main():
  """ Agent setting """
  import os; os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.GPU_ID
  agent = Agent()

  """ Learning """
  agent.learn()

  """ Test """
  agent.sess.close()

if __name__ == "__main__":
  main()
