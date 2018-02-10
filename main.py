import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
  with tf.variable_scope(scope, reuse=reuse):
    out = inpt
    out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.tanh)
    out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
    return out
    
    
    
