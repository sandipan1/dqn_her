import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)

################################ Code for LSTM update starts here
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def lstm_cell(lstm_size):
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)

def multi_lstm_cell(hiddens):
  return tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden) for hidden in hiddens])

def get_state_variables(batch_size, cell):
  # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    count = 0
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.get_variable("state_c-{:d}-{:d}".format(batch_size,count),initializer=state_c, trainable=False),
            tf.get_variable("state_h-{:d}-{:d}".format(batch_size,count),initializer=state_h, trainable=False)))
        count +=1
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

def get_state_reset_op(state_variables, cell, batch_size):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = cell.zero_state(batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)

@static_vars(lstm_state=None)
def _lstm(cell, inpt, num_actions,scope,reset_lstm_state=False,reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = inpt.shape[0]
        out = tf.cast(inpt.reshape(batch_size,1,inpt.shape[1]),tf.float32)
        if(_lstm.lstm_state == None):
          _lstm.lstm_state = get_state_variables(batch_size,cell)
        reset_op = None
        if(reset_lstm_state == True):
          reset_op = get_state_reset_op(_lstm.lstm_state,cell,batch_size)
        with tf.control_dependencies(reset_op):
          out,state = tf.nn.dynamic_rnn(cell=cell,inputs=out,initial_state=_lstm.lstm_state,dtype=tf.float32)
          update_op = get_state_update_op(_lstm.lstm_state,state)
          with tf.control_dependencies(update_op):
            out = tf.contrib.layers.fully_connected(out[:,0,:], num_outputs=num_actions, activation_fn=None)      
        return out

def lstm(hiddens):
    cell = multi_lstm_cell(hiddens)
    return lambda *args, **kwargs: _lstm(cell,*args, **kwargs)

########################## Code for LSTM update ends here
def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores
        return out


def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)

