import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32.  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 1000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FINAL_EPSILON = 0
INITIAL_EPSILON = 0
# or alternative:
# FINAL_EPSILON = 0.0001  # final value of epsilon
# INITIAL_EPSILON = 0.01  # starting value of epsilon
UPDATE_TIME = 100
EXPLORE = 100000.  # frames over which to anneal epsilon


class RobotCNNDQN:

    def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=[]):
        print("Creating a robot: CNN-DQN")
        # replay memory
        self.replay_memory = deque()
        self.time_step = 0
        self.action = actions
        self.w_embeddings = embeddings
        self.max_len = max_len
        self.num_classes = 5
        self.epsilon = INITIAL_EPSILON

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_size = 40
        self.create_qnetwork()
        self.saver = tf.train.Saver()

    def initialise(self, max_len, embeddings):
        self.max_len = max_len
        self.w_embeddings = embeddings
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])
        self.create_qnetwork()
        self.saver = tf.train.Saver()

    def create_qnetwork(self):
        # read a sentence
        self.process_sentence()
        self.state_confidence = tf.placeholder(
            tf.float32, [None, 1], name="input_confidence")
        self.process_prediction()

        # network weights
        # size of a sentence = 384
        self.w_fc1_s = self.weight_variable([384, 256])
        self.w_fc1_c = self.weight_variable([1, 256])
        self.w_fc1_p = self.weight_variable([20, 256])
        self.b_fc1 = self.bias_variable([256])
        self.w_fc2 = self.weight_variable([256, self.action])
        self.b_fc2 = self.bias_variable([self.action])

        # hidden layers
        self.h_fc1_all = tf.nn.relu(tf.matmul(self.state_content, self.w_fc1_s) + tf.matmul(
            self.state_marginals, self.w_fc1_p) + tf.matmul(self.state_confidence, self.w_fc1_c) + self.b_fc1)
        # Q Value layer
        self.qvalue = tf.matmul(self.h_fc1_all, self.w_fc2) + self.b_fc2
        # action input
        self.action_input = tf.placeholder("float", [None, self.action])
        # reword input
        self.y_input = tf.placeholder("float", [None])

        q_action = tf.reduce_sum(
            tf.multiply(self.qvalue, self.action_input), reduction_indices=1)
        # error function
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        # train method
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        self.sess = tf.Session()
        # ? multiple graphs: how to initialise variables ?
        self.sess.run(tf.global_variables_initializer())

    def train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        next_state_sent_batch = []
        next_state_confidence_batch = []
        next_state_predictions_batch = []
        for item in next_state_batch:
            sent, confidence, predictions = item
            next_state_sent_batch.append(sent)
            next_state_confidence_batch.append(confidence)
            next_state_predictions_batch.append(predictions)

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self.sess.run(self.qvalue, feed_dict={
                                     self.sent: next_state_sent_batch, self.state_confidence: next_state_confidence_batch, self.predictions: next_state_predictions_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(qvalue_batch[i]))
        sent_batch = []
        confidence_batch = []
        predictions_batch = []
        for item in state_batch:
            sent, confidence, predictions = item
            sent_batch.append(sent)
            confidence_batch.append(confidence)
            predictions_batch.append(predictions)

        self.sess.run(self.trainStep, feed_dict={
                      self.y_input: y_batch, self.action_input: action_batch, self.sent: sent_batch, self.state_confidence: confidence_batch, self.predictions: predictions_batch})

        # save network every 10000 iteration
        # if self.time_step % 10000 == 0:
        #    self.saver.save(self.sess, './' +
        #                    'network' + '-dqn', global_step=self.time_step)

    def update(self, observation, action, reward, observation2, terminal):
        self.current_state = observation
        #newState = observation2
        new_state = observation2
        self.replay_memory.append(
            (self.current_state, action, reward, new_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()
        global OBSERVE
        if self.time_step > OBSERVE:
            # Train the network
            self.train_qnetwork()

        self.current_state = new_state
        self.time_step += 1

    def get_action(self, observation):
        print "DQN is smart."
        self.current_state = observation
        sent, confidence, predictions = self.current_state
        # print sent, confidence, predictions
        qvalue = self.sess.run(self.qvalue, feed_dict={self.sent: [
                               sent], self.state_confidence: [confidence], self.predictions: [predictions]})[0]

        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def process_sentence(self):
        seq_len = self.max_len
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        filter_sizes = [3, 4, 5]
        num_filters = 128

        self.sent = tf.placeholder(
            tf.int32, [None, seq_len], name="input_x")
        # dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        dropout_keep_prob = 0.5
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # is able to train
            self.w = tf.Variable(tf.random_uniform(
                [self.vocab_size, embedding_size], -1.0, 1.0), trainable=False, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(
                self.w, self.sent)
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(list(filter_sizes))
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.state_content = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    def process_prediction(self):
        seq_len = self.max_len
        num_classes = self.num_classes
        # Placeholder for input
        self.predictions = tf.placeholder(
            tf.float32, [None, seq_len, num_classes], name="input_predictions")
        filter_sizes = [3]
        num_filters = 20
        self.predictions_expanded = tf.expand_dims(
            self.predictions, -1)
        dropout_keep_prob = 0.5

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_classes, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.predictions_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = conv
                # averagepooling over the outputs
                pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(list(filter_sizes))
        ph_pool = tf.concat(pooled_outputs, 3)
        ph_pool_flat = tf.reshape(ph_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.state_marginals = tf.nn.dropout(
                ph_pool_flat, dropout_keep_prob)

    def update_embeddings(self, embeddings):
        self.w_embeddings = embeddings
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])
        print "Assigning new word embeddings"
        print "New size", self.vocab_size
        self.sess.run(self.w.assign(self.w_embeddings))
        self.time_step = 0
        self.replay_memory = deque()
