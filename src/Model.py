import tensorflow as tf
import numpy as np
import random


class Model_DNN():
    def __init__(self, args):
        self.args = args
        assert self.args.model_name == 'DNN'

        # Input placeholders
        self.input_view = tf.placeholder(tf.float32)  # 2-D, [batch_size, view_size(width x length x depth)]
        self.actions = tf.placeholder(tf.int32)  # 1-D, [batch_size]
        self.reward = tf.placeholder(tf.float32)  # 1-D, [batch_size]
        self.maxQ = tf.placeholder(tf.float32)  # 1-D, [batch_size], the max Q-value of next state
        self.learning_rate = tf.placeholder(tf.float32)

        self.agent_embeddings = {}

        # Build Graph
        self.x = self.input_view
        last_hidden_size = self.args.view_flat_size
        for i, hidden_size in enumerate(self.args.model_hidden_size):
            with tf.variable_scope('layer_%d' % i):
                self.W = tf.get_variable(name='weights',
                                         initializer=tf.truncated_normal([last_hidden_size, hidden_size], stddev=0.1))
                self.b = tf.get_variable(name='bias', initializer=tf.zeros([hidden_size]))
                last_hidden_size = hidden_size
                self.x = tf.matmul(self.x, self.W) + self.b
                # activation function
                if self.args.activations[i] == 'sigmoid':
                    self.x = tf.sigmoid(self.x)
                elif self.args.activations[i] == 'tanh':
                    self.x = tf.nn.tanh(self.x)
                elif self.args.activations[i] == 'relu':
                    self.x = tf.nn.relu(self.x)

        with tf.variable_scope('layer_output'):
            self.W = tf.get_variable(name='weights',
                                     initializer=tf.truncated_normal([last_hidden_size, self.args.num_actions],
                                                                     stddev=0.1))
            self.b = tf.get_variable(name='bias', initializer=tf.zeros([self.args.num_actions]))
            self.output = tf.matmul(self.x, self.W) + self.b  # batch_size x output_size

        # Train operation
        self.reward_decay = self.args.reward_decay
        self.actions_onehot = tf.one_hot(self.actions, self.args.num_actions)
        self.loss = tf.reduce_mean(
            tf.square(
                (self.reward + self.reward_decay * self.maxQ) - tf.reduce_sum(
                    tf.multiply(self.actions_onehot, self.output), axis=1)
            )
        )
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _inference(self, sess, input_view, if_sample, policy='e_greedy', epsilon=0.1):
        """
        Perform inference for one batch
        :param if_sample: bool; If true, return Q(s,a) for all the actions; If false, return the sampled action.
        :param policy: valid when if_sample=True, sample policy of the actions taken.
                       Available: e_greedy, greedy
        :param epsilon: for e_greedy policy
        :return: numpy array; if_sample=True, [batch_size]; if_sample=False, [batch_size, num_actions]
        """
        assert policy in ['greedy', 'e_greedy']
        value_s_a = sess.run(self.output, {self.input_view: input_view})
        if if_sample:
            if policy == 'greedy':
                actions = np.argmax(value_s_a, axis=1)
                return actions
            if policy == 'e_greedy':
                all_actions = range(self.args.num_actions)
                actions = []
                for i in xrange(len(value_s_a)):
                    if random.random() < epsilon:
                        actions.append(np.random.choice(all_actions))
                    else:
                        actions.append(np.argmax(value_s_a[i]))
                return np.array(actions)
        else:
            return value_s_a

    def infer_actions(self, sess, view_batches, policy='e_greedy', epsilon=0.1):
        ret_actions = []
        ret_actions_batch = []
        for input_view in view_batches:
            batch_id, batch_view = self.process_view_with_emb_batch(input_view)
            actions_batch = self._inference(sess, batch_view, if_sample=True, policy=policy, epsilon=epsilon)
            ret_actions_batch.append(zip(batch_id, actions_batch))
            ret_actions.extend(zip(batch_id, actions_batch))

        return ret_actions, ret_actions_batch

    def infer_max_action_values(self, sess, view_batches):
        ret = []
        for input_view in view_batches:
            batch_id, batch_view = self.process_view_with_emb_batch(input_view)
            value_batches = self._inference(sess, batch_view, if_sample=False)
            ret.append(zip(batch_id, np.max(value_batches, axis=1)))
        return ret

    def process_view_with_emb_batch(self, input_view):
        # parse input into id, view as and concatenate view with embedding
        batch_view = []
        batch_id = []
        for id, view in input_view:
            batch_id.append(id)
            if id in self.agent_embeddings:
                new_view = np.concatenate((self.agent_embeddings[id], view), 0)
                batch_view.append(new_view)
            else:
                new_embedding = np.random.normal(size=[self.args.agent_emb_dim])
                self.agent_embeddings[id] = new_embedding
                new_view = np.concatenate((new_embedding, view), 0)
                batch_view.append(new_view)
        return batch_id, np.array(batch_view)

    def _train(self, sess, input_view, actions, reward, maxQ, learning_rate=0.001):
        feed_dict = {
            self.input_view: input_view,
            self.actions: actions,
            self.reward: reward,
            self.maxQ: maxQ,
            self.learning_rate: learning_rate
        }
        _ = sess.run(self.train_op, feed_dict)

    def train(self, sess, view_batches, actions_batches, rewards, maxQ_batches, learning_rate=0.001):
        def split_id_value(input_):
            ret_id = []
            ret_value = []
            for item in input_:
                ret_id.append(item[0])
                ret_value.append(item[1])
            return ret_id, ret_value

        for i in xrange(len(view_batches)):
            view_id, view_value = self.process_view_with_emb_batch(view_batches[i])
            action_id, action_value = split_id_value(actions_batches[i])
            maxQ_id, maxQ_value = split_id_value(maxQ_batches[i])
            assert view_id == action_id == maxQ_id
            reward_value = []
            for id in view_id:
                if id in rewards:
                    reward_value.append(rewards[id])
                else:
                    reward_value.append(0.)

            self._train(sess, view_value, action_value, reward_value, maxQ_value, learning_rate)

    def save(self, sess, filename):
        saver = tf.train.Saver()
        saver.save(sess, filename)

    def load(self, sess, filename):
        saver = tf.train.Saver()
        saver.restore(sess, filename)

    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            del self.agent_embeddings[id]
