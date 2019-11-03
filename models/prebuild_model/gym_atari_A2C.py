import tensorflow as tf
import numpy as np
from models.basic_model import BasicModel
from models.model_backends import Conv3D_normal_LSTM
from models.model_optimizer import A2C
from models.model_outputs import Policy_gradient_output


action_size = 5
entropy_coefficient = 0.5
value_coefficient = 0.2
max_grad_norm = 1
scope = 'global'

class Model(BasicModel):

    def __init__(self):

        self.sess = tf.get_default_session()

        with tf.variable_scope(scope):
            inputs, outputs, lstm_out = Conv3D_normal_LSTM((None, 300, 300, 1))

            self.input = inputs[0]
            self.state_in = inputs[1]

            state_out = outputs[0]
            self.state_init = outputs[1]

            self.policy, self.value = Policy_gradient_output(state_out, action_size)

        self.train_placeholder, self.output_varibal_list = A2C(self.policy, self.value, scope, action_size, entropy_coefficient, value_coefficient, max_grad_norm)

        #[actions, advantages, rewards, lr]
        #[policy_loss, value_loss, entropy, _train]

    def fit(self, environment_observations_list, rnn_state, actions_list, rewards_list, values_list, lr):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages_list = actions_list - values_list

        # We create the feed dictionary
        feed_dict = {}
        feed_dict.update({self.input: environment_observations_list})
        feed_dict.update({self.state_in[i]: rnn_state[i] for i in range(len(self.state_in))})
        feed_dict.update({self.train_placeholder[0]: actions_list})
        feed_dict.update({self.train_placeholder[1]: advantages_list}) # Use to calculate our policy loss
        feed_dict.update({self.train_placeholder[2]: rewards_list}) # Use as a bootstrap for real value
        feed_dict.update({self.train_placeholder[3]: lr})



        policy_loss, value_loss, policy_entropy, _ = self.sess.run(self.output_varibal_list, feed_dict)

        return policy_loss, value_loss, policy_entropy

    # Function use to take a step returns action to take and V(s)
    def predict_step(self, environment_observations, rnn_state):

        feed_dict = {self.input: environment_observations}
        feed_dict.update({self.state_in[i]: rnn_state[i] for i in range(len(self.state_in))})

        return self.sess.run([self.policy, self.value], feed_dict)


    # Function that calculates only the V(s)
    def predict_value(self, environment_observations, rnn_state):

        feed_dict = {self.input: environment_observations}
        feed_dict.update({self.state_in[i]: rnn_state[i] for i in range(len(self.state_in))})

        return self.sess.run(self.value, feed_dict)

    # Function that output only the action to take
    def predict_policy(self, environment_observations, rnn_state):

        feed_dict = {self.input: environment_observations}
        feed_dict.update({self.state_in[i]: rnn_state[i] for i in range(len(self.state_in))})

        return self.sess.run(self.policy, feed_dict)

    def get_init_rnn_state(self):

        return self.state_init














