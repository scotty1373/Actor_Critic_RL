import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import Input
from tensorflow_probability import distributions as tfp
import matplotlib.pyplot as plt
from IPython import display

LEARNING_RATE = 0.001
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make('Pendulum-v0')  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


num_inputs = 4
num_actions = 2
num_hidden = 128
sample_range = 200


class ac_Net:
    def __init__(self, in_num, out_num):
        self.in_num = in_num
        self.out_num = out_num
        self.epsilon = 1e-07
        self.lr = LEARNING_RATE
        self.model = self.layer_build()
        self.action_prob_history = []
        self.critic_prob_history = []
        self.reward_history = []


    def layer_build(self):
        input_ = Input(shape=(self.in_num,))
        common_ = layers.Dense(units=128, activation='tanh')(input_)
        actor_mu = layers.Dense(units=self.out_num, activation='softplus')(common_)
        actor_sigma = layers.Dense(units=self.out_num, activation='softplus')(common_)
        critic = layers.Dense(1)(common_)

        model = models.Model(inputs=input_, outputs=[actor_mu, actor_sigma, critic])
        return model

    def action(self, s):
        mu, sigma, _ = self.model(s)
        mu = np.squeeze(mu)
        sigma = np.squeeze(sigma)
        normal_dist = np.random.normal(loc=mu, scale=sigma, size=sample_range)
        normal_action = np.clip(normal_dist, env.action_space.low, env.action_space.high)
        action = np.random.choice(normal_action)
        self.action_prob_history.append([mu, sigma, action])
        return action

    def critic(self, s, a, s_t1):
        _, _, critic_value = self.model(s)
        _, _, critic_value_t1 = self.model(s_t1)
        td_error =


    def loss_cacula(self, action, td):
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        huber_loss = keras.losses.Huber()
        with tf.GradientTape as tape:
            mu, sigma, _ = self.model.model.output
            pdf = 1 / np.sqrt(2. * np.pi * sigma) * np.exp(- np.square(action - mu) / (2 * np.square(sigma)))
            log_prob = np.log(pdf + self.epsilon )
            actor_loss = log_prob * td
        grad = tape.gradient(actor_loss, self.model.trainable_variables)
        optimizer.minimize(-grad)







    def actor_train(self, s, a, t):
