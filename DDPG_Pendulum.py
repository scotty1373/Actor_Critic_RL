#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_probability as tfp
from collections import deque
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
import platform
import gym
import time
import os

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.001
MAX_MEMORY_LEN = 10000
MAX_STEP_EPISODE = 3200
TRAINABLE = True
DECAY = 0.99


# if platform.system() == 'windows':
#     temp = os.getcwd()
#     CURRENT_PATH = temp.replace('\\', '/')
# else:
#     CURRENT_PATH = os.getcwd()
# CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
# if not os.path.exists(CURRENT_PATH):
#     os.makedirs(CURRENT_PATH)


class ddpg_Net:
    def __init__(self, shape_in, num_output, range):
        self.initializer = keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.memory = deque(maxlen=MAX_MEMORY_LEN)
        self.train_start = 6000
        self.batch_size = 32
        self.gamma = 0.9
        self.sigma_fixed = 3
        self.critic_input_action_shape = 1
        self.actor_range = range
        self.actor_model = self.actor_net_builder()
        self.critic_model = self.critic_net_build()
        self.actor_target_model = self.actor_net_builder()
        self.critic_target_model = self.critic_net_build()


        # self.actor_target_model.trainable = False
        # self.critic_target_model.trainable = False

        self.actor_history = []
        self.critic_history = []
        self.reward_history = []
        self.weight_hard_update()

    def state_store_memory(self, s, a, r, s_t1):
        self.memory.append((s, a, r, s_t1))

    def actor_net_builder(self):
        input_ = keras.Input(shape=self.input_shape, dtype='float', name='actor_input')
        common = keras.layers.Dense(units=32, activation='tanh',
                                    kernel_initializer=self.initializer,
                                    bias_initializer=self.initializer)(input_)

        actor_ = keras.layers.Dense(units=self.out_shape,
                                    activation='tanh',
                                    kernel_initializer=self.initializer,
                                    bias_initializer=self.initializer)(common)

        model = keras.Model(inputs=input_, outputs=actor_, name='actor')
        return model

    def critic_net_build(self):
        input_state = keras.Input(shape=self.input_shape,
                                  dtype='float', name='critic_state_input')
        input_actor_ = keras.Input(shape=self.critic_input_action_shape,
                                   dtype='float', name='critic_action_angle_input')
        concatenated_layer = keras.layers.Concatenate()([input_state, input_actor_])
        common = keras.layers.Dense(units=32, activation='tanh',
                                    kernel_initializer=self.initializer,
                                    bias_initializer=self.initializer)(concatenated_layer)

        critic_output = keras.layers.Dense(units=self.out_shape)(common)
        model = keras.Model(inputs=[input_state, input_actor_],
                            outputs=critic_output,
                            name='critic')
        return model

    def image_process(self, obs):
        obs = rgb2gray(obs)
        return obs

    def action_choose(self, s):
        actor = self.actor_model(s)
        actor = tf.multiply(actor, self.actor_range)
        return actor

    # Exponential Moving Average update weight
    def weight_soft_update(self):
        for i, j in zip(self.critic_model.trainable_weights, self.critic_target_model.trainable_weights):
            j.assign(j * DECAY + i * (1 - DECAY))
        for i, j in zip(self.actor_model.trainable_weights, self.actor_target_model.trainable_weights):
            j.assign(j * DECAY + i * (1 - DECAY))

    def weight_hard_update(self):
        self.actor_target_model.set_weights(self.actor_model.get_weights())
        self.critic_target_model.set_weights(self.critic_model.get_weights())


    '''
    for now the critic loss return target and real q value, that's
    because I wanna tape the gradient in one gradienttape, if the result
    is not good enough, split the q_real in another gradienttape to update
    actor network!!!
    '''

    def critic_loss(self, s, r, s_t1, a):
        # critic model q real
        q_real = self.critic_model([s, a])
        # target critic model q estimate
        a_t1 = self.actor_target_model(s_t1)    # actor denormalization waiting!!!, doesn't matter with the truth action
        q_estimate = self.critic_target_model([s_t1, a_t1])
        # TD-target
        q_target = r + q_estimate * self.gamma
        return q_target, q_real

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_data = random.sample(self.memory, self.batch_size)
        s_, a_, r_, s_t1_ = zip(*batch_data)
        s_ = np.array(s_, dtype='float').squeeze(axis=1)

        a_ = np.array(a_, dtype='float').squeeze(axis=2)   # ang = a[:, 0, :], acc = a[:, 1, :]

        r_ = np.array(r_, dtype='float').reshape(self.batch_size, -1)

        s_t1_ = np.array(s_t1_, dtype='float').squeeze(axis=1)

        # parameters initiation
        optimizer_actor = keras.optimizers.Adam(-self.learning_rate_a)
        optimizer_critic = keras.optimizers.Adam(self.learning_rate_c)

        with tf.GradientTape(persistent=True) as tape:
            q_target, q_real = self.critic_loss(s_, r_, s_t1_, a_)
            td_error = tf.reduce_mean(tf.square(q_target - q_real))
        grad_critic_loss = tape.gradient(td_error, agent.critic_model.trainable_weights)
        optimizer_critic.apply_gradients(zip(grad_critic_loss, agent.critic_model.trainable_weights))
        del tape

        with tf.GradientTape(persistent=True) as tape:
            a = self.actor_model(s_)
            q = self.critic_model([s_, a])
            actor_loss = tf.reduce_mean(q)
        grad_a = tape.gradient(actor_loss, agent.actor_model.trainable_weights)
        # grad_loss = tape.gradient(a, agent.actor_model.trainable_weights, output_gradients=grad_a)
        optimizer_actor.apply_gradients(zip(grad_a, agent.actor_model.trainable_weights))
        del tape
        agent.sigma_fixed *= .9995
        agent.weight_soft_update()


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    # env.seed(1)
    test_train_flag = TRAINABLE

    action_shape = env.action_space.shape
    state_shape = np.array(env.observation_space.shape)
    action_range = env.action_space.high            # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    agent = ddpg_Net((3,), np.ndim(action_shape), action_range)
    agent.actor_model.summary()
    agent.critic_model.summary()
    epochs = 200
    timestep = 0

    count = 0
    ep_history = []
    for e in range(epochs):
        obs = env.reset()

        obs = obs.reshape(1, 3)
        ep_rh = 0
        for index in range(MAX_STEP_EPISODE):
            env.render()

            actor = agent.action_choose(obs)

            actor = np.clip(np.random.normal(actor, agent.sigma_fixed), -2, 2)

            obs_t1, reward, done, _ = env.step(actor)

            # obs_t1 = np.append(obs[:, :, :, 1:], obs_t1, axis=3)
            obs_t1 = obs_t1.reshape(1, 3)
            c_v = agent.critic_model([obs_t1, actor])
            c_v_target = agent.critic_target_model([obs_t1, actor])
            # if acc >= 0:
            #     action = np.array((ang, acc, 0), dtype='float')
            #     obs_t1, reward, done, _ = env.step(action)
            # else:
            #     action = np.array((ang, 0, -acc), dtype='float')
            #     obs_t1, reward, done, _ = env.step(action)
            reward = (reward + 16) / 16
            ep_rh += reward
            agent.state_store_memory(obs, actor, reward, obs_t1)

            if test_train_flag is True:
                agent.train_replay()

            # print(f'timestep: {timestep},'
            #       f'epoch: {count}, reward: {reward}, actor: {actor},'
            #       f'reward_mean: {np.array(ep_history).sum()} '
            #       f'c_r: {c_v}, c_t: {c_v_target}')

            timestep += 1
            obs = obs_t1
        ep_history.append(ep_rh)
        print(f'epoch: {e},'
              f'reward_mean: {np.array(ep_rh).sum()}, explore: {agent.sigma_fixed}')
        count += 1
    data = np.array(ep_history)
    plt.plot(np.arange(data.shape[0]), data)
    plt.show()
    env.close()