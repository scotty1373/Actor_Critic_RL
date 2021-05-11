# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
# from tensorflow.compat.v1 import distributions
from tensorflow_probability import distributions
import numpy as np
import os
import sys
from keras import Input
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K

IN_CHANNEL = 1
image_Shape = (80, 80)
tfp = distributions


class ac_Net:
    def __init__(self, channel, shape):
        self.channel = channel
        self.shape = shape
        self.learning_rate = 0.001
        self.a_action = 1
        self.actor = self.build_action_net()

    def build_action_net(self):
        print('Action net building')
        input_S = Input(shape=(80, 80, self.channel))
        # input_V = Input(shape=(self.channel,))
        share = layers.Conv2D(16, kernel_size=(5, 5), strides=(3, 3),
                              activation='relu')(input_S)
        share = layers.MaxPool2D(pool_size=(2, 2))(share)
        share = layers.Conv2D(32, kernel_size=(5, 5), strides=(3, 3),
                              padding='same', activation='relu')(share)
        share = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', activation='relu')(share)
        share = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', activation='relu')(share)
        share = layers.Flatten()(share)

        x = models.Model(inputs=input_S, outputs=share)

        # angle normal output
        angle_mu = layers.Dense(units=64, activation='tanh')(x.output)
        angle_mu = layers.Dense(units=self.a_action, activation='tanh', name='angle_mu')(angle_mu)
        angle_mu_out = models.Model(inputs=x.inputs, outputs=angle_mu)

        angle_sigma = layers.Dense(units=64, activation='tanh')(x.output)
        angle_sigma = layers.Dense(units=self.a_action, activation='tanh', name='angle_sigma')(angle_sigma)
        angle_sigma_out = models.Model(inputs=x.inputs, outputs=angle_sigma)

        # acceleration normal output
        accele_mu = layers.Dense(units=64, activation='softplus')(x.output)
        accele_mu = layers.Dense(units=self.a_action, activation='softplus', name='accele_mu')(accele_mu)
        accele_mu_out = models.Model(inputs=x.inputs, outputs=accele_mu)

        accele_sigma = layers.Dense(units=64, activation='softplus')(x.output)
        accele_sigma = layers.Dense(units=self.a_action, activation='softplus', name='accele_sigma')(accele_sigma)
        accele_sigma_out = models.Model(inputs=x.inputs, outputs=accele_sigma)

        model = models.Model(inputs=x.input, outputs=[angle_mu_out.output,
                                                      angle_sigma_out.output,
                                                      accele_mu_out.output,
                                                      accele_sigma_out.output])

        print('Action net build finished')
        return model

    # model compile loss, using policy grad

    def action_learn(self, state, a, td_error):
        model = self.actor

        accele_mu_out = tf.squeeze(model.accele_mu_out.output * 2)
        accele_sigma_out = tf.squeeze(model.accele_sigma_out.output + 0.1)
        accele_norm_dist = tfp.Normal(accele_mu_out, accele_sigma_out)
        # continuous action sample
        accele_act = tf.clip_by_value(accele_norm_dist.sample(1), clip_value_min=-1, clip_value_max=1)

        log_prob = accele_norm_dist.log_prob(a)
        exp_v = log_prob * td_error
        exp_v += 0.01 * accele_norm_dist.entropy()

    def

if __name__ == '__main__':
    agent = ac_Net(IN_CHANNEL, image_Shape)
    net = agent.build_action_net()
    net.summary()
