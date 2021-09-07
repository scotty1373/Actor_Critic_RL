#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import os
import sys
import time

lr = 0.001
MNIST_dir = '/home/ubuntu/Python_Project/keras_tfbackend/mnist'

mnist = tf.keras.datasets.mnist
(x, x_l), (x_test, x_test_l) = mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
x_l = tf.one_hot(x_l, 10)
db = tf.data.Dataset.from_tensor_slices((x, x_l))
db = db.shuffle(buffer_size=10)
db = db.batch(batch_size=64)

epochs = 3


def model_builder():
    input_x = Input(shape=(28, 28, 1), dtype='float', name='input')
    x = layers.Conv2D(32, (5, 5), strides=(1, 1), activation='relu')(input_x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='SAME', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    output_x = layers.Dense(units=10, activation='softmax')(x)
    net_ = models.Model(input_x, output_x)
    net_.summary()
    return net_


def accurency(x_data, y_labels):
    out_data = net(x_data)
    pred_index = tf.argmax(out_data, 1)
    correct = tf.equal(pred_index, tf.argmax(y_labels, axis=1))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))
    return acc


if __name__ == '__main__':
    net = model_builder()
    net.summary()

    # '''
    # +++++++using gradient to apply_gradient+++++++
    # '''
    # optimizer = keras.optimizers.Adam(learning_rate=lr)
    # for epoch in range(epochs):
    #     print(f'start epoch {epoch}')
    #     for batch_idx, (x_batch, y_batch) in enumerate(db):
    #
    #         with tf.GradientTape() as tape:
    #             x_batch = tf.reshape(x_batch, (-1, 28, 28, 1))
    #             out = net(x_batch)
    #             loss = tf.reduce_sum(tf.square(out - y_batch)) / x_batch.shape[0]
    #             # loss_grad = loss(y_batch, out)
    #         grad = tape.gradient(loss, net.trainable_variables)
    #
    #         optimizer.apply_gradients(zip(grad, net.trainable_variables))
    #
    #         if batch_idx % 100 == 0:
    #             acc = accurency(tf.reshape(x, (-1, 28, 28, 1)), x_l)
    #             print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss.numpy()}, acc: {acc}')
    #
    # net.save('model.h5')
    # print('loss:', loss)

    '''
    optimizer.minimaize == tf.GradientTape + gradient + apply_gradients
    '''
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    '''
    loss == 'callable' 
    '''
    # loss = lambda: tf.reduce_sum(tf.square(net(tf.reshape(input, [-1, 28, 28, 1])) - output)) /\
    #                tf.reshape(input, [-1, 28, 28, 1]).shape[0]
    # for epoch in range(epochs):
    #     for batch_idx, (input, output) in enumerate(db):
    #         optimizer.minimize(loss, var_list=net.trainable_variables)
    #         if batch_idx % 100 == 0:
    #             print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss().numpy()}')

    '''
    loss == Tensor 
    '''
    for epoch in range(epochs):
        for batch_idx, (input, output) in enumerate(db):
            with tf.GradientTape() as tape:
                x_batch = tf.reshape(input, (-1, 28, 28, 1))
                out = net(x_batch)
                loss = tf.reduce_sum(tf.square(out - output)) / x_batch.shape[0]
            optimizer.minimize(loss, var_list=net.trainable_variables, tape=tape)
            if batch_idx % 100 == 0:
                print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss.numpy()}')


    time.sleep(1)
