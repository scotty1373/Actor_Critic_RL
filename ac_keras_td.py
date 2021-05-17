import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import Input
import tensorflow_probability as tfp
import time


LEARNING_RATE = 0.001
LEARNING_RATE_ENHANCE = 0.01
gamma = 0.7
max_steps_per_episode = 10000
epochs = 200


num_inputs = 4
num_actions = 2
num_hidden = 128
step_max = 200


class ac_Net:
    def __init__(self, in_num: int, out_num: int):
        self.in_num = in_num
        self.out_num = out_num
        self.epsilon = 1e-07
        self.lr_a = LEARNING_RATE
        self.lr_c = LEARNING_RATE_ENHANCE
        self.gamma = gamma
        self.action_model = self.action_layer_build()
        self.critic_model = self.critic_layer_build()
        self.action_prob_history = []
        self.critic_prob_history = []
        self.reward_history = []
        self.td_error_history = []

    def action_layer_build(self):
        input_ = Input(shape=(3,))
        common_ = layers.Dense(units=128, activation='relu')(input_)
        common_ = layers.Dense(units=32, activation='relu')(common_)
        actor_mu = layers.Dense(units=self.out_num, activation='sigmoid')(common_)
        actor_sigma = layers.Dense(units=self.out_num, activation='sigmoid')(common_)

        model = models.Model(inputs=input_, outputs=[actor_mu, actor_sigma])
        return model

    def critic_layer_build(self):
        input_ = Input(shape=(3,))
        common_ = layers.Dense(units=128, activation='relu')(input_)
        common_ = layers.Dense(units=32, activation='relu')(common_)
        critic_value = layers.Dense(units=self.out_num, activation='sigmoid')(common_)

        model = models.Model(inputs=input_, outputs=critic_value)
        return model

    # def action(self, s):
    #     mu, sigma, _ = self.model(s)
    #     mu = np.squeeze(mu)
    #     sigma = np.squeeze(sigma)
    #     normal_dist = np.random.normal(loc=mu, scale=sigma, size=sample_range)
    #     normal_action = np.clip(normal_dist, env.action_space.low, env.action_space.high)
    #     action = np.random.choice(normal_action)
    #     self.action_prob_history.append([mu, sigma, action])
    #     return action

    # gradient tape record critic td_error
    def loss_critic(self, s, r, s_t1):
        cv = self.critic_model(s)
        cv_t1 = self.critic_model(s_t1)
        td_error = (cv_t1 * self.gamma + r) - cv
        self.td_error_history.append(td_error)
        return td_error

    # gradient tape record actor loss
    def loss_actor(self, normal_dist, td):
        with tf.GradientTape() as tape:
            # mu, sigma, _ = self.model.model.output
            # pdf = 1 / np.sqrt(2. * np.pi * sigma) * np.exp(- np.square(action - mu) / (2 * np.square(sigma)))

            log_prob = np.log(pdf + self.epsilon)
            actor_loss = -(log_prob * td)
        return actor_loss

    # # loss optimizer La + Lc
    # def loss_op(self, critic_loss, actor_loss):
    #     optimizer = keras.optimizers.Adam(learning_rate=self.lr)
    #     loss_sum = critic_loss + actor_loss
    #     optimizer.apply_gradients(loss_sum, self.model.trainable_variables)


def train_loop(epochs_, max_step):
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped

    action_shape = env.action_space.shape[0]
    state_shape = env.observation_space.shape[0]
    action_range = env.action_space.high[0]

    agent = ac_Net(state_shape, action_shape)
    optimizer_actor = keras.optimizers.Adam(learning_rate=agent.lr_a)
    optimizer_critic = keras.optimizers.Adam(learning_rate=agent.lr_c)
    loss = keras.losses.Huber()
    for epoch in range(epochs_):
        obs = env.reset()
        obs = tf.reshape(obs, (1, 3))
        count = 0
        ep_hs = []
        while True:
            env.render()
            with tf.GradientTape(persistent=True) as tape:
                mu, sigma = agent.action_model(obs)
                mu = tf.squeeze(mu)
                sigma = tf.squeeze(sigma)
                normal_dist = tfp.distributions.Normal(mu, sigma)
                action = tf.clip_by_value(normal_dist.sample(1), -action_range, action_range)
                agent.action_prob_history.append(action)

                obs_t1, reward, done, info = env.step(action)
                reward = reward / 10
                obs_t1 = tf.reshape(obs_t1, (1, 3))

                td_error = agent.loss_critic(obs, reward, obs_t1)
                agent.critic_prob_history.append(td_error)
                log_prob = normal_dist.log_prob(action)
                loss_action = -log_prob * td_error
                loss_critic = loss(td_error, 0)

            grad_action = tape.gradient(loss_action, agent.action_model.trainable_weights)
            grad_critic = tape.gradient(loss_critic, agent.critic_model.trainable_weights)

            optimizer_actor.apply_gradients(zip(grad_action, agent.action_model.trainable_weights))
            optimizer_critic.apply_gradients(zip(grad_critic, agent.critic_model.trainable_weights))
            del tape

            agent.reward_history.append(reward)
            obs = obs_t1
            count += 1
            ep_hs.append(reward)

            if count > max_step:
                ep_sum = sum(ep_hs)
                print(f'epoch:{epoch}, reward: {ep_sum}')
                break
    agent.action_model.save('action_model.h5')
    agent.critic_model.save('critic_model.h5')


if __name__ == '__main__':
    train_loop(epochs, step_max)



