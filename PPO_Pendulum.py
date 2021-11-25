#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import torch
from torch.distributions import Normal

from collections import deque
import copy
import pandas as pd
import numpy as np
import gym
import time
import os

LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.0001
MAX_STEP_EPISODE_LEN = 2000
TRAINABLE = True
T_len = 64
EPOCHS = 200


# if platform.system() == 'windows':
#     temp = os.getcwd()
#     CURRENT_PATH = temp.replace('\\', '/')
# else:
#     CURRENT_PATH = os.getcwd()
# CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
# if not os.path.exists(CURRENT_PATH):
#     os.makedirs(CURRENT_PATH)

class actor_builder(torch.nn.Module):
    def __init__(self, innershape, outershape, actor_space):
        super(actor_builder, self).__init__()
        self.input_shape = innershape
        self.output_shape = outershape
        self.action_bound = actor_space[1]

        self.common = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True)
        )
        self.mu_out = torch.nn.Linear(128, 1)
        self.mu_tanh = torch.nn.Tanh()
        self.sigma_out = torch.nn.Linear(128, 1)
        self.sigma_tanh = torch.nn.Softplus()

    def forward(self, obs_ac):
        common = self.common(obs_ac)
        mean = self.mu_out(common)
        mean = self.mu_tanh(mean) * self.action_bound
        log_sigma = self.sigma_out(common)
        log_sigma = self.sigma_tanh(log_sigma)

        return mean, log_sigma


class critic_builder(torch.nn.Module):
    def __init__(self, innershape):
        super(critic_builder, self).__init__()
        self.input_shape = innershape
        self.common = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True)
        )
        self.value = torch.nn.Linear(128, 1)

    def forward(self, obs_ac):
        out = self.common(obs_ac)
        out = self.value(out)
        return out


class PPO:
    def __init__(self, inputshape, outputshape, action_space):
        self.action_space = action_space
        self.input_shape = inputshape
        self.output_shape = outputshape
        self._init(self.input_shape, self.output_shape, self.action_space)
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.batch_size = 32
        self.decay_index = 0.95
        # self.sigma = 0.5
        # self.sigma_actor = np.full((T_len, 1), 0.5, dtype='float32')
        self.epilson = 0.2
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)
        self.update_actor_epoch = 3
        self.update_critic_epoch = 3
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, inner, outter, actionspace):
        self.pi = actor_builder(inner, outter, actionspace)
        self.piold = actor_builder(inner, outter, actionspace)
        self.v = critic_builder(inner)
        self.memory = deque(maxlen=T_len)

    def get_action(self, obs_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        mean, sigma = self.pi(obs_)
        # print(f'mu: {mean.cpu().item()}')
        dist = Normal(mean.cpu().detach(), sigma.cpu().detach())
        prob_ = dist.sample()
        log_prob_ = dist.log_prob(prob_)
        return prob_, log_prob_

    def state_store_memory(self, s, a, r, logprob_):
        self.memory.append((s, a, r, logprob_))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, reward_):
        decayed_rd = []
        state_frame = torch.Tensor(singal_state_frame)
        value_target = ppo.v(state_frame).detach().numpy()
        for rd_ in reward_[::-1]:
            value_target = rd_ + value_target * self.decay_index
            decayed_rd.append(value_target)
        decayed_rd.reverse()
        return decayed_rd

    # 计算actor更新用的advantage value
    def advantage_calcu(self, decay_reward, state_t1):
        state_t1 = torch.Tensor(state_t1)
        critic_value_ = self.v(state_t1)
        d_reward = torch.Tensor(decay_reward)
        advantage = d_reward - critic_value_
        return advantage

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, state_t1, d_reward_):
        q_value = torch.Tensor(d_reward_).squeeze(-1)

        target_value = self.v(state_t1).squeeze(-1)
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        self.c_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.c_opt.step()

    def actor_update(self, state, action_, advantage):
        action_ = torch.FloatTensor(action_)
        self.a_opt.zero_grad()
        pi_mean, pi_sigma = self.pi(state)
        pi_mean_old, pi_sigma_old = self.piold(state)

        pi_dist = Normal(pi_mean, pi_sigma)
        pi_dist_old = Normal(pi_mean_old, pi_sigma_old)

        logprob_ = pi_dist.log_prob(action_.reshape(-1, 1))
        logprob_old = pi_dist_old.log_prob(action_.reshape(-1, 1))

        ratio = torch.exp(logprob_ - logprob_old)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

        actor_loss = torch.min(torch.cat((surrogate1, surrogate2), dim=1), dim=1)[0]
        actor_loss = -torch.mean(actor_loss)
        self.history_actor = actor_loss.detach().item()

        actor_loss.backward(retain_graph=True)
        self.a_opt.step()

    def update(self, state, action_, discount_reward_):
        self.hard_update(self.pi, self.piold)
        state_ = torch.Tensor(state)
        act = action_
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_)

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, act, adv)
            print(f'epochs: {self.ep}, timestep: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(state_, d_reward)
            print(f'epochs: {self.ep}, timestep: {self.t}, critic_loss: {self.history_critic}')

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    # env.seed(1)
    test_train_flag = TRAINABLE

    action_shape = [env.action_space.low.item(), env.action_space.high.item()]
    state_shape = np.array(env.observation_space.shape)          # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    ppo = PPO(state_shape.item(), 1, action_shape)

    count = 0
    ep_history = []

    for epoch in range(EPOCHS):
        obs = env.reset()
        obs = obs.reshape(1, 3)
        ep_rh = 0
        ppo.ep += 1
        for t in range(MAX_STEP_EPISODE_LEN):
            env.render()
            action, logprob = ppo.get_action(obs)
            obs_t1, reward, done, _ = env.step(action.detach().numpy().reshape(1, 1))
            obs_t1 = obs_t1.reshape(1, 3)
            reward = (reward + 16) / 16
            ppo.state_store_memory(obs, action.detach().numpy().reshape(1, 1), reward, logprob)
            obs = obs_t1
            ep_rh += reward

            if (t+1) % T_len == 0 or t == MAX_STEP_EPISODE_LEN - 1:
                s_t, a, rd, _ = zip(*ppo.memory)
                s_t = np.concatenate(s_t).squeeze()
                a = np.concatenate(a).squeeze()
                rd = np.concatenate(rd)

                discount_reward = ppo.decayed_reward(obs_t1, rd)

                ppo.update(s_t, a, discount_reward)
                ppo.memory.clear()

            ppo.t += 1
        ep_history.append(ep_rh)
        print(f'epochs: {ppo.ep}, ep_reward: {ep_rh}')
    env.close()