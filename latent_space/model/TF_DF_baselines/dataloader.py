from curses import window
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle
import pdb
import tensorflow_probability as tfp
import csv
import cv2


class transform:
    def __init__(self):
        super(transform, self).__init__()
        parameters = pickle.load(open("./dataset/parameter_merge.pkl", "rb"))
        self.state_m = parameters["state_m"]
        self.state_std = parameters["state_std"]
        self.obs_m = parameters["obs_m"]
        self.obs_std = parameters["obs_std"]

    def state_transform(self, state):
        """
        state -> [batch_size, num_ensemble, dim_x]
        """
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state - self.state_m) / self.state_std
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def state_inv_transform(self, state):
        """
        state -> [batch_size, num_ensemble, dim_x]
        """
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state * self.state_std) + self.state_m
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def obs_transform(self, obs):
        """
        state -> [batch_size, num_ensemble, dim_obs]
        """
        batch_size = obs.shape[0]
        num_ensemble = obs.shape[1]
        dim_x = obs.shape[2]
        obs = tf.reshape(obs, [batch_size * num_ensemble, dim_x])
        obs = (obs - self.obs_m) / self.obs_std
        obs = tf.reshape(obs, [batch_size, num_ensemble, dim_x])
        return obs

    def obs_inv_transform(self, obs):
        """
        state -> [batch_size, num_ensemble, dim_obs]
        """
        batch_size = obs.shape[0]
        num_ensemble = obs.shape[1]
        dim_x = obs.shape[2]
        obs = tf.reshape(obs, [batch_size * num_ensemble, dim_x])
        obs = (obs * self.obs_std) + self.obs_m
        obs = tf.reshape(obs, [batch_size, num_ensemble, dim_x])
        return obs


class DataLoader:
    # TF dataset format
    #       state -> [window, batch_size, 1, 7]
    # observation -> [window, batch_size, 1, 30]
    def __init__(self):
        self.transform_ = transform()
        self.dim_x = 7
        self.dim_obs = 30

    def load_train_data(self, dataset_path, batch_size, window_size, norm):
        dataset = []
        dataset = pickle.load(open(dataset_path, "rb"))
        N = len(dataset["state_gt"])
        select = random.sample(range(0, N - window_size), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation = []
        for idx in select:
            states_gt = []
            state_pre = []
            obs_input = []
            for i in range(window_size):
                gt = dataset["state_gt"][idx + i]
                pre = dataset["state_pre"][idx + i]
                obs = dataset["obs"][idx + i]
                states_gt.append(gt)
                state_pre.append(pre)
                obs_input.append(obs)
            states_gt_save.append(states_gt)
            states_pre_save.append(state_pre)
            observation.append(obs_input)

        # post process the collected data
        states_gt_save = np.array(states_gt_save)
        states_pre_save = np.array(states_pre_save)
        observation = np.array(observation)
        states_gt_save = np.swapaxes(states_gt_save, 0, 1)
        states_pre_save = np.swapaxes(states_pre_save, 0, 1)
        observation = np.swapaxes(observation, 0, 1)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # apply transformation
        if norm == True:
            states_pre_save = self.transform_.state_transform(states_pre_save)
            states_gt_save = self.transform_.state_transform(states_gt_save)
            observation = self.transform_.obs_transform(observation)

        states_pre_save = tf.reshape(
            states_pre_save, [window_size, batch_size, 1, self.dim_x]
        )
        states_gt_save = tf.reshape(
            states_gt_save, [window_size, batch_size, 1, self.dim_x]
        )
        observation = tf.reshape(
            observation, [window_size, batch_size, 1, self.dim_obs]
        )

        return states_pre_save, states_gt_save, observation

    def load_test_data(self, dataset_path, batch_size, window_size, norm):
        dataset = []
        dataset = pickle.load(open(dataset_path, "rb"))
        N = len(dataset["state_gt"])
        select = random.sample(range(0, N - window_size), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation = []
        for idx in select:
            states_gt = []
            state_pre = []
            obs_input = []
            for i in range(window_size):
                gt = dataset["state_gt"][idx + i]
                pre = dataset["state_pre"][idx + i]
                obs = dataset["obs"][idx + i]
                states_gt.append(gt)
                state_pre.append(pre)
                obs_input.append(obs)
            states_gt_save.append(states_gt)
            states_pre_save.append(state_pre)
            observation.append(obs_input)

        # post process the collected data
        states_gt_save = np.array(states_gt_save)
        states_pre_save = np.array(states_pre_save)
        observation = np.array(observation)
        states_gt_save = np.swapaxes(states_gt_save, 0, 1)
        states_pre_save = np.swapaxes(states_pre_save, 0, 1)
        observation = np.swapaxes(observation, 0, 1)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # apply transformation
        if norm == True:
            states_pre_save = self.transform_.state_transform(states_pre_save)
            states_gt_save = self.transform_.state_transform(states_gt_save)
            observation = self.transform_.obs_transform(observation)

        states_pre_save = tf.reshape(
            states_pre_save, [window_size, batch_size, 1, self.dim_x]
        )
        states_gt_save = tf.reshape(
            states_gt_save, [window_size, batch_size, 1, self.dim_x]
        )
        observation = tf.reshape(
            observation, [window_size, batch_size, 1, self.dim_obs]
        )

        return states_pre_save, states_gt_save, observation

    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range(batch_size):
            if n == 0:
                ensemble = tf.reshape(
                    tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x]
                )
            else:
                tmp = tf.reshape(
                    tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x]
                )
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input

    def format_init_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range(batch_size):
            if n == 0:
                ensemble = tf.reshape(
                    tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x]
                )
            else:
                tmp = tf.reshape(
                    tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x]
                )
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input

    def format_EKF_init_state(self, state, batch_size, dim_x):
        P = np.diag(np.ones((dim_x))).astype(np.float32)
        P = tf.convert_to_tensor(P, dtype=tf.float32)
        P = tf.stack([P] * batch_size)
        P = tf.reshape(P, [batch_size, dim_x, dim_x])
        state_input = (state, P)
        return state_input

    def format_particle_state(self, state, batch_size, num_particles, dim_x):
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_particles), [batch_size, num_particles, dim_x])
        for n in range(batch_size):
            if n == 0:
                particles = tf.reshape(
                    tf.stack([state[n]] * num_particles), [1, num_particles, dim_x]
                )
            else:
                tmp = tf.reshape(
                    tf.stack([state[n]] * num_particles), [1, num_particles, dim_x]
                )
                particles = tf.concat([particles, tmp], 0)
        particles = particles + Q
        ud = tfp.distributions.Uniform(low=0.0, high=1.0)
        weights = tf.reshape(
            ud.sample(batch_size * num_particles), [batch_size, num_particles]
        )
        w = tf.reduce_sum(weights, axis=1)
        w = tf.stack([w] * num_particles)
        w = tf.transpose(w, perm=[1, 0])
        weights = weights / w
        state_input = (particles, weights, state)
        return state_input


########################### test dataloader  ######################
# DataLoader = DataLoader()
# states_pre_save, states_gt_save, observation_save = DataLoader.load_train_data(
#     "./dataset/train_dataset_52.pkl", batch_size=8, window_size=3, norm=True
# )
# print(states_pre_save.shape)
# print(states_gt_save.shape)
# print(observation_save.shape)
# print("--------")
#########################################################################
