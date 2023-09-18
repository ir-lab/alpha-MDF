# Copyright (c) 2020 Max Planck Gesellschaft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_probability as tfp

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# tf.compat.v1.disable_eager_execution()
"""
This is the code for setting up a differentiable version of the ensemble kalman filter
The filter is trained using simulated/real data where we have access to the ground truth state at each timestep
The filter is suppose to learn the process noise model Q, observation noise model R, the process model f(.) 
and the observation model h(.)
Author: Xiao Liu -> I have made decent amount of changes to the original codebase.
"""


class SensorModel(tf.keras.Model):
    """
    sensor model is used for learning a representation of the current observation,
    the representation can be explainable or latent.
    observation = [batch_size, img_h, img_w, channel]
    encoding = [batch_size, dim_fc3]
    # 128, 64, 64, 32, 32, 32, fc 128, 64, 32, 32, dim_z
    """

    def __init__(self, batch_size, num_ensemble, dim_z):
        super(SensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

    def build(self, input_shape):
        self.sensor_fc1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="sensor_fc1",
        )

        self.sensor_fc2 = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="sensor_fc2",
        )

        self.sensor_fc3 = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="sensor_fc3",
        )

        # bayesian neural networks
        self.bayes_sensor_fc1 = tfp.layers.DenseFlipout(
            units=256, activation=tf.nn.relu, name="bayes_sensor_fc1"
        )
        self.bayes_sensor_fc2 = tfp.layers.DenseFlipout(
            units=64, activation=tf.nn.relu, name="bayes_sensor_fc2"
        )
        self.bayes_sensor_fc3 = tfp.layers.DenseFlipout(
            units=self.dim_z, activation=tf.nn.relu, name="bayes_sensor_fc3"
        )

    def call(self, input):
        fc1 = self.sensor_fc1(input)
        fc2 = self.sensor_fc2(fc1)
        fc3 = self.sensor_fc3(fc2)

        bayes_fc1 = self.bayes_sensor_fc1(fc3)
        bayes_fc2 = self.bayes_sensor_fc2(bayes_fc1)
        observation = self.bayes_sensor_fc3(bayes_fc2)

        encoding = bayes_fc2
        observation = tf.reshape(observation, [self.batch_size, 1, self.dim_z])
        encoding = tf.reshape(encoding, [self.batch_size, 64])

        return observation, encoding


class ProcessModel(tf.keras.Model):
    def __init__(self, batch_size, dim_x, jacobian):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.jacobian = jacobian
        self.dim_x = dim_x

    def build(self, input_shape):
        self.process_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_fc1",
        )
        self.process_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_fc2",
        )
        self.process_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_fc3",
        )

    def call(self, last_state):
        if self.jacobian:
            new_state, F = self.with_jacobian(last_state)
        else:
            fc1 = self.process_fc1(last_state)
            fc2 = self.process_fc2(fc1)
            update = self.process_fc3(fc2)

            new_state = last_state + update
            F = None

        return new_state, F

    @tf.function
    def with_jacobian(self, last_state):
        with tf.GradientTape() as tape:
            tape.watch(last_state)
            fc1 = self.process_fc1(last_state)
            fc2 = self.process_fc2(fc1)
            update = self.process_fc3(fc2)

            new_state = last_state + update
        F = tape.batch_jacobian(new_state, last_state)
        F = tf.stop_gradient(F)

        return new_state, F


class ProcessNoise(tf.keras.Model):
    """
    Noise model is asuming the noise to be heteroscedastic
    The noise is not constant at each step
    The fc neural network is designed for learning the diag(Q)
    Q = [batch_size, dim_x, dim_x]
    i.e.,
    if the state has 4 inputs
    state vector 4 -> fc 32 -> fc 64 -> 4
    the result is the diag of Q where Q is a 4x4 matrix
    """

    def __init__(self, batch_size, dim_x, q_diag):
        super(ProcessNoise, self).__init__()
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.q_diag = q_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_x) * 1e-3
        init = np.sqrt(np.square(self.q_diag) - constant)
        self.fixed_process_noise_bias = self.add_weight(
            name="fixed_process_noise_bias",
            shape=[self.dim_x],
            regularizer=tf.keras.regularizers.l2(l=1e-3),
            initializer=tf.constant_initializer(constant),
        )
        self.process_noise_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_noise_fc1",
        )
        self.process_noise_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_noise_fc_add1",
        )
        self.process_noise_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_noise_fc2",
        )
        self.process_noise_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="process_noise_fc3",
        )
        self.learned_process_noise_bias = self.add_weight(
            name="learned_process_noise_bias",
            shape=[self.dim_x],
            regularizer=tf.keras.regularizers.l2(l=1e-3),
            initializer=tf.constant_initializer(init),
        )

    def call(self, state):
        fc1 = self.process_noise_fc1(state)
        fcadd1 = self.process_noise_fc_add1(fc1)
        fc2 = self.process_noise_fc2(fcadd1)
        diag = self.process_noise_fc3(fc2)

        diag = tf.square(diag + self.learned_process_noise_bias)
        diag = diag + self.fixed_process_noise_bias
        Q = tf.linalg.diag(diag)
        Q = tf.reshape(Q, [self.batch_size, self.dim_x, self.dim_x])

        return Q


class ObservationModel(tf.keras.Model):
    """
    Observation matrix H is given, which does not require learning
    the jacobians. It requires one's knowledge of the whole system
    z_pred = [batch_size, 1, dim_z]
    """

    def __init__(self, batch_size, dim_z):
        super(ObservationModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z

    def call(self, state):
        H = tf.concat(
            [
                tf.tile(
                    np.array([[[1, 0, 0, 0, 0, 0, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 1, 0, 0, 0, 0, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 0, 1, 0, 0, 0, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 0, 0, 0, 1, 0, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 0, 0, 0, 0, 1, 0]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
                tf.tile(
                    np.array([[[0, 0, 0, 0, 0, 0, 1]]], dtype=np.float32),
                    [self.batch_size, 1, 1],
                ),
            ],
            axis=1,
        )
        z_pred = tf.matmul(H, tf.transpose(state, perm=[0, 2, 1]))
        Z_pred = tf.transpose(z_pred, perm=[0, 2, 1])
        z_pred = tf.reshape(z_pred, [self.batch_size, 1, self.dim_z])

        return z_pred, H


class ObservationNoise(tf.keras.Model):
    """
    observation noise model is used for estimating the aleatoric noise.
    inputs: an intermediate representation of the raw observation
    denoted as encoding.
    R = [batch_size, dim_z, dim_z]
    The fc neural network is designed for learning the diag(R)
    i.e.,
    if the state has 4 inputs, the encoding has size 64,
    observation vector z is with size 2, the R has the size
    2 + (64 -> fc 2 -> 2) + fixed noise,
    the result is the diag of R where R is a 2x2 matrix
    """

    def __init__(self, batch_size, dim_z, r_diag):
        super(ObservationNoise, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.r_diag = r_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        self.fixed_observation_noise_bias = self.add_weight(
            name="fixed_observation_noise_bias",
            shape=[self.dim_z],
            regularizer=tf.keras.regularizers.l2(l=1e-3),
            initializer=tf.constant_initializer(constant),
        )
        self.observation_noise_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="observation_noise_fc1",
        )
        self.observation_noise_fc2 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name="observation_noise_fc2",
        )
        self.learned_observation_noise_bias = self.add_weight(
            name="learned_observation_noise_bias",
            shape=[self.dim_z],
            regularizer=tf.keras.regularizers.l2(l=1e-3),
            initializer=tf.constant_initializer(init),
        )

    def call(self, inputs):
        diag = self.observation_noise_fc1(inputs)
        diag = self.observation_noise_fc2(diag)
        diag = tf.square(diag + self.learned_observation_noise_bias)
        diag = diag + self.fixed_observation_noise_bias

        R = tf.linalg.diag(diag)
        R = tf.reshape(R, [self.batch_size, self.dim_z, self.dim_z])
        diag = tf.reshape(diag, [self.batch_size, self.dim_z])

        return R, diag


class utils:
    def __init__(self):
        super(utils, self).__init__()
        self.scale = 1

    ###########################################################################
    # convenience functions for ensuring stability

    ###########################################################################
    def _condition_number(self, s):
        """
        Compute the condition number of a matrix based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        Returns
        -------
        r_corrected : tensor
            the condition number of the matrix
        """
        r = s[..., 0] / s[..., -1]

        # Replace NaNs in r with infinite
        r_nan = tf.math.is_nan(r)
        r_inf = tf.fill(tf.shape(r), tf.constant(np.Inf, r.dtype))
        r_corrected = tf.where(r_nan, r_inf, r)

        return r_corrected

    def _is_invertible(self, s, epsilon=1e-6):
        """
        Check if a matrix is invertible based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        epsilon : float, optional
            threshold for the condition number
        Returns
        -------
        invertible : tf.bool tensor
            true if the matrix is invertible
        """
        # "c"
        # Epsilon may be smaller with tf.float64
        eps_inv = tf.cast(1.0 / epsilon, s.dtype)
        cond_num = self._condition_number(s)
        invertible = tf.logical_and(
            tf.math.is_finite(cond_num), tf.less(cond_num, eps_inv)
        )
        return invertible

    def _make_valid(self, covar):
        """
        Trys to make a possibly degenerate covariance valid by
          - replacing nans and infs with high values/zeros
          - making the matrix symmetric
          - trying to make the matrix invertible by adding small offsets to
            the smallest eigenvalues
        Parameters
        ----------
        covar : tensor
            a covariance matrix that is possibly degenerate
        Returns
        -------
        covar_valid : tensor
            a covariance matrix that is hopefully valid
        """
        # eliminate nans and infs (replace them with high values on the
        # diagonal and zeros else)
        bs = covar.get_shape()[0]
        dim = covar.get_shape()[-1]
        covar = tf.where(
            tf.math.is_finite(covar), covar, tf.eye(dim, batch_shape=[bs]) * 1e6
        )

        # make symmetric
        covar = (covar + tf.linalg.matrix_transpose(covar)) / 2.0

        # add a bit of noise to the diagonal of covar to prevent
        # nans in the gradient of the svd
        noise = tf.random.uniform(
            covar.get_shape().as_list()[:-1], minval=0, maxval=0.001 / self.scale**2
        )
        s, u, v = tf.linalg.svd(covar + tf.linalg.diag(noise))
        # test if the matrix is invertible
        invertible = self._is_invertible(s)
        # test if the matrix is positive definite
        pd = tf.reduce_all(tf.greater(s, 0), axis=-1)

        # try making a valid version of the covariance matrix by ensuring that
        # the minimum eigenvalue is at least 1e-4/self.scale
        min_eig = s[..., -1:]
        eps = tf.tile(
            tf.maximum(1e-4 / self.scale - min_eig, 0), [1, s.get_shape()[-1]]
        )
        covar_invertible = tf.matmul(
            u, tf.matmul(tf.linalg.diag(s + eps), v, adjoint_b=True)
        )

        # if the covariance matrix is valid, leave it as is, else replace with
        # the modified variant
        covar_valid = tf.where(
            tf.logical_and(invertible, pd)[:, None, None], covar, covar_invertible
        )

        # make symmetric again
        covar_valid = (covar_valid + tf.linalg.matrix_transpose(covar_valid)) / 2.0

        return covar_valid

    ###########################################################################


class getloss:
    def _mse(self, diff):
        """
        Returns the mean squared error of diff = label - pred plus their
        euclidean distance (dist)
        Parameters
        ----------
        diff : tensor
            difference between label and prediction
        reduce_mean : bool, optional
            if true, return the mean errors over the complete tensor. The
            default is False.
        Returns
        -------
        loss : tensor
            the mean squared error
        dist : tensor
            the euclidean distance
        """
        diff_a = tf.expand_dims(diff, axis=-1)
        diff_b = tf.expand_dims(diff, axis=-2)

        loss = tf.matmul(diff_b, diff_a)

        # the loss needs to be finite and positive
        loss = tf.where(tf.math.is_finite(loss), loss, tf.ones_like(loss) * 1e20)
        loss = tf.where(tf.greater_equal(loss, 0), loss, tf.ones_like(loss) * 1e20)

        loss = tf.squeeze(loss, axis=-1)
        dist = tf.sqrt(loss)

        loss = tf.reduce_mean(loss)
        dist = tf.reduce_mean(dist)

        loss = dist + loss

        return loss


# Xiao's version
class EKF(tf.keras.Model):
    def __init__(self, batch_size, **kwargs):
        super(EKF, self).__init__()

        # initialization
        self.batch_size = batch_size

        self.dim_x = 7
        self.dim_z = 7

        self.jacobian = True
        self.q_diag = np.ones((self.dim_x)) * 0.1
        self.q_diag = self.q_diag.astype(np.float32)

        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        # learned process model
        self.process_model = ProcessModel(self.batch_size, self.dim_x, jacobian=True)

        # learned process noise
        self.process_noise_model = ProcessNoise(
            self.batch_size, self.dim_x, self.q_diag
        )

        # learned observation model
        self.observation_model = ObservationModel(self.batch_size, self.dim_z)

        # learned observation noise
        self.observation_noise_model = ObservationNoise(
            self.batch_size, self.dim_z, self.r_diag
        )

        # learned sensor model
        self.sensor_model = SensorModel(self.batch_size, 32, self.dim_z)

        self.utils_ = utils()

    def call(self, inputs, states):
        # decompose inputs and states
        raw_sensor = inputs

        dim_obs = raw_sensor.shape[-1]
        raw_sensor = tf.reshape(raw_sensor, [self.batch_size, dim_obs])

        state_old, covar_old = states

        state_old = tf.reshape(state_old, [self.batch_size, 1, self.dim_x])
        covar_old = tf.reshape(covar_old, [self.batch_size, self.dim_x, self.dim_x])

        """
        prediction step
        state_pred: x_{t}
                 F: learnd Jacobian
                 Q: process noise
        covar_pred: p_{t}
        """
        # get prediction and noise of next state

        # use batch gradient
        state_p, F = self.process_model(state_old)
        F = tf.squeeze(F)

        state_pred = tf.reshape(state_p, [self.batch_size, 1, self.dim_x])
        state_old = tf.reshape(state_old, [self.batch_size, 1, self.dim_x])
        F = tf.reshape(F, [self.batch_size, self.dim_x, self.dim_x])

        update = state_pred - state_old
        d_state = tf.reshape(update, [self.batch_size, 1, self.dim_x])

        Q = self.process_noise_model(state_old)

        F_prime = tf.transpose(F, perm=[0, 2, 1])

        # pdb.set_trace()

        # calculate predicted covariance matrix
        covar_pred = tf.matmul(F, tf.matmul(covar_old, F_prime)) + Q

        """
        update step
        state_new: hat_x_{t}
        covar_new: hat_p_{t}
                H: observation Jacobians
                S: innovation matrix
                K: kalman gain

        """
        # get sensor reading
        z, encoding = self.sensor_model(raw_sensor)
        z = tf.reshape(z, [self.batch_size, 1, self.dim_z])

        # get observation noise
        R, _ = self.observation_noise_model(encoding)

        # get predicted observation and its jacobian H
        z_pred, H = self.observation_model(state_pred)

        # difference between sensor readings and predicted observations
        y = z - z_pred
        y = tf.linalg.matrix_transpose(y)

        H_prime = tf.transpose(H, perm=[0, 2, 1])
        # calculated innovation matrix s
        innovation = tf.matmul(H, tf.matmul(covar_pred, H_prime)) + R

        try:
            innovation_inv = tf.linalg.inv(innovation)
        except:
            innovation = self.utils_._make_valid(innovation)
            innovation_inv = tf.linalg.inv(innovation)

        # calculating Kalman gain
        K = tf.matmul(covar_pred, tf.matmul(H_prime, innovation_inv))

        # update state
        state_new = state_pred + tf.linalg.matrix_transpose(tf.matmul(K, y))

        # update covariance
        mult = tf.eye(self.dim_x) - tf.matmul(K, H)
        covar_up = tf.matmul(
            mult, tf.matmul(covar_pred, tf.linalg.matrix_transpose(mult))
        )
        covar_up = covar_up + tf.matmul(K, tf.matmul(R, tf.linalg.matrix_transpose(K)))

        covar_new = self.utils_._make_valid(covar_up)

        state_new = tf.reshape(state_new, [self.batch_size, 1, self.dim_x])
        covar_new = tf.reshape(covar_new, [self.batch_size, self.dim_x, self.dim_x])
        output = (state_new, covar_new, d_state)

        return output
