from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision.models as models
from einops import rearrange, repeat
import numpy as np
import math
from .utils import SensorModel, miniSensorModel, DecoderModel, miniDecoderModel
from .utils import ImgToLatentModel, miniImgToLatentModel, ImageRecover
from .utils import ImgToLatentModel_Baseline, SensorModel_Baseline
import pickle


class ProcessModel(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. the process model is flexiable, we can inject the known
    dynamics into it, we can also change the model architecture which takes sequential
    data as input

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_x]
    """

    def __init__(self, num_ensemble, dim_x):
        super(ProcessModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x

        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=512)
        self.bayes3 = LinearFlipout(in_features=512, out_features=256)
        self.bayes4 = LinearFlipout(in_features=256, out_features=self.dim_x)

    def forward(self, last_state):
        batch_size = last_state.shape[0]
        last_state = rearrange(
            last_state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        update, _ = self.bayes4(x)
        state = last_state + update
        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return state


class ProcessModelAction(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. this process model takes in the state and actions
    and outputs a predicted state

    input -> [batch_size, num_ensemble, dim_x]
    action -> [batch_size, num_ensemble, dim_a]
    output ->  [batch_size, num_ensemble, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, dim_a):
        super(ProcessModelAction, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a

        # channel for state variables
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=64)

        # channel for action variables
        self.bayes_a1 = LinearFlipout(in_features=self.dim_a, out_features=64)
        self.bayes_a2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes_a3 = LinearFlipout(in_features=128, out_features=64)

        # merge them
        self.bayes4 = LinearFlipout(in_features=128, out_features=64)
        self.bayes5 = LinearFlipout(in_features=64, out_features=self.dim_x)

    def forward(self, last_state, action):
        batch_size = last_state.shape[0]
        last_state = rearrange(
            last_state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        action = rearrange(
            action, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )

        # branch for the state variables
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)

        # branch for the action variables
        y, _ = self.bayes_a1(action)
        y = F.relu(y)
        y, _ = self.bayes_a2(y)
        y = F.relu(y)
        y, _ = self.bayes_a3(y)
        y = F.relu(y)

        # merge branch
        merge = torch.cat((x, y), axis=1)
        merge, _ = self.bayes4(merge)
        update, _ = self.bayes5(merge)
        state = last_state + update
        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return state


class ObservationModel(nn.Module):
    """
    observation model takes a predicted state at t-1 and
    predict the corresponding oberservations. typically, the observation is part of the
    state (H as an identity matrix), unless we are using some observations indirectly to
    update the state

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_x, dim_z):
        super(ObservationModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.linear1 = torch.nn.Linear(self.dim_x, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, self.dim_z)

    def forward(self, state):
        batch_size = state.shape[0]
        state = rearrange(
            state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        z_pred = self.linear5(x)
        z_pred = rearrange(
            z_pred, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return z_pred


class ObservationNoise(nn.Module):
    def __init__(self, dim_z, r_diag):
        """
        observation noise model is used to learn the observation noise covariance matrix
        R from the learned observation, kalman filter require a explicit matrix for R
        therefore we construct the diag of R to model the noise here

        input -> [batch_size, 1, encoding/dim_z]
        output -> [batch_size, dim_z, dim_z]
        """
        super(ObservationNoise, self).__init__()
        self.dim_z = dim_z
        self.r_diag = r_diag

        self.fc1 = nn.Linear(self.dim_z, 32)
        self.fc2 = nn.Linear(32, self.dim_z)

    def forward(self, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = inputs.shape[0]
        constant = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        diag = self.fc1(inputs)
        diag = F.relu(diag)
        diag = self.fc2(diag)
        diag = torch.square(diag + torch.Tensor(constant).to(device)) + torch.Tensor(
            init
        ).to(device)
        diag = rearrange(diag, "bs k dim -> (bs k) dim")
        R = torch.diag_embed(diag)
        return R


class Naive_Fusion(nn.Module):
    """
    input-1 -> [batch_size, num_ensemble, dim_z]
    input-2 -> [batch_size, num_ensemble, dim_z]

    output -> [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_z):
        super(Naive_Fusion, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_z = dim_z

        self.fc1 = nn.Linear(32 * 2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.dim_z)

    def forward(self, inputs):
        m1, m2 = inputs
        batch_size = m1.shape[0]
        m1 = repeat(m1, "bs k dim -> n bs k dim", bs=batch_size, n=self.num_ensemble)
        m2 = repeat(m2, "bs k dim -> n bs k dim", bs=batch_size, n=self.num_ensemble)
        m1 = rearrange(
            m1, "n bs k dim -> bs (n k) dim", bs=batch_size, n=self.num_ensemble
        )
        m2 = rearrange(
            m2, "n bs k dim -> bs (n k) dim", bs=batch_size, n=self.num_ensemble
        )
        m1 = rearrange(m1, "bs k dim -> (bs k) dim")
        m2 = rearrange(m2, "bs k dim -> (bs k) dim")
        merge = torch.cat((m1, m2), axis=1)
        x = self.fc1(merge)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = rearrange(x, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble)
        return x


class Crossmodal_Fusion(nn.Module):
    """
    input-1 -> [batch_size, num_ensemble, dim_z]
    input-2 -> [batch_size, num_ensemble, dim_z]

    output -> [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_z):
        super(Crossmodal_Fusion, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_z = dim_z

        self.fc1 = nn.Linear(32 * 2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.dim_z)

    def forward(self, inputs):
        m1, m2 = inputs
        batch_size = m1.shape[0]
        m1 = repeat(m1, "bs k dim -> n bs k dim", bs=batch_size, n=self.num_ensemble)
        m2 = repeat(m2, "bs k dim -> n bs k dim", bs=batch_size, n=self.num_ensemble)
        m1 = rearrange(
            m1, "n bs k dim -> bs (n k) dim", bs=batch_size, n=self.num_ensemble
        )
        m2 = rearrange(
            m2, "n bs k dim -> bs (n k) dim", bs=batch_size, n=self.num_ensemble
        )
        m1 = rearrange(m1, "bs k dim -> (bs k) dim")
        m2 = rearrange(m2, "bs k dim -> (bs k) dim")
        merge = torch.cat((m1, m2), axis=1)
        x = self.fc1(merge)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        w = F.softmax(x, dim=1)
        w = rearrange(w, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble)
        w = torch.mean(w, axis=1)
        w = rearrange(w, "bs (k dim) -> bs k dim", k=1)
        return w


class Ensemble_KF_Naive_Fusion(nn.Module):
    def __init__(
        self,
        num_ensemble,
        win_size,
        dim_x,
        dim_z,
        dim_a,
        dim_gt,
        sensor_len,
        channel_img_1,
        channel_img_2,
        input_size_1,
        input_size_2,
    ):
        super(Ensemble_KF_Naive_Fusion, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel_Baseline(self.num_ensemble, self.dim_x, channel_img_1),
                SensorModel_Baseline(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )
        self.sensor_fusion = Naive_Fusion(self.num_ensemble, self.dim_z)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        obs_img, obs_1 = inputs
        state_old, m_state = states

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        encoded_feat = []
        ensemble_m1, encoding_1 = self.encoder_models[0](obs_img)
        ensemble_m2, encoding_2 = self.encoder_models[1](obs_1)
        to_fuse = (encoding_1, encoding_2)

        # get ensemble mean for modalities
        m1 = torch.mean(ensemble_m1, axis=1)
        m1 = rearrange(m1, "bs (k dim) -> bs k dim", k=1)
        m2 = torch.mean(ensemble_m2, axis=1)
        m2 = rearrange(m2, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(m1.to(dtype=torch.float32))
        encoded_feat.append(m2.to(dtype=torch.float32))

        # sensor fusion
        ensemble_z = self.sensor_fusion(to_fuse)
        z = torch.mean(ensemble_z, axis=1)
        z = rearrange(z, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(z.to(dtype=torch.float32))
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")

        R = self.observation_noise(z)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),  # -> ensemble
            m_state_new.to(dtype=torch.float32),  # -> x
            m_state_pred.to(dtype=torch.float32),  # -> x_hat
            encoded_feat,
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),  # -> obs
        )
        return output


class Ensemble_KF_Unimodal_Fusion(nn.Module):
    def __init__(
        self,
        num_ensemble,
        win_size,
        dim_x,
        dim_z,
        dim_a,
        dim_gt,
        sensor_len,
        channel_img_1,
        channel_img_2,
        input_size_1,
        input_size_2,
    ):
        super(Ensemble_KF_Unimodal_Fusion, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel_Baseline(self.num_ensemble, self.dim_x, channel_img_1),
                SensorModel_Baseline(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def normal_distribution(self):
        cov = torch.eye(self.dim_z)
        init_ensemble = self.multivariate_normal_sampler(
            torch.zeros(self.dim_z), cov, self.num_ensemble
        )
        return init_ensemble

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        obs_img, obs_1 = inputs
        state_old, m_state = states

        init_ensemble = pickle.load(open("./model/init_ensemble_push.pkl", "rb")).to(
            self.device
        )
        init_ensemble = repeat(init_ensemble, "en dim -> bs en dim", bs=batch_size)
        init_ensemble = init_ensemble.to(self.device)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        encoded_feat = []
        ensemble_m1, encoding_1 = self.encoder_models[0](obs_img)
        ensemble_m2, encoding_2 = self.encoder_models[1](obs_1)
        to_fuse = (encoding_1, encoding_2)

        # get ensemble mean for modalities
        m1 = torch.mean(ensemble_m1, axis=1)
        m1 = rearrange(m1, "bs (k dim) -> bs k dim", k=1)
        m2 = torch.mean(ensemble_m2, axis=1)
        m2 = rearrange(m2, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(m1.to(dtype=torch.float32))
        encoded_feat.append(m2.to(dtype=torch.float32))

        # sensor fusion (unimodal)
        ensemble_m1 = rearrange(ensemble_m1, "bs en dim -> bs dim en")
        ensemble_m2 = rearrange(ensemble_m2, "bs en dim -> bs dim en")
        cov_1 = self.cov(ensemble_m1)
        cov_2 = self.cov(ensemble_m2)
        cov_1_ = torch.linalg.inv(cov_1)
        cov_2_ = torch.linalg.inv(cov_2)
        cov_fuse = cov_1_ + cov_2_
        cov_fuse_ = torch.linalg.inv(cov_fuse)
        # use reparameterization trick
        ensemble_z = torch.matmul(init_ensemble, cov_fuse_)

        z = torch.mean(ensemble_z, axis=1)
        z = rearrange(z, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(z.to(dtype=torch.float32))
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")

        R = self.observation_noise(z)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),  # -> ensemble
            m_state_new.to(dtype=torch.float32),  # -> x
            m_state_pred.to(dtype=torch.float32),  # -> x_hat
            encoded_feat,
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),  # -> obs
        )
        return output


class Ensemble_KF_crossmodal_Fusion(nn.Module):
    def __init__(
        self,
        num_ensemble,
        win_size,
        dim_x,
        dim_z,
        dim_a,
        dim_gt,
        sensor_len,
        channel_img_1,
        channel_img_2,
        input_size_1,
        input_size_2,
    ):
        super(Ensemble_KF_crossmodal_Fusion, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel_Baseline(self.num_ensemble, self.dim_x, channel_img_1),
                SensorModel_Baseline(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )
        self.sensor_fusion = Crossmodal_Fusion(self.num_ensemble, self.dim_z)

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def normal_distribution(self):
        cov = torch.eye(self.dim_z)
        init_ensemble = self.multivariate_normal_sampler(
            torch.zeros(self.dim_z), cov, self.num_ensemble
        )
        return init_ensemble

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        obs_img, obs_1 = inputs
        state_old, m_state = states

        init_ensemble = pickle.load(open("./model/init_ensemble_push.pkl", "rb")).to(
            self.device
        )
        init_ensemble = repeat(init_ensemble, "en dim -> bs en dim", bs=batch_size)
        init_ensemble = init_ensemble.to(self.device)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        encoded_feat = []
        ensemble_m1, encoding_1 = self.encoder_models[0](obs_img)
        ensemble_m2, encoding_2 = self.encoder_models[1](obs_1)
        to_fuse = (encoding_1, encoding_2)

        # get ensemble mean for modalities
        m1 = torch.mean(ensemble_m1, axis=1)
        m1 = rearrange(m1, "bs (k dim) -> bs k dim", k=1)
        m2 = torch.mean(ensemble_m2, axis=1)
        m2 = rearrange(m2, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(m1.to(dtype=torch.float32))
        encoded_feat.append(m2.to(dtype=torch.float32))

        # sensor fusion (crossmodal)
        w = self.sensor_fusion(to_fuse)
        w_T = rearrange(w, "bs k dim -> bs dim k")
        B_1 = torch.matmul(w_T, w)
        B_2 = torch.ones_like(B_1) - B_1

        ensemble_m1 = rearrange(ensemble_m1, "bs en dim -> bs dim en")
        ensemble_m2 = rearrange(ensemble_m2, "bs en dim -> bs dim en")
        cov_1 = self.cov(ensemble_m1)
        cov_2 = self.cov(ensemble_m2)
        cov_fuse_ = torch.mul(B_1, cov_1) + torch.mul(B_2, cov_2)
        # use reparameterization trick
        ensemble_z = torch.matmul(init_ensemble, cov_fuse_)

        z = torch.mean(ensemble_z, axis=1)
        z = rearrange(z, "bs (k dim) -> bs k dim", k=1)
        encoded_feat.append(z.to(dtype=torch.float32))
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")

        R = self.observation_noise(z)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),  # -> ensemble
            m_state_new.to(dtype=torch.float32),  # -> x
            m_state_pred.to(dtype=torch.float32),  # -> x_hat
            encoded_feat,
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),  # -> obs
        )
        return output
