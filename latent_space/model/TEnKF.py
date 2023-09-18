from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from einops import rearrange, repeat
import numpy as np
import math
from .attention import PositionalEncoder
from .attention import TypeEncoder
from .attention import ResidualAttentionBlock
from .attention import KalmanAttentionBlock
from .utils import SensorModel, miniSensorModel, DecoderModel, miniDecoderModel
from .utils import ImgToLatentModel, miniImgToLatentModel, ImageRecover
import pdb

"""
this framework is built upon the differentiable Kalman Filters but the attention
and the transformer modules are used to provide not only 1st order markov properties
The process model is replaced by a transformer encoder, the Kalman update step is then
using the attention weights to represent the innovation matrix in Kalman update
"""


class transformer_process_model(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. the process model is flexiable, we can inject the known
    dynamics into it, we can also change the model architecture which takes sequential
    data as input

    input -> [batch_size, ensemble, timestep, dim_x]
    output ->  [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, win_size, dim_model, num_heads):
        super(transformer_process_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.win_size = win_size

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )
        self.attention_layer_1 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_2 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_3 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=32)
        self.bayes2 = LinearFlipout(in_features=32, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=256)
        self.bayes_m1 = torch.nn.Linear(256, 128)
        self.bayes_m2 = torch.nn.Linear(128, self.dim_x)

    def forward(self, input):
        batch_size = input.shape[0]
        input = rearrange(input, "n en k dim -> (n en) k dim")
        input = rearrange(input, "n k dim -> (n k) dim")

        # branch of the state
        x, _ = self.bayes1(input)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        x = rearrange(x, "(n k) dim -> n k dim", k=self.win_size)

        # for pos embedding layers
        x = rearrange(x, "n k dim -> k n dim")
        x = self.positional_encoding_layer(x)

        x, _ = self.attention_layer_1(x)
        x, _ = self.attention_layer_2(x)
        x, _ = self.attention_layer_3(x)

        x = rearrange(x, "k n dim -> n k dim", n=batch_size * self.num_ensemble)
        x = rearrange(x, "n k dim -> (n k) dim", n=batch_size * self.num_ensemble)
        x = self.bayes_m1(x)
        x = F.relu(x)
        x = self.bayes_m2(x)
        output = rearrange(x, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble)
        output = rearrange(output, "(n en) k dim -> n en k dim", en=self.num_ensemble)

        return output


class transformer_process_model_action(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. this process model takes in the state and actions
    and outputs a predicted state

    input -> [batch_size, num_ensemble, timestep, dim_x]
    action -> [batch_size, num_ensemble, dim_a]

    output ->  [batch_size, num_ensemble, timestep, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, dim_a, win_size, dim_model, num_heads):
        super(transformer_process_model_action, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a
        self.win_size = win_size

        self.type_encoder = TypeEncoder(
            dropout=0.1,
            win_size=win_size,
            type=3,
            d_model=dim_model,
            batch_first=True,
            length=win_size + 2,
        )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )

        self.attention_layer_1 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_2 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_3 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )

        # channel for state variables
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=256)

        # channel for action variables
        self.bayes_a1 = LinearFlipout(in_features=self.dim_a, out_features=32)
        self.bayes_a2 = LinearFlipout(in_features=32, out_features=128)
        self.bayes_a3 = LinearFlipout(in_features=128, out_features=256)

        # merge them
        self.bayes4 = LinearFlipout(in_features=256, out_features=128)
        self.bayes5 = LinearFlipout(in_features=128, out_features=self.dim_x)

    def forward(self, input, action):
        batch_size = input.shape[0]
        input = rearrange(input, "n en k dim -> (n en) k dim")
        input = rearrange(input, "n k dim -> (n k) dim")
        action = rearrange(action, "bs en dim -> (bs en) dim")

        # branch for the state variables
        x, _ = self.bayes1(input)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        x = rearrange(x, "(n k) dim -> n k dim", k=self.win_size)

        # branch for the action variables
        y, _ = self.bayes_a1(action)
        y = F.relu(y)
        y, _ = self.bayes_a2(y)
        y = F.relu(y)
        y, _ = self.bayes_a3(y)
        y = F.relu(y)
        y = rearrange(y, "(n k) dim -> n k dim", k=1)
        place_holder = torch.zeros_like(y)
        y = torch.cat((y, place_holder), axis=1)

        # for pos embedding layers
        x = rearrange(x, "n k dim -> k n dim")
        x = self.positional_encoding_layer(x)
        y = rearrange(y, "n k dim -> k n dim")
        merge = torch.cat((x, y), axis=0)
        merge = self.type_encoder(merge)

        # attention
        x, _ = self.attention_layer_1(merge)
        x, _ = self.attention_layer_2(x)
        x, _ = self.attention_layer_3(x)

        # post process branch
        x = rearrange(x, "k n dim -> n k dim", n=batch_size * self.num_ensemble)
        x = rearrange(x, "n k dim -> (n k) dim", n=batch_size * self.num_ensemble)
        x, _ = self.bayes4(x)
        x = F.relu(x)
        x, _ = self.bayes5(x)
        x = rearrange(x, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble)
        output = rearrange(x, "(n en) k dim -> n en k dim", en=self.num_ensemble)
        return output


class NewAttentionGain(nn.Module):
    def __init__(self, dim_x, dim_z, dim_model, num_heads):
        """
        attention gain module is used to replace the Kalman update step, this module
        takes a predicted state and a learned observation
        learn the measuremenmt update by calculating attention of the state w.r.t to
        the concatnated vector [state,observation], and update state accordingly

        input:
        y -> [batch_size, ensemble, dim_z]
        x -> [batch_size, ensemble, dim_x]


        output:
        atten -> [batch_size, dim_x, dim_y]
        """
        super(NewAttentionGain, self).__init__()
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.num_ensemble = dim_model

        p1 = torch.eye(dim_x)
        p2 = torch.eye(dim_z)
        fill = torch.zeros((dim_x - dim_z, dim_z))
        p2 = torch.cat((fill, p2), axis=0)
        attn_mask = torch.cat((p1, p2), axis=1)

        self.Q = nn.Parameter(torch.rand(self.dim_x, dim_model))

        tmp = torch.tensor(float("-inf"), dtype=torch.float32)
        zero = torch.tensor(0, dtype=torch.float32)
        self.attn_mask = torch.where(attn_mask < 1, tmp, zero)

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )
        self.attention_layer_1 = KalmanAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=self.attn_mask
        )

    def forward(self, state, obs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = state.shape[0]

        # zero means
        m_x = torch.mean(state, axis=1)
        m_x = repeat(m_x, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state - m_x

        m_y = torch.mean(obs, axis=1)
        m_y = repeat(m_y, "bs dim -> bs k dim", k=self.num_ensemble)
        Y = obs - m_y

        A = rearrange(A, "bs en dim -> dim bs en")
        Y = rearrange(Y, "bs en dim -> dim bs en")

        state = rearrange(state, "bs en dim -> dim bs en")
        obs = rearrange(obs, "bs en dim -> dim bs en")

        ################# -> trying new idea #################
        query = repeat(self.Q, "dim en -> dim bs en", bs=batch_size)
        query = self.positional_encoding_layer(query)

        xy = torch.cat((A, Y), axis=0)
        xy = self.positional_encoding_layer(xy)
        _, atten = self.attention_layer_1(query, xy)

        # use the attention to do a weighted sum
        state = rearrange(state, "dim bs en -> bs dim en")
        obs = rearrange(obs, "dim bs en -> bs dim en")

        merge = torch.cat((state, obs), axis=1)

        state = torch.matmul(atten, merge)
        ##############################################################

        state = rearrange(state, "bs dim en -> bs en dim")

        return state, atten


class latentAttentionGain(nn.Module):
    def __init__(self, full_mod, dim_x, dim_z, dim_model, num_heads):
        """
        attention gain module is used to replace the Kalman update step, this module
        takes a predicted state and a learned observation
        learn the measuremenmt update by calculating attention of the state w.r.t to
        the concatnated vector [state,observation], and update state accordingly

        input:
        obs -> [y1, y2, y3]
        modality_list -> [0, 1, 2]
        state -> x
        y -> [batch_size, ensemble, dim_z]
        x -> [batch_size, ensemble, dim_x]


        output:
        atten -> [batch_size, dim_x, dim_y]
        """
        super(latentAttentionGain, self).__init__()
        self.full_mod = full_mod
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.num_ensemble = dim_model

        self.Q = nn.Parameter(torch.rand(self.dim_x, dim_model))

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )
        self.attention_layer_1 = KalmanAttentionBlock(
            d_model=dim_model, n_head=num_heads
        )

    def generate_attn_mask(self, dim_x, dim_z, full_mod, mod_list):
        tmp = torch.tensor(float("-inf"), dtype=torch.float32)
        zero = torch.tensor(0, dtype=torch.float32)
        attn_mask = torch.eye(dim_x)
        for modality in full_mod:
            if modality in mod_list:
                p2 = torch.eye(dim_z)
            else:
                p2 = torch.zeros((dim_z, dim_z))
            fill = torch.zeros((dim_x - dim_z, dim_z))
            p2 = torch.cat((fill, p2), axis=0)
            attn_mask = torch.cat((attn_mask, p2), axis=1)
        attn_mask = torch.where(attn_mask < 1, tmp, zero)
        return attn_mask

    def forward(self, state, obs_list, mod_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = state.shape[0]

        # generate attn_mask from

        attn_mask = self.generate_attn_mask(
            self.dim_x, self.dim_z, self.full_mod, mod_list
        )

        # zero means
        m_x = torch.mean(state, axis=1)
        m_x = repeat(m_x, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state - m_x

        # collect latent obs if one modality is missing, then use 0
        Y_list = []
        Y_ = []
        idx = 0
        for modality in self.full_mod:
            if modality in mod_list:
                obs = obs_list[idx]
                m_y = torch.mean(obs, axis=1)
                m_y = repeat(m_y, "bs dim -> bs k dim", k=self.num_ensemble)
                Y = obs - m_y
                Y = rearrange(Y, "bs en dim -> dim bs en")
                Y_list.append(Y.to(device))
                Y_.append(obs)
                idx = idx + 1
            else:
                Y = torch.rand(self.dim_z, batch_size, self.num_ensemble) * 0.0
                Y_list.append(Y.to(device))
                tmp = torch.rand(batch_size, self.num_ensemble, self.dim_z) * 0.0
                Y_.append(tmp.to(device))

        # define Q
        query = repeat(self.Q, "dim en -> dim bs en", bs=batch_size)
        query = self.positional_encoding_layer(query)

        # define K
        A = rearrange(A, "bs en dim -> dim bs en")
        xy = A
        for Y in Y_list:
            xy = torch.cat((xy, Y), axis=0)
        xy = self.positional_encoding_layer(xy)
        _, atten = self.attention_layer_1(query, xy, attn_mask)

        # define V
        merge = state
        for obs in Y_:
            merge = torch.cat((merge, obs), axis=2)
        merge = rearrange(merge, "bs en dim -> bs dim en")

        # use the attention to do a weighted sum
        state = torch.matmul(atten, merge)
        state = rearrange(state, "bs dim en -> bs en dim")

        return state, atten


####################################################################
############### put all model together (don't touch this ) #########


class Test_latent_enKF(nn.Module):
    def __init__(self, num_ensemble, win_size, dim_x, dim_z, dim_a, input_size):
        super(Test_latent_enKF, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.input_size = input_size
        self.win_size = win_size

        # instantiate model
        self.process_model = transformer_process_model(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.img_2_latent = ImgToLatentModel(
            self.num_ensemble, self.win_size, self.dim_x
        )
        self.sensor_model = SensorModel(self.num_ensemble, self.input_size, self.dim_z)
        self.attention_update = NewAttentionGain(
            self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.latent_2_img = ImageRecover(self.num_ensemble, self.dim_x)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        raw_obs = inputs
        images = states

        ##### prediction step #####
        # img -> latent
        latent = self.img_2_latent(images)

        # process model on latent state
        state_pred = self.process_model(latent)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####

        # get latent observation from raw sensor
        ensemble_z, z, encoding = self.sensor_model(raw_obs)

        # measurement update
        state_new, atten = self.attention_update(state_pred, ensemble_z)
        m_state_new = torch.mean(state_new, axis=1)

        # recover image
        out_image = self.latent_2_img(m_state_new)

        # gather output
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            out_image.to(dtype=torch.float32),
            m_state_new.to(dtype=torch.float32),
            m_state_pred.to(dtype=torch.float32),
            z.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
        )
        return output


####################################################################
class AuxiliaryStateModel(nn.Module):
    """
    input -> [batch_size, timestep, dim_gt]
    output ->  [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(self, num_ensemble, win_size, dim_gt, dim_x):
        super(AuxiliaryStateModel, self).__init__()
        self.win_size = win_size
        self.dim_x = dim_x
        self.dim_gt = dim_gt
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(self.dim_gt, 128)
        self.fc3 = LinearFlipout(128, 256)
        self.fc4 = LinearFlipout(256, 512)
        self.fc5 = LinearFlipout(512, 1024)
        self.fc6 = LinearFlipout(1024, self.dim_x)

    def forward(self, input):
        batch_size = input.shape[0]
        input = repeat(input, "bs k dim -> bs en k dim", en=self.num_ensemble)
        input = rearrange(input, "n en k dim -> (n en) k dim")
        input = rearrange(input, "n k dim -> (n k) dim")

        x = self.fc2(input)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)
        x = F.leaky_relu(x)
        x, _ = self.fc5(x)
        x = F.leaky_relu(x)
        obs, _ = self.fc6(x)

        output = rearrange(
            obs, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble
        )
        output = rearrange(output, "(n en) k dim -> n en k dim", en=self.num_ensemble)

        latent_state = output
        return latent_state


class miniAuxiliaryStateModel(nn.Module):
    """
    input -> [batch_size, timestep, dim_gt]
    output ->  [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(self, num_ensemble, win_size, dim_gt, dim_x):
        super(miniAuxiliaryStateModel, self).__init__()
        self.win_size = win_size
        self.dim_x = dim_x
        self.dim_gt = dim_gt
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(self.dim_gt, 64)
        self.fc3 = LinearFlipout(64, 128)
        self.fc4 = LinearFlipout(128, self.dim_x)

    def forward(self, input):
        batch_size = input.shape[0]
        input = repeat(input, "bs k dim -> bs en k dim", en=self.num_ensemble)
        input = rearrange(input, "n en k dim -> (n en) k dim")
        input = rearrange(input, "n k dim -> (n k) dim")

        x = self.fc2(input)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)

        output = rearrange(x, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble)
        output = rearrange(output, "(n en) k dim -> n en k dim", en=self.num_ensemble)

        latent_state = output
        return latent_state

##############################################################
class UR5_latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1, img2, sensor1, sensor2]
    modality_list = [0, 2, 3]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

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
    ):
        super(UR5_latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))

        # instantiate model
        self.process_model = transformer_process_model(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_1),
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_2),
                SensorModel(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )
        self.attention_update = latentAttentionGain(
            self.full_modality, self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.decoder = DecoderModel(self.num_ensemble, self.dim_x, self.dim_gt)

    def forward(self, inputs, states):
        # decompose inputs and states
        list_inputs, mod_list = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for idx in mod_list:
            input_idx = mod_list.index(idx)
            output = self.encoder_models[idx](list_inputs[input_idx])
            encoded_feat.append(output)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####
        # measurement update

        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        output = (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

        return output


class UR5_push_latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1, img2, sensor1, sensor2]
    modality_list = [0, 2, 3]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

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
        super(UR5_push_latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))

        # instantiate model
        self.process_model = transformer_process_model(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_1),
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_2),
                SensorModel(self.num_ensemble, input_size_1, self.dim_z),
                SensorModel(self.num_ensemble, input_size_2, self.dim_z),
            ]
        )
        self.attention_update = latentAttentionGain(
            self.full_modality, self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.decoder = DecoderModel(self.num_ensemble, self.dim_x, self.dim_gt)

    def forward(self, inputs, states):
        # decompose inputs and states
        list_inputs, mod_list = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for idx in mod_list:
            input_idx = mod_list.index(idx)
            output = self.encoder_models[idx](list_inputs[input_idx])
            encoded_feat.append(output)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####
        # measurement update

        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        output = (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

        return output


##############################################################
class KITTI_latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1]
    modality_list = [0]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

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
    ):
        super(KITTI_latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))

        # instantiate model
        self.process_model = transformer_process_model(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_1),
            ]
        )
        self.attention_update = latentAttentionGain(
            self.full_modality, self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.decoder = DecoderModel(self.num_ensemble, self.dim_x, self.dim_gt)

    def forward(self, inputs, states):
        # decompose inputs and states
        list_inputs, mod_list = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for idx in mod_list:
            input_idx = mod_list.index(idx)
            output = self.encoder_models[idx](list_inputs[input_idx])
            encoded_feat.append(output)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####
        # measurement update

        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        output = (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

        return output


##############################################################
class UR5real_latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1]
    modality_list = [0]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

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
    ):
        super(UR5real_latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))

        # instantiate model
        self.process_model = transformer_process_model(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_1),
                SensorModel(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )
        self.attention_update = latentAttentionGain(
            self.full_modality, self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.decoder = DecoderModel(self.num_ensemble, self.dim_x, self.dim_gt)

    def forward(self, inputs, states):
        # decompose inputs and states
        list_inputs, mod_list = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for idx in mod_list:
            input_idx = mod_list.index(idx)
            output = self.encoder_models[idx](list_inputs[input_idx])
            encoded_feat.append(output)

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####
        # measurement update
        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        output = (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

        return output


class Soft_robot_latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1]
    modality_list = [0]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

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
    ):
        super(Soft_robot_latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))

        # instantiate model
        self.process_model = transformer_process_model_action(
            self.num_ensemble, self.dim_x, self.dim_a, self.win_size, 256, 8
        )
        self.encoder_models = torch.nn.ModuleList(
            [
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_1),
                ImgToLatentModel(self.num_ensemble, self.dim_x, channel_img_2),
                SensorModel(self.num_ensemble, input_size_1, self.dim_z),
            ]
        )
        self.attention_update = latentAttentionGain(
            self.full_modality, self.dim_x, self.dim_z, self.num_ensemble, 4
        )
        self.decoder = DecoderModel(self.num_ensemble, self.dim_x, self.dim_gt)

    def forward(self, inputs, states):
        # decompose inputs and states
        list_inputs, mod_list, action = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for idx in mod_list:
            input_idx = mod_list.index(idx)
            output = self.encoder_models[idx](list_inputs[input_idx])
            encoded_feat.append(output)

        ##### prediction step #####
        state_pred = self.process_model(state_old, action)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        ##### update step #####
        # measurement update
        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        output = (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

        return output
