import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from dataset import UR5_sim_dataloader, UR5_real_dataloader, UR5_push_dataloader
from dataset import soft_robot_dataloader
from model import (
    Ensemble_KF_Naive_Fusion,
    Ensemble_KF_Unimodal_Fusion,
    Ensemble_KF_crossmodal_Fusion,
)
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import random
import pickle


class Engine:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.train.batch_size
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.dim_gt = self.args.train.dim_gt
        self.sensor_len = self.args.train.sensor_len
        self.channel_img_1 = self.args.train.channel_img_1
        self.channel_img_2 = self.args.train.channel_img_2
        self.input_size_1 = self.args.train.input_size_1
        self.input_size_2 = self.args.train.input_size_2
        self.input_size_3 = self.args.train.input_size_3
        self.num_ensemble = self.args.train.num_ensemble
        self.win_size = self.args.train.win_size
        self.global_step = 0
        self.mode = self.args.mode.mode
        self.dataset_name = self.args.train.dataset
        if self.dataset_name == "UR5_sim":
            self.dataset = UR5_sim_dataloader(self.args, self.mode)
        if self.dataset_name == "UR5_real":
            self.dataset = UR5_real_dataloader(self.args, self.mode)
        if self.dataset_name == "UR5_push":
            self.dataset = UR5_push_dataloader(self.args, self.mode)
        if self.dataset_name == "soft_robot":
            self.dataset = soft_robot_dataloader(self.args, self.mode)
        self.model = Ensemble_KF_Naive_Fusion(
            self.num_ensemble,
            self.win_size,
            self.dim_x,
            self.dim_z,
            self.dim_a,
            self.dim_gt,
            self.sensor_len,
            self.channel_img_1,
            self.channel_img_2,
            self.input_size_1,
            self.input_size_2,
        )

        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()

    def test(self):
        string = "[================ testing ================]"
        self.logger.info(string)
        if self.dataset_name == "UR5_sim":
            test_dataset = UR5_sim_dataloader(self.args, "test")
        if self.dataset_name == "UR5_real":
            test_dataset = UR5_real_dataloader(self.args, "test")
        if self.dataset_name == "UR5_push":
            test_dataset = UR5_push_dataloader(self.args, "test")
        if self.dataset_name == "soft_robot":
            test_dataset = soft_robot_dataloader(self.args, "test")

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8
        )
        step = 0
        data = {}
        data_save = []
        gt_save = []
        obs = []
        for data_1, data_2 in test_dataloader:
            data_1 = [item.to(self.device) for item in data_1]
            data_2 = [item.to(self.device) for item in data_2]

            pre, gt, list_inputs, mod_list = self.get_data(data_1)

            with torch.no_grad():
                inputs = list_inputs
                if step == 0:
                    states = (self.format_state(pre), pre)
                else:
                    states = (output[0], output[1])
                output = self.model(inputs, states)
                pred = output[1]  # -> final estimation
                final_est = pred
                obs_ = output[3][-1]

                final_est = final_est.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                obs_ = obs_.cpu().detach().numpy()

                data_save.append(final_est)
                gt_save.append(gt)
                obs.append(obs_)
                step = step + 1

        data["state"] = data_save
        data["gt"] = gt_save
        data["obs"] = obs

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "vB-soft-result-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def loss_calculation(self, output, gt):
        # loss
        loss_1 = self.criterion(output[1], gt)
        loss_2 = self.criterion(output[2], gt)
        loss_3 = self.criterion(output[-1], gt)
        add_loss = self.criterion(output[3][-1], gt)
        return loss_1, loss_2, loss_3, add_loss

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def format_state(self, state):
        batch_size = state.shape[0]
        state = rearrange(state, "n k dim -> (n k) dim")
        state = repeat(state, "k dim -> k n dim", n=self.num_ensemble)

        cov = torch.eye(self.dim_x) * 0.05
        init_dist = self.multivariate_normal_sampler(
            torch.zeros(self.dim_x), cov, batch_size * self.num_ensemble
        )
        init_dist = rearrange(init_dist, "(n k) dim -> n k dim", k=self.num_ensemble)
        state = state + init_dist.to(self.device)
        state = state.to(dtype=torch.float32)
        return state

    def get_data(self, data):
        if self.dataset_name == "UR5_sim":
            gt_input, _, _, _, gt = data
            list_inputs = [data[1], data[3]]
            mod_list = [0, 2]
        if self.dataset_name == "UR5_real":
            gt_input, _, _, gt = data
            list_inputs = [data[1], data[2]]
            mod_list = [0, 1]
        if self.dataset_name == "UR5_push":
            gt_input, _, _, _, _, gt = data
            list_inputs = [data[1], data[4]]
            mod_list = [0, 3]
        if self.dataset_name == "soft_robot":
            gt_input, action, obs_1, rgb, depth, gt = data
            list_inputs = [rgb, obs_1]
            mod_list = [0, 1]

        pre = rearrange(gt_input[:, -1, :], "(bs k) dim -> bs k dim", k=1)
        return pre, gt, list_inputs, mod_list

    def train(self):

        self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
        criterion_support = nn.MSELoss()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Total number of parameters: ", pytorch_total_params)

        # Create optimizer
        optimizer_ = build_optimizer(
            [
                self.model
            ],
            self.args.network.name,
            self.args.optim.optim,
            self.args.train.learning_rate,
            self.args.train.weight_decay,
            self.args.train.adam_eps,
        )
        # Create LR scheduler
        if self.args.mode.mode == "train":
            num_total_steps = self.args.train.num_epochs * len(dataloader)
            scheduler = build_lr_scheduler(
                optimizer_,
                self.args.optim.lr_scheduler,
                self.args.train.learning_rate,
                num_total_steps,
                self.args.train.end_learning_rate,
            )
        # Epoch calculations
        steps_per_epoch = len(dataloader)
        num_total_steps = self.args.train.num_epochs * steps_per_epoch
        epoch = self.global_step // steps_per_epoch
        duration = 0

        # tensorboard writer
        self.writer = SummaryWriter(
            f"./experiments/{self.args.train.model_name}/summaries"
        )

        ####################################################################################################
        # MAIN TRAINING LOOP
        ####################################################################################################

        while epoch < self.args.train.num_epochs:
            step = 0
            for data_1, data_2 in dataloader:
                data_1 = [item.to(self.device) for item in data_1]
                data_2 = [item.to(self.device) for item in data_2]

                # optimizer
                optimizer_.zero_grad()
                before_op_time = time.time()

                ################## the learning curriculum ######################
                pre, gt, list_inputs, mod_list = self.get_data(data_1)

                # forward pass filter
                inputs = list_inputs
                states = (self.format_state(pre), pre)
                output = self.model(inputs, states)
                out_1 = output[1]

                # loss
                loss_1, loss_2, loss_3, add_loss = self.loss_calculation(output, gt)
                #################################################################
                if epoch >= 50:
                    final_loss = loss_1 + loss_2 + loss_3 + add_loss
                else:
                    final_loss = add_loss

                # back prop
                final_loss.backward()
                optimizer_.step()
                current_lr = optimizer_.param_groups[0]["lr"]
                ###############################################

                # verbose
                if self.global_step % self.args.train.log_freq == 0:
                    string = "[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], loss1: {:.12f}, loss2: {:.12f}, loss3: {:.12f}"
                    self.logger.info(
                        string.format(
                            epoch,
                            step,
                            steps_per_epoch,
                            self.global_step,
                            # current_lr,
                            loss_1,
                            loss_2,
                            final_loss,
                        )
                    )
                    if np.isnan(final_loss.cpu().item()):
                        self.logger.warning("NaN in loss occurred. Aborting training.")
                        return -1

                # tensorboard
                duration += time.time() - before_op_time
                if (
                    self.global_step
                    and self.global_step % self.args.train.log_freq == 0
                ):
                    self.writer.add_scalar(
                        "end_to_end_loss", final_loss.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "output", loss_1.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "transition", loss_2.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "observation", loss_3.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "sensors", add_loss.cpu().item(), self.global_step
                    )
                step += 1
                self.global_step += 1
                if scheduler is not None:
                    scheduler.step(self.global_step)

            # Save a model based of a chosen save frequency
            if self.global_step != 0 and (epoch + 1) % self.args.train.save_freq == 0:
                checkpoint = {
                    "global_step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "vB-soft-model-{}".format(self.global_step),
                    ),
                )

            # online evaluation
            if (
                self.args.mode.do_online_eval
                and self.global_step != 0
                # and epoch >= 20
                and (epoch + 1) % self.args.train.eval_freq == 0
            ):
                time.sleep(0.1)
                self.model.eval()
                self.test()
                self.model.train()

            # Update epoch
            epoch += 1

    def online_test(self):
        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
        else:
            checkpoint_1 = torch.load(
                self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
        self.model.eval()

        if self.dataset_name == "UR5_sim":
            test_dataset = UR5_sim_dataloader(self.args, "test")
        if self.dataset_name == "UR5_real":
            test_dataset = UR5_real_dataloader(self.args, "test")
        if self.dataset_name == "UR5_push":
            test_dataset = UR5_push_dataloader(self.args, "test")
        if self.dataset_name == "soft_robot":
            test_dataset = soft_robot_dataloader(self.args, "test")

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        step = 0
        data = {}
        data_save = []
        gt_save = []
        ensemble_save = []
        modality = []
        atten = []
        for data_1, data_2 in test_dataloader:
            data_1 = [item.to(self.device) for item in data_1]
            data_2 = [item.to(self.device) for item in data_2]

            pre, gt, list_inputs, mod_list = self.get_data(data_1)

            with torch.no_grad():
                inputs = list_inputs
                if step == 0:
                    states = (self.format_state(pre), pre)
                else:
                    states = (output[0], output[1])
                output = self.model(inputs, states)
                pred = output[1]  # -> final estimation
                final_est = pred

                final_est = output[3][2]
                final_est = final_est.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()

                data_save.append(final_est)
                gt_save.append(gt)
                modality.append(mod_list)
                step = step + 1

        data["state"] = data_save
        data["gt"] = gt_save
        data["modality"] = modality

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "vB-soft-test-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)
