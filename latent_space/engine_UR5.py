import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from dataset import UR5_sim_dataloader
from model import UR5_latent_model
from model import AuxiliaryStateModel
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
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
        self.dataset = UR5_sim_dataloader(self.args, self.mode)
        self.model = UR5_latent_model(
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
        )
        self.support_model = AuxiliaryStateModel(
            self.num_ensemble, self.win_size, self.dim_gt, self.dim_x
        )
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()
            self.support_model.cuda()

    def test(self):
        string = "[================ testing ================]"
        self.logger.info(string)
        test_dataset = UR5_sim_dataloader(self.args, "test")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        step = 0
        data = {}
        data_save = []
        gt_save = []
        modality = []
        for data_1, data_2 in test_dataloader:
            data_1 = [item.to(self.device) for item in data_1]
            data_2 = [item.to(self.device) for item in data_2]

            gt_input, _, _, _, gt = data_1
            list_inputs, mod_list = self.modality_selection(data_1)
            enforce = False
            if enforce == True:
                # enforce selected modalities
                list_inputs = [data_1[1], data_1[3]]
                mod_list = [0, 2]

            with torch.no_grad():
                if step == 0:
                    latent_state = self.support_model(gt_input)
                else:
                    latent_input = output[-1]  # -> previous latent state
                    latent_input = rearrange(
                        latent_input, "bs (en k) dim -> bs en k dim", k=1
                    )
                    latent_state = torch.cat(
                        (latent_state[:, :, 1:, :], latent_input), axis=2
                    )

                inputs = (list_inputs, mod_list)
                output = self.model(inputs, latent_state)

                pred = output[1]  # -> final estimation
                final_est = pred

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
            "v-UR5-result-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def modality_selection(self, data):
        _, gt_image, obs_1, obs_2, _ = data
        full_modality = list(range(0, self.sensor_len))
        obs = [gt_image, obs_1, obs_2]

        p = random.uniform(0, 1)
        if p < 0.4:
            mod_list = full_modality
            list_inputs = obs
        else:
            mod_list = sorted(random.sample(full_modality, 2))
            list_inputs = [obs[mod_list[0]], obs[mod_list[1]]]
        return list_inputs, mod_list

    def loss_calculation(self, output, gt):
        # out_1 = output[1]
        # loss
        loss_1 = self.criterion(output[1], gt)
        add1_loss = 0
        add2_loss = 0

        loss_p = self.criterion(output[-2][-1], gt)
        for j in range(len(output[-2]) - 2):
            add1_loss = add1_loss + self.criterion(output[-2][j], gt)
            add2_loss = add2_loss + self.criterion(output[-2][j], output[-2][j + 1])
        return loss_1, loss_p, add1_loss, add2_loss

    def train(self):
        # load the pretrained model and keep training
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(self.args.test.checkpoint_path_2)
            self.support_model.load_state_dict(checkpoint_2["model"])
        else:
            checkpoint_1 = torch.load(
                self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(
                self.args.test.checkpoint_path_2, map_location=torch.device("cpu")
            )
            self.support_model.load_state_dict(checkpoint_2["model"])

        enforce = False  # False by default

        self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
        criterion_support = nn.MSELoss()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Total number of parameters: ", pytorch_total_params)

        # Create optimizer
        optimizer_ = build_optimizer(
            [
                self.model,
                self.support_model,
            ],
            # self.model,
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

                gt_input, _, _, _, gt = data_1

                # optimizer
                optimizer_.zero_grad()
                before_op_time = time.time()

                ################## the learning curriculum ######################
                # forward pass
                latent_state = self.support_model(gt_input)

                # make a selection of modality
                list_inputs, mod_list = self.modality_selection(data_1)
                if enforce == True:
                    # enforce selected modalities
                    list_inputs = [data_1[1], data_1[2], data_1[3]]
                    mod_list = [0, 1, 2]

                # forward pass filter
                inputs = (list_inputs, mod_list)
                output = self.model(inputs, latent_state)
                out_1 = output[1]

                # loss
                loss_1, loss_p, add1_loss, add2_loss = self.loss_calculation(output, gt)

                # forward pass for the 2nd time
                latent_input = output[-1]
                latent_input = rearrange(
                    latent_input, "bs (en k) dim -> bs en k dim", k=1
                )
                latent = torch.cat((latent_state[:, :, 1:, :], latent_input), axis=2)
                gt_input, _, _, _, gt = data_2
                list_inputs, mod_list = self.modality_selection(data_2)
                if enforce == True:
                    # enforce selected modalities
                    list_inputs = [data_1[1], data_1[3]]
                    mod_list = [0, 1]

                inputs = (list_inputs, mod_list)
                output = self.model(inputs, latent)
                out_2 = output[1]

                loss_2, loss_p_, add1_loss_, add2_loss_ = self.loss_calculation(
                    output, gt
                )
                #################################################################
                if epoch >= 3:
                    add1_loss = add1_loss + add1_loss_
                    add2_loss = add2_loss + add2_loss_
                    loss_p = loss_p + loss_p_

                    final_loss = loss_1 + loss_2 + loss_p + add1_loss + 0.1 * add2_loss
                else:
                    add1_loss = add1_loss + add1_loss_
                    add2_loss = add2_loss + add2_loss_
                    final_loss = add1_loss + 0.1 * add2_loss

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
                    self.writer.add_scalar("t-1", loss_1.cpu().item(), self.global_step)
                    self.writer.add_scalar("t", loss_2.cpu().item(), self.global_step)
                    self.writer.add_scalar(
                        "transition", loss_p.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "decoder_loss", add1_loss.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "inner latent", add2_loss.cpu().item(), self.global_step
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
                checkpoint_tmp = {
                    "global_step": self.global_step,
                    "model": self.support_model.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "v-UR5-model-{}".format(self.global_step),
                    ),
                )
                torch.save(
                    checkpoint_tmp,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "v-UR5-support-model-{}".format(self.global_step),
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
                self.support_model.eval()
                self.model.eval()
                self.test()
                self.model.train()
                self.support_model.train()

            # Update epoch
            epoch += 1

    def online_test(self):
        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint_1 = torch.load(self.args.test.checkpoint_path_1)
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(self.args.test.checkpoint_path_2)
            self.support_model.load_state_dict(checkpoint_2["model"])
        else:
            checkpoint_1 = torch.load(
                self.args.test.checkpoint_path_1, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint_1["model"])
            checkpoint_2 = torch.load(
                self.args.test.checkpoint_path_2, map_location=torch.device("cpu")
            )
            self.support_model.load_state_dict(checkpoint_2["model"])
        self.model.eval()
        self.support_model.eval()

        test_dataset = UR5_sim_dataloader(self.args, "test")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8
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

            gt_input, _, _, _, gt = data_1
            # list_inputs, mod_list = self.modality_selection(data_1)

            full_modality = list(range(0, self.sensor_len))
            mod_list = full_modality
            list_inputs = [data_1[1], data_1[2], data_1[3]]
            enforce = True
            if enforce == True:
                # enforce selected modalities
                list_inputs = [data_1[1], data_1[2]]
                mod_list = [0, 1]

            # if step >= 50 and step <= 100:
            #     list_inputs = [data_1[2], data_1[3]]  # no rgb
            #     mod_list = [1, 2]

            # if step >= 150 and step <= 200:
            #     list_inputs = [data_1[1], data_1[3]]  # no depth
            #     mod_list = [0, 2]

            # if step >= 250 and step <= 300:
            #     list_inputs = [data_1[1], data_1[2]]  # no joint
            #     mod_list = [0, 1]

            with torch.no_grad():
                if step == 0:
                    latent_state = self.support_model(gt_input)
                else:
                    latent_input = output[-1]  # -> previous latent state
                    latent_input = rearrange(
                        latent_input, "bs (en k) dim -> bs en k dim", k=1
                    )
                    latent_state = torch.cat(
                        (latent_state[:, :, 1:, :], latent_input), axis=2
                    )

                inputs = (list_inputs, mod_list)
                output = self.model(inputs, latent_state)

                pred = output[1]  # -> final estimation
                attn = output[-3]  # -> attention map
                final_est = pred
                ensemble = output[0]

                final_est = final_est.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                attn = attn.cpu().detach().numpy()
                ensemble = ensemble.cpu().detach().numpy()

                data_save.append(final_est)
                gt_save.append(gt)
                modality.append(mod_list)
                atten.append(attn)
                ensemble_save.append(ensemble)
                step = step + 1

        data["state"] = data_save
        data["gt"] = gt_save
        data["modality"] = modality
        # data["attention"] = atten
        # data["ensemble"] = ensemble_save

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "v-UR5-test-N-{}.pkl".format(0),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)
