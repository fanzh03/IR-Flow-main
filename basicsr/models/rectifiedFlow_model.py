import logging
from collections import OrderedDict
import os
import numpy as np
import cv2
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import basicsr.models.lr_scheduler as lr_scheduler
import basicsr.models.networks as networks
from basicsr.models.optimizer import Lion

from basicsr.models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class RFOdeModel(BaseModel):
    def __init__(self, opt):
        super(RFOdeModel, self).__init__(opt)
        # self.state = None
        self.output = None
        self.x_start = None  # LQ
        self.gt = None  # GT
        self.isXT = opt["method"]["isXT"]
        self.base = opt["method"]["base"]
        self.weighta = opt["method"]["weighta"]

        if opt["dist"]:
            self.rank = dist.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                    k,
                    v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, LQ, GT):
        self.x_start = LQ.to(self.device)  # LQ
        self.gt = GT.to(self.device)  # GT

    def optimize_parameters(self, timesteps, sde=None):
        self.optimizer.zero_grad()
        timesteps = timesteps.to(self.device)

        if self.base:
            G_theta = sde.score_fn(self.x_start)
            xt_1_optimum = self.gt
        else:
            G_theta, x_t = sde.noise_fn(self.x_start, self.gt, timesteps.squeeze())
            if self.isXT:
                xt_1_optimum = x_t - self.gt
            else:
                xt_1_optimum = self.x_start - self.gt

        if self.weighta == 0:
            loss_add = 0
        else:
            loss_add = self.multi_step_consistency_constraints(sde, self.x_start, self.gt)

        loss = self.weight * self.loss_fn(G_theta, xt_1_optimum) + loss_add * self.weighta

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde, current_step, timesteps, solver="Euler-1"):
        self.model.eval()
        output = OrderedDict()
        with torch.no_grad():
            if self.base:
                output[f'step_1'] = sde.score_fn(self.x_start)
            else:
                for i in timesteps:
                    output[f'step_{i}'] = sde.reverse_ode(self.x_start, T=i, solver=solver)
            self.output = output
        self.save_pic(current_step)
        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["Output"] = {}
        out_dict["Input"] = self.x_start.detach()[0].float().cpu()
        out_dict["GT"] = self.gt.detach()[0].float().cpu()
        for key in self.output:
            out_dict["Output"][key] = self.output[key].detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
                self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')

    def save_pic(self, current_step):
        batch_size = self.gt.shape[0]
        cols = batch_size
        
        # Convert to three-channel image uniformly
        def ensure_rgb(image):
            if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)   # Convert single-channel to three-channel
            return image
        
        # Add channel handling to conversion function
        def tensor_to_cv2_image(tensor):
            # Input tensor shape: (C, H, W)
            img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to H×W×C
            img = np.clip((img * 255.0).round(), 0, 255).astype(np.uint8)
            return ensure_rgb(img)  # Ensure output is three channels
        
        # Convert all images and unify channels
        images_LQ = [tensor_to_cv2_image(self.x_start[i]) for i in range(batch_size)]
        images_GT = [tensor_to_cv2_image(self.gt[i]) for i in range(batch_size)]

        # Create image grid (ensure all images are H×W×3)
        grid_image_LQ = self.create_image_grid(images_LQ, cols)
        grid_image_GT = self.create_image_grid(images_GT, cols)

        # Initialize final image
        final_image = np.vstack((grid_image_GT, grid_image_LQ))

        # Process each output branch
        for key in self.output:
            images_output = [tensor_to_cv2_image(self.output[key][i]) for i in range(batch_size)]
            grid_image_output = self.create_image_grid(images_output, cols)
            final_image = np.vstack((final_image, grid_image_output))

        # Save results
        cv2.imwrite(os.path.join(self.opt['path']['val_images'], f'val_{current_step}.png'), final_image)

    def multi_step_consistency_constraints(self, sde, x, gt):
        self.model.eval()
        with torch.no_grad():
            K = np.random.randint(2, 11)
            h = 1000 / K
            x_i = x
            for i in list(reversed(range(1, K + 1))):
                w_i = 1 / i
                t_i = torch.tensor(i * h).long().unsqueeze(0)
                d_i = sde.score_fn(x_i, t_i)
                x_i = x_i - w_i * d_i
        loss = self.loss_fn(x_i, gt)
        self.model.train()
        return loss

    def tensor_to_cv2_image(self, tensor_img):
        """
        Convert PyTorch tensor to OpenCV image
        """
        # Convert tensor from CHW to HWC format
        image = torch.clamp(tensor_img, 0, 1)
        if image.is_cuda:
            image = image.cpu()  # Move tensor from GPU to CPU
        image = image.permute(1, 2, 0).numpy()

        # If single-channel (grayscale), no color conversion needed
        if image.shape[2] == 3:
            # Convert from RGB to BGR (OpenCV default format)
            image = image[:, :, [2, 1, 0]]

        # Scale values from [0, 1] to [0, 255]
        image = np.uint8(np.around(image * 255))
        return image

    def create_image_grid(self, images, cols):
        """
        Stitch images into a grid
        """
        rows = len(images) // cols
        assert len(images) == rows * cols
        # Get height and width of a single image
        height, width, channels = images[0].shape
        # Create a blank grid image
        grid_image = np.zeros((rows * height, cols * width, channels), dtype=np.uint8)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid_image[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = img

        return grid_image
