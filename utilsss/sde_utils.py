import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate
import random
import numpy as np
import torch.nn as nn


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################
    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(range(1, T + 1)):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################
class IRSDE(SDE):
    """
    Let timestep t start from 0 to T
    """
    def __init__(self, T=1000, device=None):
        super().__init__(T, device)
        self.model = None
        self.isTimestep = False
        self.isXT = True

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    def set_timestep(self, isTimestep):
        self.isTimestep = isTimestep

    def set_XT(self, isXT):
        self.isXT = isXT

    def drift(self, x, t):
        return self.score_fn(x, t)

    def ode_reverse_drift(self, x, score, t):
        return score * self.dt

    def score_fn(self, x, t=None, **kwargs):
        # need to pre-set isTimestep and score_model
        if (self.isTimestep) and (t is not None):
            noise = self.model(x, t, **kwargs)
        else:
            noise = self.model(x, **kwargs)
        return noise

    def _normalize_t(self, t):
        """ t∈{0,1,...,999,1000}. (0,1)={0.000,0.001, ..., 1.000}"""
        return t.float() / self.T

    def _rectified_flow(self, x_start, y, t):
        """X_t"""
        t_u = self._normalize_t(t)

        t_u = t_u.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        t_u = t_u.expand_as(x_start)  # t_u, x_start same size
        x_t = t_u * x_start + (1 - t_u) * y
        return x_t

    def noise_fn(self, x_start, gt, t, **kwargs):
        # need to pre-set mu and score_model
        x_t = self._rectified_flow(x_start, gt, t)
        if self.isTimestep:
            noise = self.model(x_t, t, **kwargs)
        else:
            noise = self.model(x_t, **kwargs)
        return noise, x_t

    def select_timesteps_nolinear(self, num):
        choices = None
        if num < 1 or num > 1000:
            raise ValueError("num must be between 1 and 1000")
        if num == 1:
            choices = [(1.0, torch.tensor(1000).long())]
        if num == 2:
            choices = [(0.8, torch.tensor(1000).long()), (1.0, torch.tensor(200).long())]
        if num == 3:
            choices = [(0.8, torch.tensor(1000).long()), (0.5, torch.tensor(200).long()), (1.0, torch.tensor(100).long())]
        if num == 4:
            choices = [(0.5, torch.tensor(1000).long()),
                       (0.5, torch.tensor(500).long()), (0.8, torch.tensor(250).long()), (1.0, torch.tensor(50).long())]
        if num == 100:
            step = 1000 / num
            choices = [((1.0/num)/(1-(1.0/num)*i), torch.tensor(1000 - i*step).long()) for i in range(num)]
        return choices
    
    def euler_1_order(self, x, num):
        x_i = x
        if num in {2,3,4}:
            timesteps_list = self.select_timesteps_nolinear(num)
            for step in timesteps_list:
                t = step[1].unsqueeze(0)
                score = self.score_fn(x_i, t)
                x_i = x_i - score * step[0]
        else: 
            h = 1000 / num
            for i in list(reversed(range(1, num + 1))):
                w_i = 1 / i
                # w_i = (1 / num) / (i * h / 1000)
                t_i = torch.tensor(i * h).long().unsqueeze(0)
                d_i = self.score_fn(x_i, t_i)
                x_i = x_i - w_i * d_i
        return x_i

    def heun_2_order(self, x, num):
        if num == 2:
            x_i = x
            deltaT = 50
            t_i = torch.tensor(1000).long().unsqueeze(0)
            d_i = self.score_fn(x_i, t_i)
            x_mid = x_i - (1 / num) * d_i
            t_i_delta1 = torch.tensor((1 / num) * 1000 + deltaT).long().unsqueeze(0)
            t_i_delta2 = torch.tensor((1 / num) * 1000 - deltaT).long().unsqueeze(0)
            d_i_delta1 = self.score_fn(x_mid, t_i_delta1)
            d_i_delta2 = self.score_fn(x_mid, t_i_delta2)
            x_i_next = x_mid - 0.5 * (d_i_delta1 + d_i_delta2)
        else:
            h = 1000 / num
            x_i = x
            x_i_next = x
            for i in list(reversed(range(1, num + 1))):
                x_i = x_i_next
                w_i = (1 / i)
                t_i = torch.tensor(i * h).long().unsqueeze(0)
                d_i = self.score_fn(x_i, t_i)
                x_i_hat = x_i - w_i * d_i
                t_i_hat = torch.tensor((i - 1) * h).long().unsqueeze(0)
                d_i_hat = self.score_fn(x_i_hat, t_i_hat)
                x_i_next = x_i - (w_i / 2) * (d_i + d_i_hat)
        return x_i_next
    
    def standard_euler(self, x, T=-1):
        dt = 1 / T
        xt = x.clone()
        h = 1000 / T
        for i in list(reversed(range(1, T + 1))):  
            num_t = i / T     
            t = torch.tensor(i * h).long().unsqueeze(0)
            pred = self.score_fn(xt, t)
            xt = xt - pred * dt   
        return xt

    def reverse_ode(self, xt, T=-1, solver="Euler-1", **kwargs):
        T = self.T if T < 0 else T
        self.dt = 1 / T
        x = xt.clone()

        x_out = None
        if T == 1:
            t = torch.tensor(1000).long().unsqueeze(0)
            score = self.score_fn(x, t, **kwargs)
            x_out = x - score
        else:
            if solver == "Euler-1":
                x_out = self.euler_1_order(x, T)
            elif solver == "Heun-2":
                x_out = self.heun_2_order(x, T)

        return x_out

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, ):
        shape = xt.shape

        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (eps, self.T), to_flattened_numpy(xt),
                                       rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    # ******************************************************************************************************************
    # sample states for training
    def generate_random_timesteps(self, gt):
        gt = gt.to(self.device)
        batch = gt.shape[0]
        timesteps = torch.randint(0, self.T + 1, (batch, 1, 1, 1)).long()  # 0~1000

        return timesteps
