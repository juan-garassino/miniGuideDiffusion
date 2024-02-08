from miniGuideDiffusion.manager.utils.scheduler import ddpm_schedules
import numpy as np
from colorama import Fore, Style

import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_diffusion_steps, device, drop_prob=0.1):
        """
        Differentiable Diffusion Probabilistic Model (DDPM) for training and sampling.

        Args:
        - nn_model (nn.Module): Neural network model used for prediction.
        - betas (tuple): Tuple of two floats representing beta parameters for DDPM schedules.
        - n_diffusion_steps (int): Number of diffusion steps.
        - device (str): Device used for computation ('cpu' or 'cuda').
        - drop_prob (float): Dropout probability for context.
        """
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # Register buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_diffusion_steps).items():
            self.register_buffer(k, v)

        self.n_diffusion_steps = n_diffusion_steps
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, context):
        """
        Forward pass of the DDPM model during training.

        Args:
        - x (torch.Tensor): Input tensor.
        - context (torch.Tensor): Context label tensor.

        Returns:
        - torch.Tensor: Mean squared error between added noise and predicted noise.
        """
        timestep_samples = torch.randint(1, self.n_diffusion_steps, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[timestep_samples, None, None, None] * x + self.sqrtmab[timestep_samples, None, None, None] * noise

        # Dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(context) + self.drop_prob).to(self.device)

        # Return mean squared error between added noise and predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, context, timestep_samples / self.n_diffusion_steps, context_mask))

    def sample(self, n_samples, size, device, guide_weight=0.0):
        """
        Sampling from the DDPM model.

        Args:
        - n_samples (int): Number of samples to generate.
        - size (tuple): Size of the samples.
        - device (str): Device used for computation ('cpu' or 'cuda').
        - guide_weight (float): Guidance weight for sampling (default: 0.0).

        Returns:
        - torch.Tensor: Generated samples.
        - np.array: Array of generated steps.
        """
        x_samples = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        context_samples = torch.arange(0, 10).to(device)  # Context for cycling through the MNIST labels
        context_samples = context_samples.repeat(int(n_samples / context_samples.shape[0]))

        # Don't drop context at test time
        context_mask = torch.zeros_like(context_samples).to(device)

        # Double the batch
        context_samples = context_samples.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_samples:] = 1.0  # Make second half of batch context free

        x_store = []  # Keep track of generated steps in case want to plot something

        for i in range(self.n_diffusion_steps, 0, -1):
            print(f"⏹ Sampling timestep {i}")

            timestep_samples = torch.tensor([i / self.n_diffusion_steps]).to(device)
            timestep_samples = timestep_samples.repeat(n_samples, 1, 1, 1)

            # Double the batch
            x_samples = x_samples.repeat(2, 1, 1, 1)
            timestep_samples = timestep_samples.repeat(2, 1, 1, 1)

            z = torch.randn(n_samples, *size).to(device) if i > 1 else 0

            # Split predictions and compute weighting
            predicted_noise = self.nn_model(x_samples, context_samples, timestep_samples, context_mask)
            noise_1 = predicted_noise[:n_samples]
            noise_2 = predicted_noise[n_samples:]
            weighted_noise = (1 + guide_weight) * noise_1 - guide_weight * noise_2
            x_samples = x_samples[:n_samples]
            x_samples = self.oneover_sqrta[i] * (x_samples - weighted_noise * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if i % 20 == 0 or i == self.n_diffusion_steps or i < 8:
                x_store.append(x_samples.detach().cpu().numpy())

        x_store = np.array(x_store)
        return x_samples, x_store
