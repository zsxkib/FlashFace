"""Noise schedules of denoising diffusion probabilistic models.

We consider a variance preserving (VP) process, and we use the standard deviation
sigma_t of the noise added to the signal at time t to represent the noise schedule. The
corresponding diffusion process is:

q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I),

where alpha_t^2 = 1 - sigma_t^2.
"""
import math
import numbers

import torch

__all__ = [
    'vp_to_ve', 've_to_vp', 'betas_to_sigmas', 'sigmas_to_betas',
    'logsnrs_to_sigmas', 'sigmas_to_logsnrs', 'linear_schedule',
    'scaled_linear_schedule', 'cosine_schedule', 'sigmoid_schedule',
    'karras_schedule', 'exponential_schedule', 'polyexponential_schedule',
    'vp_schedule', 'logsnr_cosine_schedule', 'logsnr_cosine_shifted_schedule',
    'logsnr_cosine_interp_schedule', 'noise_schedule'
]

# --------------------- conversion between VE and VP sigmas ---------------------#


def vp_to_ve(sigmas):
    if isinstance(sigmas, numbers.Number):
        sigmas = (sigmas ** 2 / (1 - sigmas ** 2)) ** 0.5 \
            if sigmas < 1. else float('inf')
    else:
        sigmas = torch.where(sigmas < 1., (sigmas**2 / (1 - sigmas**2))**0.5,
                             float('inf'))
    return sigmas


def ve_to_vp(sigmas):
    if isinstance(sigmas, numbers.Number):
        sigmas = (sigmas ** 2 / (1 + sigmas ** 2)) ** 0.5 \
            if sigmas < float('inf') else 1.
    else:
        sigmas = torch.where(sigmas < float('inf'),
                             (sigmas**2 / (1 + sigmas**2))**0.5, 1.)
    return sigmas


# ------------------------ conversion between variables ------------------------#


def betas_to_sigmas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))


def sigmas_to_betas(sigmas):
    square_alphas = 1 - sigmas**2
    betas = 1 - torch.cat(
        [square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
    return betas


def logsnrs_to_sigmas(logsnrs):
    return torch.sqrt(torch.sigmoid(-logsnrs))


def sigmas_to_logsnrs(sigmas):
    square_sigmas = sigmas**2
    return torch.log(square_sigmas / (1 - square_sigmas))


# ------------------------ noise schedules ------------------------#


def linear_schedule(n, beta_min=0.0001, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, n, dtype=torch.float64)
    return betas_to_sigmas(betas)


def scaled_linear_schedule(n, beta_min=0.00085, beta_max=0.012):
    betas = torch.linspace(beta_min**0.5,
                           beta_max**0.5,
                           n,
                           dtype=torch.float64)**2
    return betas_to_sigmas(betas)


def cosine_schedule(n, cosine_s=0.008):
    ramp = torch.linspace(0, 1, n + 1, dtype=torch.float64)
    square_alphas = torch.cos(
        (ramp + cosine_s) / (1 + cosine_s) * torch.pi / 2)**2
    betas = (1 - square_alphas[1:] / square_alphas[:-1]).clamp(max=0.999)
    return betas_to_sigmas(betas)


def sigmoid_schedule(n, beta_min=0.0001, beta_max=0.02):
    betas = torch.sigmoid(torch.linspace(
        -6, 6, n, dtype=torch.float64)) * (beta_max - beta_min) + beta_min
    return betas_to_sigmas(betas)


def karras_schedule(n,
                    sigma_min=ve_to_vp(0.002),
                    sigma_max=ve_to_vp(80.0),
                    rho=7.0):
    # VP (sigma range) -> VE (compute ve-sigmas) -> VP (output vp-sigmas)
    ramp = torch.linspace(0, 1, n, dtype=torch.float64)
    min_inv_rho = vp_to_ve(sigma_min)**(1 / rho)
    max_inv_rho = vp_to_ve(sigma_max)**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    return ve_to_vp(sigmas)


def exponential_schedule(n,
                         sigma_min=ve_to_vp(0.002),
                         sigma_max=ve_to_vp(80.0)):
    # VP (sigma range) -> VE (compute ve-sigmas) -> VP (output vp-sigmas)
    sigmas = torch.linspace(math.log(vp_to_ve(sigma_min)),
                            math.log(vp_to_ve(sigma_max)),
                            n,
                            dtype=torch.float64).exp()
    return ve_to_vp(sigmas)


def polyexponential_schedule(n,
                             sigma_min=ve_to_vp(0.002),
                             sigma_max=ve_to_vp(80.0)):
    # VP (sigma range) -> VE (compute ve-sigmas) -> VP (output vp-sigmas)
    ramp = torch.linspace(0, 1, n, dtype=torch.float64)
    sigmas = torch.exp(
        ramp *
        (math.log(vp_to_ve(sigma_max)) - math.log(vp_to_ve(sigma_min))) +
        math.log(vp_to_ve(sigma_min)))
    return ve_to_vp(sigmas)


def vp_schedule(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3):
    t = torch.linspace(eps_s, 1, n, dtype=torch.float64)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return ve_to_vp(sigmas)


def _logsnr_cosine(n, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_min))
    t_max = math.atan(math.exp(-0.5 * logsnr_max))
    t = torch.linspace(1, 0, n, dtype=torch.float64)
    logsnrs = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    return logsnrs


def _logsnr_cosine_shifted(n, logsnr_min=-15, logsnr_max=15, scale=2):
    logsnrs = _logsnr_cosine(n, logsnr_min, logsnr_max)
    logsnrs += 2 * math.log(1 / scale)
    return logsnrs


def _logsnr_cosine_interp(n,
                          logsnr_min=-15,
                          logsnr_max=15,
                          scale_min=2,
                          scale_max=4):
    t = torch.linspace(1, 0, n, dtype=torch.float64)
    logsnrs_min = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_min)
    logsnrs_max = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_max)
    logsnrs = t * logsnrs_min + (1 - t) * logsnrs_max
    return logsnrs


def logsnr_cosine_schedule(n, logsnr_min=-15, logsnr_max=15):
    return logsnrs_to_sigmas(_logsnr_cosine(n, logsnr_min, logsnr_max))


def logsnr_cosine_shifted_schedule(n, logsnr_min=-15, logsnr_max=15, scale=2):
    return logsnrs_to_sigmas(
        _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale))


def logsnr_cosine_interp_schedule(n,
                                  logsnr_min=-15,
                                  logsnr_max=15,
                                  scale_min=2,
                                  scale_max=4):
    return logsnrs_to_sigmas(
        _logsnr_cosine_interp(n, logsnr_min, logsnr_max, scale_min, scale_max))


# ------------------------ a unified wrapper for all schedules ------------------------#


def noise_schedule(schedule='logsnr_cosine_interp',
                   n=1000,
                   zero_terminal_snr=False,
                   **kwargs):
    # compute sigmas
    sigmas = {
        'linear': linear_schedule,
        'scaled_linear': scaled_linear_schedule,
        'cosine': cosine_schedule,
        'sigmoid': sigmoid_schedule,
        'karras': karras_schedule,
        'exponential': exponential_schedule,
        'polyexponential': polyexponential_schedule,
        'vp': vp_schedule,
        'logsnr_cosine': logsnr_cosine_schedule,
        'logsnr_cosine_shifted': logsnr_cosine_shifted_schedule,
        'logsnr_cosine_interp': logsnr_cosine_interp_schedule
    }[schedule](n, **kwargs)

    # post-processing
    if zero_terminal_snr and sigmas.max() != 1.0:
        scale = (1.0 - sigmas.min()) / (sigmas.max() - sigmas.min())
        sigmas = sigmas.min() + scale * (sigmas - sigmas.min())
    return sigmas
