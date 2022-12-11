#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Optional, Tuple, List, Callable, Any
from .iterators import TransitionIterator
from .torch_utility import TorchMiniBatch


"""
Ref:
  - https://github.com/amirgholami/PyHessian/

DOCS

group_product(xs, ys)
  - the inner product of two lists of variables xs,ys.

group_add(params, update, alpha=1)
  - params = params + update*alpha. (in-place addition)

normalization(v)
  - normalization of a list of vectors. 
  - [vi / (group_product(v,v) ** 0.5) for vi in v]

get_params_grad(model)
  - get model parameters and corresponding gradients.

hessian_vector_product(gradsH, params, v)
  - compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
  - gradient of (gradient * v) = H * v.

orthnormal(w, v_list)
  - make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w.
  - Done by subtracting coordinates in v_list.

"""


def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    hv = torch.autograd.grad(
        gradsH,
        params,
        grad_outputs=v,
        only_inputs=True,
        retain_graph=True
    )
    return hv

def orthnormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

def iterator_hv_product(
    vec: torch.Tensor,
    model: torch.nn.Module,
    criterion: Callable[[TorchMiniBatch, Optional[bool]], torch.Tensor],
    iterator: TransitionIterator,
    device: Optional[str] = 'cpu:0',
) -> Tuple[float, torch.Tensor]:

    num_data = 0  # count the number of datum points in the iterator
    THv = [torch.zeros(p.size()).to(device) for p in model.parameters()]  # accumulate result
    iterator.reset()
    for batch in iterator:
        model.zero_grad()
        tmp_num_data = len(batch.actions)
        loss = criterion(batch, True)
        loss.backward(create_graph=True)
        params, gradsH = get_params_grad(model)
        model.zero_grad()
        Hv = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=vec,
            only_inputs=True,
            retain_graph=False
        )
        THv = [THv1 + Hv1 * float(tmp_num_data) + 0. for THv1, Hv1 in zip(THv, Hv)]
        num_data += tmp_num_data

    THv = [THv1 / float(num_data) for THv1 in THv]
    eigenvalue = group_product(THv, vec).cpu().item()
    return eigenvalue, THv


def hessian_eigenvalues(
    model: torch.nn.Module,
    criterion: Callable[[TorchMiniBatch, Optional[bool]], torch.Tensor], 
    iterator: TransitionIterator,
    top_n: int,
    max_iter: int,
    tolerance: Optional[float]=1e-3,
    show_progress: Optional[bool]=False,
    device: Optional[str] = 'cpu:0'
) -> List[float]:
    """
    compute the top_n eigenvalues using power iteration method
    """
    assert top_n >= 1
    top_n_eigenvalues = []
    top_n_eigenvectors = []
    
    while len(top_n_eigenvalues) < top_n:
        eigenvalue = None
        v = normalization([torch.randn(p.size()).to(device) for p in model.parameters()])
        range_gen = tqdm(range(max_iter),disable=not show_progress,desc=f"HessianEigenvalues ({len(top_n_eigenvalues)+1}/{top_n})",)
        for itr in range_gen:
            v = orthnormal(v, top_n_eigenvectors)
            model.zero_grad()
            tmp_eigenvalue, Hv = iterator_hv_product(v, model, criterion, iterator, device)
            v = normalization(Hv)
            # set postfix with losses
            if show_progress:
                range_gen.set_postfix({'eigenval': tmp_eigenvalue})
            # update eigenvalue
            if eigenvalue is not None and abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tolerance:
                eigenvalue = tmp_eigenvalue
                break
            eigenvalue = tmp_eigenvalue
            
        top_n_eigenvalues.append(eigenvalue)
        top_n_eigenvectors.append(v)

    return top_n_eigenvalues


def hessien_empirical_spectral_density(
    model: torch.nn.Module,
    criterion: Callable[[TorchMiniBatch, Optional[bool]], torch.Tensor], 
    iterator: TransitionIterator,
    n_run: int,
    max_iter: int,
    show_progress: Optional[bool]=False,
    device: Optional[str] = 'cpu:0'
) -> Tuple[List]:
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    """
    eigen_list_full = []
    weight_list_full = []
    for r in range(n_run):
        v = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
        v = normalization(v)

        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        ############### Lanczos #################
        range_gen = tqdm(range(max_iter),disable=not show_progress,desc=f"HessianSpectra(SLQ) ({r+1}/{n_run})",)
        for i in range_gen:
            model.zero_grad()
            w_prime = [torch.zeros(p.size()).to(device) for p in model.parameters()]
            if i == 0:
                _, w_prime = iterator_hv_product(v, model, criterion, iterator, device)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(w_prime, v, alpha=-alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).to(device) for p in model.parameters()]
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                _, w_prime = iterator_hv_product(v, model, criterion, iterator, device)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)

        #T = torch.zeros(max_iter, max_iter).to(device)
        T = torch.diag(torch.tensor(alpha_list)).to(device)
        T[range(1, max_iter), range(0, max_iter-1)] = torch.tensor(beta_list)
        T[range(0, max_iter-1), range(1, max_iter)] = torch.tensor(beta_list)

        eigen_list, b_ = torch.linalg.eigh(T)
        weight_list = b_[0, :]**2
        eigen_list_full.append(eigen_list.cpu().tolist())
        weight_list_full.append(weight_list.cpu().tolist())
    return eigen_list_full, weight_list_full


def get_esd_plot(eigenvalues, weights):
    density, grids = density_generate(eigenvalues, weights)
    fig, ax = plt.subplots()

    ax.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density $+\epsilon$ (Log Scale, $\epsilon=10^{-7}$)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    return fig

def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.1):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 / (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)