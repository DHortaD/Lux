import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from astropy.io import fits
import os
import jaxopt
from functools import partial
import tqdm
import dill as pickle


############################################################
# FUNCTIONS TO INITIALISE THE LABELS
############################################################

# @partial(jax.jit, static_argnums=2)
def initialise_alphas_betas_zetas(labels, fluxes, P):
        """
                Function to initiliase all the latent parameters in the model. 
                Here M is number of labels, Lambda is number of wavelengths, N is number of stars, and P is size of latent space
                INPUT: 
                        labels: labels for every star; size N x M
                        fluxes: fluxes for every star; size N x Lambda
                        P: size of the latent parameter space;
                OUTPUT:
                        alphas_init: initialised alphas (label) latent params; size M x P
                        betas_init: initialised betas (wavelength) latent params; size Lambda x P
                        zetas_init: initialised (vectorised) zetas; size N x P
        """
        # for alphas and betas, we will randomly sample initial guesses between 0 and 1
        alphas_init = np.random.rand(labels.shape[1], P) 
        betas_init = np.random.rand(fluxes.shape[1], P) 

        # for zetas, we will use the initialize_zetas function
        zz = np.random.rand(labels.shape[0], P) 
        zetas_init = initialize_zetas(labels, zz)

        return alphas_init, betas_init, zetas_init

def initialize_zetas(labels, zetas):

        """
                Function to initiliase the zetas as set values to run the alpha and beta steps. We will initialise the zetas to:
                        zetas = [1, (label1 - label1_median)/(label1_97.5 - label1_2.5), ..., (labeln - labeln_median)/(labeln_97.5 - labeln_2.5), 0, 0, ..., len(P)]
                INPUT: 
                        labels: array of labels; size N x M 
                        zetas: initial guess (random) latent parameters for the stars, size N x P
                OUTPUT:
                        zetas_init: initialised (vectorised) zetas; size N x P
        """

        # get pivots and scales
        qs = jnp.nanpercentile(labels, jnp.array([2.5, 50., 97.5]), axis=0)
        pivots = qs[1]
        scales = (qs[2] - qs[0])/4. # 4 is because 95 percentile range is 4 sigma (-2*sigma to +2*sigma)

        # initialise the zetas as zetas = [1, (label1 - label1_pivot)/label1_scale, ..., (labeln - labeln_pivot)/labeln_scale, 0, 0, 0, ..., len(K)]
        linear_offsets = (labels - pivots[None, :]) / scales[None, :]
        ones = jnp.ones((labels.shape[0],1))
        zeros = jnp.zeros((labels.shape[0], zetas.shape[1] - labels.shape[1] - 1))

        # because the P vector latent variable space is of size 100, if P < vectoriser 
        if zetas.shape[1] <= linear_offsets.shape[1]: 
                zetas_init = jnp.hstack((ones, linear_offsets))
        # else, fill the remaining size of the matrix with zeros
        else:
                zetas_init = jnp.hstack((ones, linear_offsets, zeros))
        return zetas_init