# import necessary modules
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

                        
##################################################################################################################
# FUNCTIONS TO GET THE DATA IN THE RIGHT FORMAT FOR OPTIMISATION INCLUDING NOISE and regularisation in z latents
##################################################################################################################

def get_data_beta_step_fixed_betas_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the wavelength plate stored in dictionary format to optimise the flux scatters
                INPUT: 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        params: ln_noise_fluxes
                        data: fluxes, ivars, betas, zetas
        """
             
        params = {'ln_noise_fluxes': ln_noise_fluxes}
        data = {'betas': betas, 'zetas': zetas, 'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars}

        return params, data

def get_data_beta_step_fixed_fluxnoise_opt_betas(betas, zetas, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the wavelength plate stored in dictionary format to optimise betas
                INPUT: 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        params: betas
                        data: fluxes, ivars, zetas, ln_noise_fluxes
        """
             
        params = {'betas': betas}
        data = {'zetas': zetas, 'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars, 'ln_noise_fluxes': ln_noise_fluxes}

        return params, data

def get_data_zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the star plate stored in dictionary format to optimise zetas and scatters in pixels
                INPUT:
                        labels: array of flux values; size N x Lambda 
                        labels_ivars: array of flux values; size N x Lambda 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        alphas: latent parameters for the labels, size M x P
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT: 
                        params: zetas, ln_noise_fluxes
                        data: alphas, betas, labels, labels_ivars, fluxes, fluxes_ivars
        """
             
        params = {'zetas': zetas, 'ln_noise_fluxes': ln_noise_fluxes}
        data = {'alphas': alphas, 'betas': betas,\
                'labels': labels, 'labels_ivars': labels_ivars, \
                'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars}

        return params, data


############################################################
# FUNCTIONS TO ESTIMATE THE LATENT PARAMETERS
############################################################

################# FOR ALPHAS

def synthesise_all_labels_all_stars(zeta, alpha):
        """
                Function to use in the loss function for optimising all labels and all stars. For now, this is treated as linear model, but
                could in principle be replaced with any functional form (e.g., quadratic, CNN, GP)

                INPUT: 
                        zeta: latent parameter zetas
                        alpha: latent parameter alphas

                OUTPUT:
                        dot product of zeta and alpha
        """

        return zeta @ alpha.T

def all_labels_all_stars_Gaussian_likelihood(params, data):
        """
                Gaussian log-likelihood function for all labels and all stars
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        log-likelihood 
        """
        
        model_label = synthesise_all_labels_all_stars(data['zetas'], params['alphas'])
        noise = 1./data['labels_ivars'] # here the scatter values are the variances
        # compute the chi
        return -0.5 * jnp.nansum((data['labels'] - model_label)**2 / (noise)) 

def one_label_all_stars_objective(params, data):
        """
                Objective function for all labels and all stars
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        -log-likelihood 
        """
        return - (all_labels_all_stars_Gaussian_likelihood(params, data))

################# FOR BETAS

def synthesise_all_wavelengths_all_stars(zeta, beta):
        """
                Function to use in the loss function for optimising all wavelengths and all stars. For now, this is treated as linear model, but
                could in principle be replaced with any functional form (e.g., quadratic, CNN, GP)

                INPUT: 
                        zeta: latent parameter zetas
                        beta: latent parameter betas

                OUTPUT:
                        dot product of zeta and beta
        """

        return zeta @ beta.T
    
def all_wavelengths_all_stars_Gaussian_likelihood_fixed_fluxnoise_opt_betas(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise betas
                INPUT: 
                        params and data from get_data_beta_step_fixed_fluxnoise_opt_betas function
                OUTPUT:
                        log-likelihood 
        """
        model_flux = synthesise_all_wavelengths_all_stars(data['zetas'], params['betas'])
        V = jnp.exp(2 * data['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_fixed_fluxnoise_opt_betas(params, data):
        """
                Objective function for all labels and all stars to optimise betas
                INPUT: 
                        params and data from get_data_beta_step_fixed_fluxnoise_opt_betas function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_wavelengths_all_stars_Gaussian_likelihood_fixed_fluxnoise_opt_betas(params, data)

def beta_step_opt_betas(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT: 
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised betas
        """

        params, data = get_data_beta_step_fixed_fluxnoise_opt_betas(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_fixed_fluxnoise_opt_betas, tol=1e-6, maxiter=1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data) 

        return res

################# FOR NOISE FLUXES
    
def all_wavelengths_all_stars_Gaussian_likelihood_fixed_betas_opt_fluxnoise(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise scatters in pixels
                INPUT: 
                        params and data from get_data_beta_step_fixed_betas_opt_fluxnoise function
                OUTPUT:
                        log-likelihood 
        """
        model_flux = synthesise_all_wavelengths_all_stars(data['zetas'], data['betas'])
        V = jnp.exp(2 * params['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_fixed_betas_opt_fluxnoise(params, data):
        """
                Objective function for all labels and all stars to optimise scatters
                INPUT: 
                        params and data from get_data_beta_step_fixed_betas_opt_fluxnoise function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_wavelengths_all_stars_Gaussian_likelihood_fixed_betas_opt_fluxnoise(params, data)

def beta_step_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT: 
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised scatters
        """

        params, data = get_data_beta_step_fixed_betas_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_fixed_betas_opt_fluxnoise, tol=1e-6, maxiter=1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data) 

        return res

################# FOR ZETAS (not jitted to save memory)

def all_stars_all_wavelengths_all_labels_Gaussian_likelihood(params, data, reg_std, omega):
        """
                Gaussian log-likelihood function for all labels, all wavelengths, and all stars to optimise zetas
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        log-likelihood 
        """

        model_labels = synthesise_all_labels_all_stars(params['zetas'], data['alphas'])
        noise_labels = 1./ data['labels_ivars'] 
        loglike_labels = -0.5 * jnp.nansum((data['labels'] - model_labels)**2 / (noise_labels))
        
        model_fluxes = synthesise_all_wavelengths_all_stars(params['zetas'], data['betas'])
        V =  jnp.exp(2 * params['ln_noise_fluxes'])
        noise_fluxes = 1./ data['fluxes_ivars'] + V 
        loglike_fluxes = -0.5 * jnp.nansum((data['fluxes'] - model_fluxes)**2 / (noise_fluxes)) -0.5 * jnp.nansum(jnp.log(noise_fluxes))
        l2_reg = - 0.5 * jnp.sum(params['zetas']**2 / reg_std**2)

        return omega * loglike_labels + loglike_fluxes + l2_reg

def all_stars_all_wavelengths_all_labels_objective(params, data, reg_std, omega):
        """
                Objective function for all labels, all wavelengths, and all stars to optimise zetas
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_stars_all_wavelengths_all_labels_Gaussian_likelihood(params, data, reg_std, omega)

def zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT: 
                        alphas: latent parameters for labels, M x P
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        labels: labels for all stars, N x Lambda
                        labels_ivar: labels inverse variances for all stars, N x Lambda
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                        reg_std: L2 regularisation strength
                        omega: dummy variable set to 1, but initially included to allow the labels to weigh more in the likelihood
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_stars_all_wavelengths_all_labels_objective, tol = 1e-6, maxiter = 1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data, reg_std = reg_std, omega = omega) 

        return res      


###################### RUN FULL AGENDA

def run_agenda(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega):

        """
        Agenda function to run the beta_step_opt_fluxnoise, beta_step_opt_betas, and zeta_step

            INPUT:  
                alphas: latent parameters for labels, M x P
                betas: latent parameters for wavelengths, Lambda x P
                zetas: latent parameters, N x P
                labels: labels for all stars, N x Lambda
                labels_ivar: labels inverse variances for all stars, N x Lambda
                fluxes: fluxes for all stars, N x Lambda
                fluxes_ivar: flux inverse variances for all stars, N x Lambda
                ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                omega: dummy variable set to 1, but initially included to allow the labels to weigh more in the likelihood
            OUTPUT:
                updated optimised betas, zetas, ln_noise_fluxes, and current likelihood value
        """

        # first use (no-noise) optimised latents to infer noise in the fluxes
        res_fluxnoise = beta_step_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivars, ln_noise_fluxes)

        # re-determine the beta latents with noise in the fluxes
        res_beta_updated = beta_step_opt_betas(betas, zetas, fluxes, fluxes_ivars, res_fluxnoise.params['ln_noise_fluxes'])

        # with the noise and the new betas, use the old alphas to get the new zetas. Here, we will also re-optimise the noise in the fluxes to update the full model
        res_zeta_updated = zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega)

        return res_beta_updated.params['betas'], res_zeta_updated.params['zetas'], res_zeta_updated.params['ln_noise_fluxes'], res_zeta_updated.state.value

########################################################################################################################
# TESTING THE MODEL
########################################################################################################################

# NOTE: Idea is, you have now optimised for alphas, betas, and zetas (of the training set). You now need to find the 
# zetas corresponding to the test set (or new stars). Once you have those, you can estimate the labels

################### FOR ZETAS USING FLUX
def get_data_zeta_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes):
        """
                Function to get the data and parameters for the star wavelength plate stored in dictionary format to optimise zetas for test set using betas, scatters, and fluxes
                INPUT:
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT: 
                        params: zetas
                        data: betas, fluxes, fluxes_ivars, ln_noise_fluxes
        """
             
        params = {'zetas': zetas}
        data = {'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars,\
                 'betas': betas, 'ln_noise_fluxes': ln_noise_fluxes}

        return params, data

def all_wavelengths_all_stars_Gaussian_likelihood_test(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_fluxes function
                OUTPUT:
                        log-likelihood 
        """

        model_flux = synthesise_all_wavelengths_all_stars(params['zetas'], data['betas'])
        V = jnp.exp(2 * data['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_test(params, data):
        """
                Objective function for all wavelengths and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_fluxes function
                OUTPUT:
                        -log-likelihood 
        """
        return -(all_wavelengths_all_stars_Gaussian_likelihood_test(params, data))
        
def get_zetas_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes):
        """
                Function to optimise the zetas latents parameters at fixed scatters and betas using the fluxes and flux ivars
                INPUT: 
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_test, tol=1e-6, maxiter=3000, max_stepsize=1e3) # magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res


################### FOR ZETAS USING LABELS
def get_data_zeta_test_using_labels(labels, labels_ivars, alphas, zetas):
        """
                Function to get the data and parameters for the star labels plate stored in dictionary format to optimise zetas for test set using alphas and labels
                INPUT:
                        labels: array of label values; size N x M 
                        labels_ivars: array of inverse label variances; size N x M
                        alphas: latent parameters for the wavelengths, size M x P
                        zetas: latent parameters for the stars, size N x P
                OUTPUT: 
                        params: zetas
                        data: alphas, labels, labels_ivars
        """
             
        params = {'zetas': zetas}
        data = {'labels': labels, 'labels_ivars': labels_ivars, 'alphas': alphas}

        return params, data

def all_labels_all_stars_Gaussian_likelihood_test(params, data):
        """
                Gaussian log-likelihood function for all labels and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_labels function
                OUTPUT:
                        log-likelihood 
        """
        
        model_label = synthesise_all_labels_all_stars(params['zetas'], data['alphas'])
        noise = 1./data['labels_ivars'] # here the scatter values are the variances

        return -0.5 * jnp.nansum((data['labels'] - model_label)**2 / (noise)) 

def all_labels_all_stars_objective_test(params, data):
        """
                Objective function for all labels and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_labels function
                OUTPUT:
                        -log-likelihood 
        """
        return -(all_labels_all_stars_Gaussian_likelihood_test(params, data))

def get_zetas_test_using_labels(labels, labels_ivars, alphas, zetas):
        """
                Function to optimise the zetas latents parameters at fixes alphas using the labels and labels ivars
                INPUT: 
                        alphas: latent parameters for labels, M x P
                        zetas: latent parameters, N x P
                        labels: fluxes for all stars, N x M
                        labels_ivar: label inverse variances for all stars, N x M
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_test_using_labels(labels, labels_ivars, alphas, zetas)

        optimizer = jaxopt.LBFGS(fun=all_labels_all_stars_objective_test, tol=1e-6, maxiter=3000, max_stepsize=1e3) # magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res