# import necessary modules
import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from astropy.io import fits
import tqdm

import load_data as ld
import optimise as opt
import scatters as opt_sc
import init_latents as il
import kfold_cv as kf

import time

# load the data
file_name = '-train-rgbs-new'
spectra_dir_path = '../spec/spectra-reference-stars-APOGEE-giants-ref/'
file_path = '../data/master-APOGEE-giants-train.fits'
spectra_data, label_data = ld.load_data(spectra_dir_path, file_path, file_name)

# define the training set
train_ID = label_data['ids']
train_flux = spectra_data['fluxes']
train_flux_err = spectra_data['fluxes_err']
train_flux_ivar = spectra_data['fluxes_ivars']
train_label = label_data['labels']
train_label_err = label_data['labels_err']
train_label_ivar = label_data['labels_ivars']

# initialise the noise in the pixels
ln_noise_fluxes_init = jnp.full(train_flux.shape[1], -8.0)

# define a range of l2 regularisation strengths to test
l2_reg_strength = jnp.array([0.1, 1., 10., 100])

# define a range of latent dimensionality to test
P = jnp.array([13, 24, 48, 96]) # this is 1, 2, 4, 8 times size of the labels, M

# define other variables in the model and where to save outputs
iterations = 5
omega = 1. # we will set this to one as we tried varying this and it doesn't do much
k = 5 # 5 k-folds
savepath = '../sav/paper/k-fold-'

# loop over the latent dimensionality
for jndx, j in enumerate(P):
    ids_, lab_data, spec_data, alphas_init, betas_init, zetas_init = kf.split_kfold_data(train_ID, train_label, train_label_err, train_label_ivar, train_flux, train_flux_err, train_flux_ivar, k, j)

    # create empty lists to store values
    alphas_kfold = []
    betas_kfold = []
    zetas_kfold = []
    ln_noise_fluxes_kfold = []
    chi2_iter_kfold = []
    chi2_labels_kfold = []
    chi2_fluxes_kfold = []
    chi2_tot_kfold = []
    nll_iter_kfold = []

    print('Running optimisation of model latents on k-fold samples for a latent dimensionality of '+str(j))
    for hndx, h in tqdm.tqdm(enumerate(alphas_init)):
        alphas_ = h
        betas_ = betas_init[hndx]
        zetas_ = zetas_init[hndx]

        start_time = time.time()

        for lndx, l in enumerate(range(iterations)):
            
            # run optimisation routine without noise
            alphas_, betas_, zetas_, diff_chi2_iter, chi2_iter = opt.run_agenda(alphas_, betas_, zetas_, lab_data['labels_train'][hndx], lab_data['labels_train_err'][hndx],\
                                                                                            spec_data['fluxes_train'][hndx], spec_data['fluxes_train_err'][hndx], omega)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to run agenda: {elapsed_time:.2f} seconds")

        print('Running optimisation of the model latents with noise included')
        # loop over the l2 regularisation strengths
        for indx, i in tqdm.tqdm_notebook(enumerate(l2_reg_strength)):
            # run optimisation routine with noise in the flux
            betas_iter_f, zetas_iter_f, ln_noise_fluxes_iter, nll_iter = opt_sc.run_agenda(alphas_, betas_, zetas_, lab_data['labels_train'][hndx],\
                                                        lab_data['labels_train_ivars'][hndx], spec_data['fluxes_train'][hndx], spec_data['fluxes_train_ivars'][hndx],\
                                                        ln_noise_fluxes_init, i, omega)

        # store the best fitting params for every k-fold
        alphas_kfold.append(alphas_)
        betas_kfold.append(betas_iter_f)
        zetas_kfold.append(zetas_iter_f)
        ln_noise_fluxes_kfold.append(ln_noise_fluxes_iter)
        nll_iter_kfold.append(nll_iter)

        # calculate the chi2 for the labels, fluxes, and total for every k-fold
        chi2_labels, chi2_fluxes, chi2_tot = kf.calc_chi2(alphas_, betas_iter_f, zetas_iter_f, lab_data['labels_train'][hndx], lab_data['labels_train_ivars'][hndx],\
                                                        spec_data['fluxes_train'][hndx], spec_data['fluxes_train_ivars'][hndx])
        chi2_labels_kfold.append(chi2_labels)
        chi2_fluxes_kfold.append(chi2_fluxes)
        chi2_tot_kfold.append(chi2_tot)

        jnp.save(savepath + str(hndx) + '-alphas-l2strength'+str(i)+'-P'+str(j), alphas_kfold)
        jnp.save(savepath + str(hndx) + '-betas-l2strength'+str(i)+'-P'+str(j), betas_kfold)
        jnp.save(savepath + str(hndx) + '-zetas-l2strength'+str(i)+'-P'+str(j), zetas_kfold)
        jnp.save(savepath + str(hndx) + '-ln_noise_fluxes-l2strength'+str(i)+'-P'+str(j), ln_noise_fluxes_kfold)
        jnp.save(savepath + str(hndx) + '-nll-l2strength'+str(i)+'-P'+str(j), nll_iter_kfold)
        jnp.save(savepath + str(hndx) + '-chi2_labels-l2strength'+str(i)+'-P'+str(j), chi2_labels_kfold)
        jnp.save(savepath + str(hndx) + '-chi2_fluxes-l2strength'+str(i)+'-P'+str(j), chi2_fluxes_kfold)
        jnp.save(savepath + str(hndx) + '-chi2_tot-l2strength'+str(i)+'-P'+str(j), chi2_tot_kfold)



