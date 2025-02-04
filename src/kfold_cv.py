import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import tqdm
from functools import partial
import load_data as ld
import optimise as opt
import scatters as opt_sc
import init_latents as il

##################################################################################
# FUNCTIONS TO RUN K-FOLD CV AND TO CALCULATE HOW WELL THE MODEL DOES
##################################################################################

########################## K-FOLD
def run_kfold(ids, labels, labels_err, labels_ivars, fluxes, fluxes_err, fluxes_ivars, ln_noise_fluxes_init, k, P, l2_reg_strength, omega, iterations = 5):

    print('Getting k-fold samples')
    ids_, lab_data, spec_data, alphas_init, betas_init, zetas_init = split_kfold_data(ids, labels, labels_err, labels_ivars, fluxes, fluxes_err, fluxes_ivars, k, P)

    alphas_kfold = []
    betas_kfold = []
    zetas_kfold = []
    ln_noise_fluxes_kfold = []
    chi2_iter_kfold = []
    chi2_labels_kfold = []
    chi2_fluxes_kfold = []
    chi2_tot_kfold = []

    print('Running optimisation of model latents on k-fold samples')
    for indx, i in tqdm.notebook.tqdm(enumerate(alphas_init)):
        alphas_ = i
        betas_ = betas_init[indx]
        zetas_ = zetas_init[indx]
        for jndx, j in enumerate(range(iterations)):
            # run optimisation routine without noise
            alphas_, betas_, zetas_, diff_chi2_iter, chi2_iter = opt.run_agenda(alphas_, betas_, zetas_, lab_data['labels_train'][indx], lab_data['labels_train_err'][indx],\
                                                                                            spec_data['fluxes_train'][indx], spec_data['fluxes_train_err'][indx], omega)

        # run optimisation routine with noise in the flux
        betas_iter_f, zetas_iter_f, ln_noise_fluxes_iter, nll_iter = opt_sc.run_agenda(alphas_, betas_, zetas_, lab_data['labels_train'][indx],\
                                                    lab_data['labels_train_ivars'][indx], spec_data['fluxes_train'][indx], spec_data['fluxes_train_ivars'][indx],\
                                                    ln_noise_fluxes_init, l2_reg_strength, omega)

        # store the best fitting params for every k-fold
        alphas_kfold.append(alphas_)
        betas_kfold.append(betas_iter_f)
        zetas_kfold.append(zetas_iter_f)
        ln_noise_fluxes_kfold.append(ln_noise_fluxes_iter)
        nll_iter.append(nll_iter)
        
        print('Calculating chi2 for k-fold samples')
        # calculate the chi2 for the labels, fluxes, and total for every k-fold
        chi2_labels, chi2_fluxes, chi2_tot = calc_chi2(alphas_, betas_iter_f, zetas_iter_f, lab_data['labels_train'][indx], lab_data['labels_train_ivars'][indx],\
                                                       spec_data['fluxes_train'][indx], spec_data['fluxes_train_ivars'][indx])
        chi2_labels_kfold.append(chi2_labels)
        chi2_fluxes_kfold.append(chi2_fluxes)
        chi2_tot_kfold.append(chi2_tot)

    return jnp.array(alphas_kfold), jnp.array(betas_kfold), jnp.array(zetas_kfold), jnp.array(ln_noise_fluxes_kfold), \
            jnp.array(nll_iter), chi2_labels_kfold, chi2_fluxes_kfold, chi2_tot_kfold, jnp.array(chi2_iter_kfold)
        
def calc_chi2(alphas, betas, zetas, labels, labels_ivar, fluxes, fluxes_ivar):

    model_labels = zetas @ alphas.T
    chi2_labels = jnp.nansum((labels - model_labels)**2 * (labels_ivar))

    model_fluxes = zetas @ betas.T
    chi2_fluxes = jnp.nansum((fluxes - model_fluxes)**2 * (fluxes_ivar))

    return chi2_labels, chi2_fluxes, chi2_labels + chi2_fluxes

def split_kfold_data(ids, labels, labels_err, labels_ivars, fluxes, fluxes_err, fluxes_ivars, k, P):

    spec_data_traintest = {'fluxes' : fluxes, 'fluxes_err' : fluxes_err, 'fluxes_ivars' : fluxes_ivars}
    lab_data_traintest = {'labels' : labels, 'labels_err' : labels_err, 'labels_ivars' : labels_ivars}

    ids_, lab_data, spec_data = get_kfold_data(ids, lab_data_traintest, spec_data_traintest, k)

    alphas_init, betas_init, zetas_init = init_params_kfold(lab_data['labels_train'], spec_data['fluxes_train'], P)

    return ids_, lab_data, spec_data, alphas_init, betas_init, zetas_init

########################################################################
# Functions to run k-fold cross-validation
########################################################################

def get_kfold_data(ids, label_data, spectra_data, k):

    """
        get the samples of the train and test sets for k-fold cross-validation
        k: number of subsets for k-fold cross-validation
        delta: difference between the starting point in each sample (i.e., shift in the samples)
    """
    ids_reshaped = ids.reshape(k, int(len(ids)/k))
    flux_reshaped = spectra_data['fluxes'].reshape(k, int(spectra_data['fluxes'].shape[0]/k), spectra_data['fluxes'].shape[1])
    flux_err_reshaped = spectra_data['fluxes_err'].reshape(k, int(spectra_data['fluxes_err'].shape[0]/k), spectra_data['fluxes_err'].shape[1])
    flux_ivars_reshaped = spectra_data['fluxes_ivars'].reshape(k, int(spectra_data['fluxes_ivars'].shape[0]/k), spectra_data['fluxes_ivars'].shape[1])
    labels_reshaped = label_data['labels'].reshape(k, int(label_data['labels'].shape[0]/k), label_data['labels'].shape[1])
    labels_err_reshaped = label_data['labels_err'].reshape(k, int(label_data['labels_err'].shape[0]/k), label_data['labels_err'].shape[1])
    labels_ivars_reshaped = label_data['labels_ivars'].reshape(k, int(label_data['labels_ivars'].shape[0]/k), label_data['labels_ivars'].shape[1])

    train_ids_list = []
    train_flux_list = []
    train_flux_err_list = []
    train_flux_ivars_list = []
    train_label_list = []
    train_label_err_list = []
    train_label_ivars_list = []

    test_ids_list = []
    test_flux_list = []
    test_flux_err_list = []
    test_flux_ivars_list = []
    test_label_list = []
    test_label_err_list = []
    test_label_ivars_list = []

    for jndx, j in enumerate(range(k)):
        temp_ids_train = []
        temp_flux_train = []
        temp_flux_err_train = []
        temp_flux_ivars_train = []
        temp_label_train = []
        temp_label_err_train = []
        temp_label_ivars_train = []
        temp_ids_test = []
        temp_flux_test = []
        temp_flux_err_test = []
        temp_flux_ivars_test = []
        temp_label_test = []
        temp_label_err_test = []
        temp_label_ivars_test = []
        for indx, i in enumerate(flux_reshaped):
            if indx != jndx:
                temp_ids_train.extend(ids_reshaped[indx])
                temp_flux_train.extend(i)
                temp_flux_err_train.extend(flux_err_reshaped[indx])
                temp_flux_ivars_train.extend(flux_ivars_reshaped[indx])
                temp_label_train.extend(labels_reshaped[indx])
                temp_label_err_train.extend(labels_err_reshaped[indx])
                temp_label_ivars_train.extend(labels_ivars_reshaped[indx])
            else:
                temp_ids_test.extend(ids_reshaped[indx])
                temp_flux_test.extend(i)
                temp_flux_err_test.extend(flux_err_reshaped[indx])
                temp_flux_ivars_test.extend(flux_ivars_reshaped[indx])
                temp_label_test.extend(labels_reshaped[indx])
                temp_label_err_test.extend(labels_err_reshaped[indx])
                temp_label_ivars_test.extend(labels_ivars_reshaped[indx])
        
        train_ids_list.append(temp_ids_train)
        train_flux_list.append(temp_flux_train)
        train_flux_err_list.append(temp_flux_err_train)
        train_flux_ivars_list.append(temp_flux_ivars_train)
        train_label_list.append(temp_label_train)
        train_label_err_list.append(temp_label_err_train)
        train_label_ivars_list.append(temp_label_ivars_train)

        test_ids_list.append(temp_ids_test)
        test_flux_list.append(temp_flux_test)
        test_flux_err_list.append(temp_flux_err_test)
        test_flux_ivars_list.append(temp_flux_ivars_test)
        test_label_list.append(temp_label_test)
        test_label_err_list.append(temp_label_err_test)
        test_label_ivars_list.append(temp_label_ivars_test)

    

    ids = {'train_ids': train_ids_list, 'test_ids': test_ids_list}
    spec_data = {'fluxes_train' : jnp.array(train_flux_list), 'fluxes_train_err' : jnp.array(train_flux_err_list), 'fluxes_train_ivars' : jnp.array(train_flux_ivars_list),\
                 'fluxes_test' : jnp.array(test_flux_list), 'fluxes_test_err' : jnp.array(test_flux_err_list), 'fluxes_test_ivars' : jnp.array(test_flux_ivars_list)}
    lab_data = {'labels_train' : jnp.array(train_label_list), 'labels_train_err' : jnp.array(train_label_err_list), 'labels_train_ivars' : jnp.array(train_label_ivars_list),\
                 'labels_test' : jnp.array(test_label_list), 'labels_test_err' : jnp.array(test_label_err_list), 'labels_test_ivars' : jnp.array(test_label_ivars_list)}
    
    return ids, lab_data, spec_data

def init_params_kfold(train_label, train_flux, P):

    alphas = []
    betas = []
    zetas = []

    for indx, i in enumerate(train_label):
        alphas_, betas_, zetas_ = il.initialise_alphas_betas_zetas(i, train_flux[indx], P)
        alphas.append(alphas_)
        betas.append(betas_)
        zetas.append(zetas_)

    return jnp.array(alphas), jnp.array(betas), jnp.array(zetas)




