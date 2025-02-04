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
# FUNCTIONS TO LOAD IN THE SPECTRA AND LABELS
############################################################

def load_data(spectra_dir_path, labels_file, file_name):
        """
                Load all the spectra and label data in one go

                INPUT:
                        spectra_dir_path: path to the directory with all the spectra files
                        labels_file: file name and path to the labels file

                OUTPUT:
                        spectra_data: wl, fluxes, fluxes_ivar
                        label_data: ids, labels, labels_err, labels_ivar
        """

        spectra_data = load_spectra(spectra_dir_path, file_name)
        label_data = load_labels(labels_file)
        print('Loaded data successfully')

        return spectra_data, label_data

def load_data_galah(spectra_dir_path, labels_file, file_name):
        """
                Load all the spectra and label data in one go

                INPUT:
                        spectra_dir_path: path to the directory with all the spectra files
                        labels_file: file name and path to the labels file

                OUTPUT:
                        spectra_data: wl, fluxes, fluxes_ivar
                        label_data: ids, labels, labels_err, labels_ivar
        """

        spectra_data = load_spectra(spectra_dir_path, file_name)
        label_data = load_labels_galah(labels_file)
        print('Loaded data successfully')

        return spectra_data, label_data

def load_spectra(path, file_name):

        """
                Load the spectra files and clean it up. Remove any bad pixels or anything with a bitmask set to =! 0

                INPUT:
                        path to the directory with all the spectra files

                OUTPUT:
                        wavelength, fluxes, and inverse variances
        """

        # create a path to save the file
        if os.path.isdir(path):
                pass
        else:
               os.mkdir(path)
        print(path + 'spectra_data'+str(file_name)+'.dat')

        # load the spectra data
        force = False
        if os.path.exists(path + 'spectra_data'+str(file_name)+'.dat') and not force:
                print('File already exists. Loading spectra data')
                with open(path + 'spectra_data'+str(file_name)+'.dat', 'rb') as f:
                       spectra_data = pickle.load(f)

        else:
                print('File does not already exists')
                print("Loading spectra from directory %s" %path)
                files = list(sorted([path + "/" + filename for filename in os.listdir(path) if filename.endswith('.fits')]))
                nstars = len(files)  

                for file, fits_file in tqdm.tqdm(enumerate(files)):
                        file_in = fits.open(fits_file)
                        flux_ = jnp.array(file_in[1].data) 
                        flux_err_ = jnp.array((file_in[2].data))

                        if file == 0:
                                npixels = len(flux_)
                                fluxes = jnp.zeros((nstars, npixels), dtype=float)
                                fluxes_err = jnp.zeros(fluxes.shape, dtype=float)
                                ivars = jnp.zeros(fluxes.shape, dtype=float)
                                start_wl = file_in[1].header['CRVAL1']
                                diff_wl = file_in[1].header['CDELT1']
                                val = diff_wl * (npixels) + start_wl
                                wl_full_log = jnp.arange(start_wl,val, diff_wl)
                                wl_full = [10 ** aval for aval in wl_full_log]
                                wl = jnp.array(wl_full)

                        fluxes = fluxes.at[file,:].set(flux_)
                        fluxes_err = fluxes_err.at[file,:].set(flux_err_)
                        ivar = 1. /flux_err_**2
                        ivars = ivars.at[file,:].set(ivar)

                        # do some quality controls
                        # where the inverse variances are low
                        pixmask = (ivar<0.01)
                        fluxes = fluxes.at[file,pixmask].set(1)
                        fluxes_err = fluxes_err.at[file,pixmask].set(9999)
                        ivars = ivars.at[file,pixmask].set(0.01)

                        pixmask2 = (flux_ > 0)
                        fluxes = fluxes.at[file, ~pixmask2].set(1)
                        fluxes_err = fluxes_err.at[file, ~pixmask2].set(9999)
                        ivars = ivars.at[file, ~pixmask2].set(0.01)
                        

                print("Spectra loaded")
                spectra_data = {'wl': jnp.array(wl), 'fluxes': jnp.array(fluxes),\
                                        'fluxes_err': jnp.array(fluxes_err), 'fluxes_ivars': jnp.array(ivars)}
        # save the file
        with open(path + '/spectra_data'+str(file_name)+'.dat', 'wb') as f:
                pickle.dump(spectra_data, f)
                
        return spectra_data


def load_labels(path):
        """ 
                Extracts reference labels from a file

                INPUT:
                        path to the directory with all the spectra files

                OUTPUT
                        ids, labels, labels_err, labels_ivar
        """

        tb_tr = fits.open(path)
        dat_tr = tb_tr[1].data 

        # get all the names of the headers in the file
        names = []
        for i in (dat_tr.columns):
                names.append(i.name)

        # divide the names into the values and their errors
        names_lab = []
        names_lab_err = []

        # get the label names
        for indx, i in enumerate(names[2:]): #skip the first one as its the IDs, the second is the SNR
                if '_ERR' in i: # this is for the error arrays
                        names_lab_err.append(i)
                else:
                        names_lab.append(i)

        # create an empty array to store values in for the dictionary
        len_labels = (len(names[2:])) #skip the first one as its the IDs, the second is the SNR
        labels = jnp.zeros((int(len_labels), len(dat_tr))) 
        labels_err = jnp.zeros((int(len_labels), len(dat_tr)))

        # ids are always first column
        ids = dat_tr[names[0]]
        # SNR is always second column
        snr = dat_tr[names[1]]

        # loop through all the columns in the file and store the data
        for indx, i in tqdm.tqdm(enumerate(names[2:])):
                if i in names_lab:
                        labl = jnp.array(dat_tr[i])
                        mask = jnp.isnan(labl) | (labl<-10) # if the value is a NaN, inf, or-9999
                        label = labl.at[mask].set(jnp.nanmedian(labl[(labl>-10)]))
                        labels = labels.at[indx].set(label)
                elif i in names_lab_err:
                        labl_err = jnp.array(dat_tr[i])
                        mask = jnp.isnan(labl_err) | (labl_err<-10) # if the error is a NaN, inf, or -9999
                        label_err = labl_err.at[mask].set(9999)
                        labels_err = labels_err.at[indx].set(label_err)                

        labels_err = labels_err[~np.all(labels_err == 0, axis=1)]
        labels = labels[~np.all(labels == 0, axis=1)]

        label_data = {'ids': ids, 'snr': snr, 'label_names': names_lab,  'labels': labels.T, 'label_names_err': names_lab_err, 'labels_err': labels_err.T, 'labels_ivars': (1./labels_err**2).T}

        return label_data


def load_labels_galah(path):
        """ 
                Extracts reference labels from a file

                INPUT:
                        path to the directory with all the spectra files

                OUTPUT
                        ids, labels, labels_err, labels_ivar
        """

        tb_tr = fits.open(path)
        dat_tr = tb_tr[1].data 

        # get all the names of the headers in the file
        names = []
        for i in (dat_tr.columns):
                names.append(i.name)

        # divide the names into the values and their errors
        names_lab = []
        names_lab_err = []

        # get the label names
        for indx, i in enumerate(names[2:]): #skip the first one as its the IDs, the second is the SNR
                if i == 'fe_h':
                        names_lab.append(i)
                elif i == 'Ce_fe':
                        names_lab.append(i)
                elif 'e_' in i: # this is for the error arrays
                        names_lab_err.append(i)
                else:
                        names_lab.append(i)

        # create an empty array to store values in for the dictionary
        len_labels = (len(names[2:])) #skip the first one as its the IDs, the second is the SNR
        labels = jnp.zeros((int(len_labels), len(dat_tr))) 
        labels_err = jnp.zeros((int(len_labels), len(dat_tr)))

        # ids are always first column
        ids = dat_tr[names[0]]
        # SNR is always second column
        snr = dat_tr[names[1]]

        # loop through all the columns in the file and store the data
        for indx, i in tqdm.tqdm(enumerate(names[2:])):
                if i in names_lab:
                        labl = jnp.array(dat_tr[i])
                        mask = jnp.isnan(labl) | (labl<-10) # if the value is a NaN, inf, or-9999
                        label = labl.at[mask].set(jnp.nanmedian(labl[(labl>-10)]))
                        labels = labels.at[indx].set(label)
                elif i in names_lab_err:
                        labl_err = jnp.array(dat_tr[i])
                        mask = jnp.isnan(labl_err) | (labl_err<-10) # if the error is a NaN, inf, or -9999
                        label_err = labl_err.at[mask].set(9999)
                        labels_err = labels_err.at[indx].set(label_err)                

        labels_err = labels_err[~np.all(labels_err == 0, axis=1)]
        labels = labels[~np.all(labels == 0, axis=1)]

        label_data = {'ids': ids, 'snr': snr, 'label_names': names_lab,  'labels': labels.T, 'label_names_err': names_lab_err, 'labels_err': labels_err.T, 'labels_ivars': (1./labels_err**2).T}

        return label_data