import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from astropy.io import fits
import os
import jaxopt
from functools import partial
import tqdm


def load_spectra(path):

        """
                Load the spectra files and clean it up. Remove any bad pixels or anything with a bitmask set to =! 0

                INPUT:
                        path to the directory with all the spectra files

                OUTPUT:
                        wavelength, fluxes, and inverse variances
        """

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
                        ivars = jnp.zeros(fluxes.shape, dtype=float)
                        start_wl = file_in[1].header['CRVAL1']
                        diff_wl = file_in[1].header['CDELT1']
                        val = diff_wl * (npixels) + start_wl
                        wl_full_log = jnp.arange(start_wl,val, diff_wl)
                        wl_full = [10 ** aval for aval in wl_full_log]
                        wl = jnp.array(wl_full)

                ivar = 1. /flux_err_**2
                fluxes = fluxes.at[file,:].set(flux_)
                ivars = ivars.at[file,:].set(ivar)
                ivars = ivars.at[file,ivars[file]<0.01].set(0.01)
                fluxes = fluxes.at[file,ivars[file]<0.01].set(1)

        pixmask = jnp.all(fluxes>0,axis=0)&jnp.all(jnp.isfinite(fluxes),axis=0)

        print("Spectra loaded")
        return jnp.array(wl[pixmask]), jnp.array(fluxes[:,pixmask]), jnp.array(ivars[:,pixmask])
            
def load_labels(path):
        """ 
        Extracts reference labels from a file

        INPUT:
                path to the directory with all the spectra files

        OUTPUT
                ids, labels
        """

        tb_tr = fits.open(path)
        dat_tr = tb_tr[1].data 

        # for now, just pick the labels. This needs to be tweaked for each project goals
        ids = dat_tr['APOGEE_ID']
        teff = jnp.array(dat_tr['TEFF'])
        logg = jnp.array(dat_tr['LOGG'])
        feh = jnp.array(dat_tr['FE_H'])
        mgh = jnp.array(dat_tr['MG_FE']) + feh 

        return ids , jnp.dstack((teff,logg,feh,mgh))[0]


class CannonModel(object):
        def __init__(self, order):
                self.order = order # this is the order of the polynomial for the Cannon model (default is 2)


        ####################################
        # Training code
        ###################################

        @partial(jax.jit, static_argnames=["self"])
        def get_data_theta_opt(self, fluxes_train, ivars_train, labels_train, pivots, scales):

                # just to get the right size 
                lvec_train = self.get_lvec(labels_train, pivots, scales)

                theta_array = jnp.ones((fluxes_train.shape[1], lvec_train.shape[1]))
                params = {'theta': theta_array}
                data = {'fluxes': fluxes_train, 'ivars':ivars_train, 'labels': labels_train}
                return params, data

        @partial(jax.jit, static_argnames=["self"])
        def get_data_label_opt(self,thetas, fluxes_train, ivars_train, labels_train, pivots, scales):

                params = {'labels': labels_train}
                data = {'fluxes': fluxes_train, 'ivars':ivars_train, 'theta': thetas, 'pivots': pivots, 'scales': scales}
                return params, data

        @partial(jax.jit, static_argnames=["self"])
        def get_pivots_scales(self, labels_train):        
                """
                Function to get the pivots and scales (i.e., the median and the range)
                """       
                # get the pivots and scales
                qs = jnp.percentile(labels_train, jnp.array([2.5, 50., 97.5]), axis=0)
                pivots = qs[1]
                # pivots and scales here are used to regularise the function (i.e., Teff is in the 1000s but logg in the units, so to make the model treat each label value similarly you normalise it)
                scales = (qs[2] - qs[0])/4. # 4 is because 95 percentile range is 4 sigma (-2*sigma to +2*sigma)

                return pivots, scales                


        @partial(jax.jit, static_argnames=["self"])
        def get_lvec(self, labels_train, pivots, scales):
                """
                Function to get the vector of the label values and associated cross terms
                """
                
                linear_offsets = (labels_train - pivots[None, :]) / scales[None, :]

                quadratic_off = jnp.array([jnp.outer(l, l)[jnp.triu_indices(labels_train.shape[1])]for l in (linear_offsets)])
                ones = jnp.ones((labels_train.shape[0],1))

                lvec = jnp.hstack((ones, linear_offsets, quadratic_off))

                return lvec
        
        @partial(jax.jit, static_argnames=["self"])
        def least_sq_solve_onewavelength(self, params, data_train, pivots, scales):

                """
                Solve by least squares at every flux in the wavelength; here wavelength flux is L, number of stars is N, and number of labels is L
                A = label vector, of size (N, L)
                C = covariance matrix, of size (N, N) (I like to think of this as identity matrix times N vector for the variances)
                Y = flux, of size (N)

                Least squares solution is : theta = (A.T * C^{-1} @ A) ^{-1} @ A.T * C^{-1} @ Y

                INPUT: 
                        lvec, ivars, fluxes for training data
                OUTPUT:
                        optimised thetas
                """
                ##### NOTE: No regularisation function for now.

                lvec = self.get_lvec(data_train['labels'], pivots, scales)

                A = lvec
                Cinv = data_train['ivars'] # already inverted variances here
                Y = data_train['fluxes']
                
                return jnp.linalg.solve(A.T * Cinv @ A, A.T * Cinv @ Y) 

        @partial(jax.jit, static_argnames="self")
        def get_synthetic_spectra_onestar(self, thetas, data_train, pivots, scales):
                
                """"
                Get synthetic spectra (i.e., optimised thetas times the lvec array) to compare to the original spectra

                INPUT:
                        data array [lvec, ivars, fluxes]

                OUTPUT:
                        synthetic spectra (i.e., lvec @ thetas.T)
                """
                
                lvec = self.get_lvec(data_train['labels'], pivots, scales)

                return lvec@thetas.T


        @partial(jax.jit, static_argnames="self")
        def get_chi(self, data_train, synthetic_spectra):
                """
                Calculate the chi (i.e., the difference between the fluxes array and the synthetic flux divided by the sigma) for one star

                INPUT:
                        data array [lvec, ivars, fluxes]

                OUTPUT:
                        chi for one star (i.e., how good the synthetic spectra compares to real spectra)
                """

                return (data_train['fluxes']-synthetic_spectra)*jnp.sqrt(data_train['ivars'])
        

        ####################################
        # Testing code
        ###################################
        
        @partial(jax.jit, static_argnames=["self"])
        # static argument just means to ignore (propagate through without being part of jax tree)
        def ln_gaussian_likelihood_solve_for_labels(self, params, data):

                """
                Compute the chi2 (Gaussian) log-likelihood at every flux pixel in the wavelength.

                INPUT: 
                        data test array [test fluxes, test ivars] and thetas

                """

                lvec = self.get_lvec(params['labels'], data['pivots'], data['scales'])

                model_flux = jnp.dot(lvec, data['theta'].T)
                var = (1./data['ivars']) #scatter2 should be in param

                # summing over lambda wavelength
                return jnp.nansum(-0.5 * ((data['fluxes'] - model_flux) ** 2 /var) - 0.5 * jnp.log(var)) 

        @partial(jax.jit, static_argnames=["self"])
        def objective_labels(self, params, data):

                """
                Compute the negative log-likelihood of the Gaussian likelihood function.
                """

                return  -(self.ln_gaussian_likelihood_solve_for_labels(params, data)) 


        @partial(jax.jit, static_argnames=["self"])
        def optimise_labels(self, thetas, fluxes_train, ivars_train, labels_train, pivots, scales):
        
                """
                Get the optimised parameters for one flux pixel
                """

                params, data = self.get_data_label_opt(
                        thetas, jnp.atleast_2d(fluxes_train), jnp.atleast_2d(ivars_train), jnp.atleast_2d(labels_train), pivots, scales
                        )
                
                # optimizer = jaxopt.ScipyMinimize(fun=self._objective_thetas, method='L-BFGS-B')
                # optimizer = jaxopt.GradientDescent(fun=self._objective_labels)
                optimizer = jaxopt.LBFGS(fun=self.objective_labels)
                
                res = optimizer.run(init_params = params, data = data)

                return res

class Dataset(object):
    """ A class to represent Cannon input: a dataset of spectra and labels """

    def __init__(self, wl, ids_train, flux_train, ivar_train, label_train, ids_test, flux_test, ivar_test, label_test):
        """ Initiate a Dataset object

        Parameters
        ----------
        wl: grid of wavelength values, onto which all spectra are mapped
        ids_train: array of IDs of training objects
        flux_train: array of flux values for training objects, [nobj, npix]
        ivar_train: array [nobj, npix] of inverse variance values for training objects
        label_train: array [nobj, nlabel]
        ids_test: array [nobj[ of IDs of test objects
        test_flux: array [nobj, npix] of flux values for test objects
        test_ivar: array [nobj, npix] of inverse variance values for test objects
        """
        print("Loading dataset")
        self.wl = wl
        self.ids_train = ids_train
        self.flux_train = flux_train
        self.ivar_train = ivar_train
        self.label_train = label_train
        self.ids_test = ids_test
        self.flux_test = flux_test
        self.ivar_test = ivar_test
        self.label_test = label_test
        self.ranges = None
        

                