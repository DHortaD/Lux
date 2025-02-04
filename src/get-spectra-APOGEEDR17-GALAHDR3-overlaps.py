"""
Script to download the SDSS-IV spectra for overlapping stars in APOGEE DR17 and GALAH DR3
"""

# import the necessary modules
import numpy as np
from astropy.io import fits
from astropy.table import Table

# load in the file of stars you want the spectra for
file = 'APOGEEDR17_GALAHDR3.fits'
path = '/Users/dhortadarrington/Documents/Projects/the-paton/data/'+str(file)
tb = fits.open(path)
dat = tb[1].data 

# load in the GC VAC to remove globular cluster stars
file2 = 'VAC_GC_DR17_synspec_rev1_beta.fits'
path2 = '/Users/dhortadarrington/Documents/Master/data/'+str(file2)
tb2 = fits.open(path2)
dat2 = tb2[1].data 

# create a mask to remove GC stars
mask_gcs = np.isin(list(dat['APOGEE_ID']),list(dat2['APOGEE_ID']))

# create a mask to remove duplicates
unique, ix = np.unique(dat['APOGEE_ID'], return_index=True)
mask_unique = np.full(dat['APOGEE_ID'].shape, True)
mask_unique[ix] = False

# mask out any bad or missing values
mask = (dat['teff']>3800)&(dat['teff']<6000)&(dat['logg']>0.)&(dat['logg']<3.5)&(dat['SNR']>50)&(dat['snr_c1_iraf']>20)&(dat['snr_c2_iraf']>20)\
    &(dat['snr_c3_iraf']>20)&(dat['snr_c4_iraf']>20)&(dat['fe_h']>-2.)&(dat['fe_h']<1.)&(dat['Mg_fe']>-1.)&(dat['Mg_fe']<1.)

print('Initial size of sample: '+str(len(dat)))
print('Size of sample with good stellar parameters: '+str(len(dat[mask])))

dat = dat[mask]
dat = dat[:5000]

print('Size of final sample: '+str(len(dat)))

####################### IF APOGEE SPECTRA

# command line command to get spectra for one star 
# wget --spider https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/000+02/apStar-dr17-2M17335483-2753043.fits
# the "redux" path gets you the raw visit spectra, which isn't combined. Below is the path needed to get the combined spectra

# Code to do bulk download
file_name = 'APOGEE-giants-GALAH'
master = 'wget -P spectra-reference-stars-'+str(file_name)+'/ -np -xnH --cut-dirs 9 --no-check-certificate --user sdss --password 2.5-meters -r https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/'
telescope = dat['TELESCOPE']
field = dat['FIELD']
file = dat['FILE']
apoid = dat['APOGEE_ID']


print('Downloading spectra for '+str(len(file))+' stars')

paths = []
for indx, i in enumerate(file):
    # if there is a FILE id
    paths.append(master+telescope[indx]+str('/')+field[indx]+str('/aspcapStar-dr17-')+apoid[indx]+str('.fits')) 


# save the file with all the spectra
savepath = '/Users/dhortadarrington/Documents/Projects/the-paton/spec/'
np.savetxt(savepath+'wget_spectra-'+str(file_name)+'.txt', paths, fmt = "%s") 

###################### After getting the wget file
# you run in the command line, followed by 
# chmod 775 wget_spectra.txt
# ./wget_spectra.txt

# this will download all the spectra files into the folder "spectra-reference-aspcapStar/" within the directory you are in


#############################
# if by any chance the spectra are downloaded like on the SAS into subdirectories

# you can then run:
# " find . -name '*.fits' -exec mv {} /Users/dhortadarrington/Documents/Projects/El-Cañón/spec \; "
# to get all the files in a single directory

#############################

## As GALAH stores not measured quantities as NaNs, we need to turn those into numbers
dat['Y_fe'][np.isnan(dat['Y_fe'])] = -9999.
dat['Li_fe'][np.isnan(dat['Li_fe'])] = -9999.
dat['O_fe'][np.isnan(dat['O_fe'])] = -9999.
dat['Na_fe'][np.isnan(dat['Na_fe'])] = -9999.
dat['Ba_fe'][np.isnan(dat['Ba_fe'])] = -9999.
dat['Eu_fe'][np.isnan(dat['Eu_fe'])] = -9999.
dat['C_fe'][np.isnan(dat['C_fe'])] = -9999.
dat['Ce_fe'][np.isnan(dat['Ce_fe'])] = -9999.

dat['e_Y_fe'][np.isnan(dat['e_Y_fe'])] = -9999.
dat['e_Li_fe'][np.isnan(dat['e_Li_fe'])] = -9999.
dat['e_O_fe'][np.isnan(dat['e_O_fe'])] = -9999.
dat['e_Na_fe'][np.isnan(dat['e_Na_fe'])] = -9999.
dat['e_Ba_fe'][np.isnan(dat['e_Ba_fe'])] = -9999.
dat['e_Eu_fe'][np.isnan(dat['e_Eu_fe'])] = -9999.
dat['e_C_fe'][np.isnan(dat['e_C_fe'])] = -9999.
dat['e_Ce_fe'][np.isnan(dat['e_Ce_fe'])] = -9999.


########################## save the input training data file with the stellar parameters of interest
# this will need to be modified depending on what you wanna train on

id = dat['APOGEE_ID']
snr = dat['SNR']
snr_galah1 = dat['snr_c1_iraf']
snr_galah2 = dat['snr_c2_iraf']
snr_galah3 = dat['snr_c3_iraf']
snr_galah4 = dat['snr_c4_iraf']
teff = dat['teff']
logg = dat['logg']
feh = dat['fe_h']
cfe = dat['C_fe']
nafe = dat['Na_fe']
ofe = dat['O_fe']
mgfe = dat['Mg_fe']
life = dat['Li_fe']
yfe = dat['Y_fe']
cefe = dat['Ce_fe']
bafe = dat['Ba_fe']
eufe = dat['Eu_fe']

teff_err = dat['e_teff']
logg_err = dat['e_logg']
feh_err = dat['e_fe_h']
cfe_err = dat['e_C_fe']
nafe_err = dat['e_Na_fe']
ofe_err = dat['e_O_fe']
mgfe_err = dat['e_Mg_fe']
life_err = dat['e_Li_fe']
yfe_err = dat['e_Y_fe']
cefe_err = dat['e_Ce_fe']
bafe_err = dat['e_Ba_fe']
eufe_err = dat['e_Eu_fe']

t = Table([id, snr, snr_galah1, snr_galah2, snr_galah3, snr_galah4, teff, logg, feh, life, cfe, nafe, ofe, mgfe, yfe, cefe, bafe, eufe, \
           teff_err, logg_err, feh_err, life_err, cfe_err, nafe_err, ofe_err, mgfe_err, yfe_err, cefe_err, bafe_err, eufe_err], \
          names=('APOGEE_ID', 'SNR', 'snr_c1_iraf', 'snr_c2_iraf', 'snr_c3_iraf', 'snr_c4_iraf', 'teff', 'logg','fe_h', 'Li_fe','C_fe',\
                'Na_fe', 'O_fe', 'Mg_fe', 'Y_fe', 'Ce_fe', 'Ba_fe', 'Eu_fe',\
                'e_teff', 'e_logg','e_fe_h', 'e_Li_fe','e_C_fe', 'e_Na_fe', 'e_O_fe', 'e_Mg_fe', 'e_Y_fe', 'e_Ce_fe', 'e_Ba_fe', 'e_Eu_fe'))

savepath = '/Users/dhortadarrington/Documents/Projects/the-paton/data/'
t.write(savepath+'master-'+str(file_name)+'.fits', format='fits',overwrite=True)
