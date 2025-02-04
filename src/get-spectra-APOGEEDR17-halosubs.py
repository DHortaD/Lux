"""
Script to download the SDSS-IV spectra for overlapping stars in APOGEE DR17 and GALAH DR3
"""

# import the necessary modules
import numpy as np
from astropy.io import fits
from astropy.table import Table

# load in the file of stars you want the spectra for
file = 'APOGEE-DR17-halosubs.fits'
path = '/Users/dhortadarrington/Documents/Projects/the-paton/data/'+str(file)
tb = fits.open(path)
dat = tb[1].data 

mask = np.where(np.unique(dat['APOGEE_ID']))
dat = dat[mask]
print('Size of final sample: '+str(len(dat)))

# ####################### IF MWM SPECTRA (need to change path)

# # Code to do bulk download
# master = 'wget --no-check-certificate --user sdss --password 2.5-meters -r https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/'
# telescope = dat['TELESCOPE']
# field = dat['FIELD']
# file = dat['FILE']
# apoid = dat['APOGEE_ID_1']

# paths = []
# for indx, i in enumerate(file):
#     # if there is a FILE id
#     if len(i)>0:
#         paths.append(master+telescope[indx]+str('/')+field[indx]+str('/')+file[indx]) 
#     else:
#         paths.append(master+telescope[indx]+str('/')+field[indx]+str('/apStar-dr17-')+apoid[indx]+str('.fits')) 
# # save the file with all the spectra
# savepath = '/Users/dhortadarrington/Projects/El-Cañón/spec/'
# jnp.savetxt(savepath+'spectra.txt', paths, fmt = "%s") 

####################### IF APOGEE SPECTRA

# command line command to get spectra for one star 
# wget --spider https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/000+02/apStar-dr17-2M17335483-2753043.fits
# the "redux" path gets you the raw visit spectra, which isn't combined. Below is the path needed to get the combined spectra

# Code to do bulk download
file_name = 'APOGEE-giants-halosubs'
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

id = dat['APOGEE_ID']
snr = dat['SNR']
name = dat['name']


t = Table([id, snr, name], \
          names=('APOGEE_ID', 'SNR', 'name'))

savepath = '/Users/dhortadarrington/Documents/Projects/the-paton/data/'
t.write(savepath+'master-'+str(file_name)+'.fits', format='fits',overwrite=True)