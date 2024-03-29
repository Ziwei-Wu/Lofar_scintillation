#!/usr/bin/env python

"""
create_dyn_psrflux.py
----------------------------------
Crate dynamic spectrum with "psrflux" 
"""

import time
import matplotlib.pyplot as plt
import argparse
import warnings
import subprocess
import psrchive as psr
import numpy as np
import os


def ignorewarnings():   
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', FutureWarning)

def archiveinfo(archive):
    arch = psr.Archive_load(archive)
    name = arch.get_source()
    freq_lo = arch.get_centre_frequency() - arch.get_bandwidth()/2.0
    freq_hi = arch.get_centre_frequency() + arch.get_bandwidth()/2.0
    #mjd_start = arch.get_Integration(0).get_start_time().in_days()
    #mjd_end = arch.get_Integration(0).get_end_time().in_days()
    mjd_start = float(arch.start_time().strtempo())
    mjd_end = float(arch.end_time().strtempo())
    site = arch.get_telescope()
    nsubint = arch.get_nsubint()
    nchan = arch.get_nchan()
    coord = arch.get_coordinates()
    ra = coord.ra().getHMS()
    dec = coord.dec().getHMS()

    return name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec, nsubint, nchan

def standard_psrflux(archives):
    """
    get dynamic spectrum with psrflux with self-standard profile
    """
    for archive in archives:
        dyn_name = archive + '.ds'
        if not os.path.isfile(dyn_name):
            start = time.time()
            print "Loadding dynamic spectrum with psrflux ......"
            subprocess.call('psrflux %s' % archive, shell=True)
            end = time.time()
            print("...Loaded in {0} seconds\n".format(round(end-start, 2)))

    """
    Load a dynamic spectrum from psrflux-format file
    """
    for archive in archives:
        archive_dyn = archive + '.ds'
        name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec, nsubint, nchan = archiveinfo(archive)
        with open (archive_dyn, "r") as file:
            rawdata = np.loadtxt(archive_dyn).transpose()
            fluxes = rawdata[4]
            dynspec = fluxes.reshape(nsubint, nchan)
            full_name = archive + '_dyn.txt'
            np.savetxt('%s' % full_name, dynspec,  fmt='%.5f')
            with open("%s" % (full_name), "a") as myfile:
                myfile.write("\n # %s %s %f %f %f %f %s %s" % (name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec))
            #subprocess.call('rm %s' % archive_dyn, shell=True)
    return dynspec, name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the archive.')
    parser.add_argument('archives', nargs='+', help='The chosen archives file')
    args = parser.parse_args()
    archives = args.archives

    ignorewarnings()
  
    dynspec, name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec = standard_psrflux(archives)
