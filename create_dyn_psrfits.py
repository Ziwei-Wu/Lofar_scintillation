import argparse
import warnings
import numpy as np
import psrchive as psr
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import os
import subprocess


def ignorewarning():
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', FutureWarning)

def chunks(l,n):
    """Yield successive n-sized chunks from l."""
    #for i in range(0, len(l), n):
    #    yield l[i,i+n]
    return ([l[i:i+n] for i in range(0,len(l), n)])

def archiveinfo(archives):
    freq_lo_list = []
    freq_hi_list = []
    for archive in archives:
        arch = psr.Archive_load(archive)
        name = arch.get_source()
        freq_lo = arch.get_centre_frequency() - arch.get_bandwidth()/2.0
        freq_hi = arch.get_centre_frequency() + arch.get_bandwidth()/2.0
        mjd_start = float(arch.start_time().strtempo())
        mjd_end = float(arch.end_time().strtempo())
        site = arch.get_telescope()
        coord = arch.get_coordinates()
        ra = coord.ra().getHMS()
        dec = coord.dec().getHMS()
        freq_lo_list.append(freq_lo)
        freq_hi_list.append(freq_hi)

    freq_lo = np.min(freq_lo_list)
    freq_hi = np.max(freq_hi_list)
        

    return name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec

def readpsrfits(fname, undo_scaling=True, dedisperse=True):
   
    """Read folded spectrum from psrfits file, based on Dana's code
    Parameters
    ----------
    fname : string
        Fits file to be opened
    undo_scaling : True or False
        Remove the automatic normalization applied by psrfits.  This renders
        output identical to psrchive within machine precision
    dedisperse : True or False
        Roll in phase by closest incoherent time-shift, using the in-header
        values of DM, tbin.  Phase 0 point is arbitrary
    Returns: Data cube [time, pol, freq, phase], 
             frequency array [freq], 
             time array [time]
    """

    f = fits.open(fname)
    freq = f['SUBINT'].data['DAT_FREQ'][0]
    #F = freq*u.MHz
    
    shape = f['SUBINT'].data[0][-1].shape
    nt = f['SUBINT'].header['NAXIS2']
    nchan = shape[1]

    # Exact start time is encompassed in three header values
    t0 = Time(f['PRIMARY'].header['STT_IMJD'], format='mjd', precision=9)
    t0 += (f['PRIMARY'].header['STT_SMJD'] + f['PRIMARY'].header['STT_OFFS']) *u.s
    # Array of subint lengths, 
    Tint = f['SUBINT'].data['TSUBINT']*u.s
    T = t0 + np.cumsum(Tint)
    
    # Read and polulate data with each subint, one at a time
    data = np.zeros((nt,shape[0],shape[1],shape[2]), dtype=np.float32)
    for i in np.arange(nt):
        data[i,:,:,:] = f['SUBINT'].data[i][-1]
    
    if undo_scaling:
        # Data scale factor (outval=dataval*scl + offs) 
        scl = f['SUBINT'].data['DAT_SCL'].reshape(nt, 4, nchan)
        offs = f['SUBINT'].data['DAT_OFFS'].reshape(nt, 4, nchan)
        data = data * scl[...,np.newaxis] + offs[..., np.newaxis]

    # Remove best integer bin incoherent time shift
    if dedisperse:
        fref = f['HISTORY'].data['CTR_FREQ'][0]
        DM = f['SUBINT'].header['DM']
        tbin = f['HISTORY'].data['Tbin'][0]

        for i in np.arange(len(freq)):
            dt = (1 / 2.41e-4) * DM * (1./fref**2.-1./freq[i]**2.)
            tshift = np.int(np.rint(dt/tbin))
            data[:,:,i,:] = np.roll(data[:,:,i,:],tshift,axis=-1)


    # Sum to form total intensity, mostly a memory/speed concern
    if len(data.shape) == 4:
        data = data[:,(0,1)].mean(1)

    return data, freq

def foldspectrum(archives):
    switch = 1
    freq_list = []
    i = 0
    for archive in archives:
        i += 1
        print ("loading %s %s" % (archive, i))
        rf_name = archive.rsplit('.', 1)[0] + '.rf'
        if not os.path.isfile(rf_name):
            subprocess.call('psrconv -o PSRFITS %s' % (archive), shell=True)
        I, f = readpsrfits(rf_name, undo_scaling=True, dedisperse=True)
        if switch:
            Ic = np.copy(I)
            F = np.copy(f)
            switch = 0
        else:
            Ic = np.concatenate((Ic, I), axis=1)
            F = np.concatenate((F,f), axis=0)

    #F = F*u.MHz

    return Ic, F

def clean_foldspec(I, apply_mask=True, rfimethod='var'):
    """
    Clean and rescale folded spectrum
    
    Parameters
    ----------
    I: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots
    apply_mask: Bool
    Multiply dynamic spectrum by mask
    rfimethod: String
    RFI flagging method, currently only supports var
    Returns folded spectrum, after bandpass division, 
    off-gate subtractionand RFI masking
    """

    
    # Use median over time to not be dominated by outliers
    bpass = np.median( I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass
    
    mask = np.ones_like(I.mean(-1))

    if rfimethod == 'var':
        print ('Cleaning RFI, it will take some time, just waitting ......')
        flag = np.std(foldspec, axis=-1)

        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize//4), int(3*flagsize//4) )
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean+10*flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0)<0.5] = 0
        if apply_mask:
            I = I*mask[...,np.newaxis]
    

    # determine off_gates as lower 50% of profile
    profile = I.mean(0).mean(0)
    off_gates = np.argwhere(profile<np.median(profile)).squeeze()

    # renormalize, now that RFI are zapped
    #bpass = I[...,off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    bpass = I[...,off_gates].mean(-1, keepdims=True)
    bpass = bpass.mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = 0
    foldspec -= np.mean(foldspec[...,off_gates], axis=-1, keepdims=True)

    return foldspec, flag, mask

def getdynspec(foldspec, profsig=5., bint=1, plot=False, plotbin=4, sigma=10.):
    """
    Create dynamic spectrum from folded data cube
    
    Uses average profile as a weight, sums over phase
    Parameters 
    ----------                                                                                                         
    foldspec: [time, frequency, phase] array
    profsig: S/N value, mask all profile below this
    bint: integer, bin dynspec by this value in time
    """
    
    # Create profile by summing over time, frequency, normalize peak to 1
    template = foldspec.mean(0).mean(0)
    template /= np.max(template)

    # Noise from bottom 50% of profile
    tnoise = np.std(template[template<np.median(template)])
    template[template < tnoise*profsig] = 0
    profplot2 = np.concatenate((template, template), axis=0)

    # Multiply the profile by the template, sum over phase
    
    dynspec = (foldspec*template[np.newaxis,np.newaxis,:]).mean(-1)
    tbins = int(dynspec.shape[0] // bint)
    dynspec = dynspec[-bint*tbins:].reshape(tbins, bint, -1).mean(1)
    dynspec /= np.std(dynspec)

    #clean agagin using basic zapping of dynamic spectrum
    #dynspec = np.abs(dynspec - np.median(dynspec[~np.isnan(dynspec)]))
    #mdev = np.median(dynspec[~np.isnan(dynspec)])
    #sn = dynspec/mdev
    #dynspec[sn > sigma] = np.nan
    

    if plot:
        # # Bin in time, freq, by some factor (only for plotting)
        bint = plotbin
        binf = plotbin
        # # finding array bounds divisible by binninf factor
        tbins = dynspec.shape[0] // bint
        fbins = dynspec.shape[1] // binf
        dyn_binned = dynspec[:tbins*bint,:fbins*binf].reshape(tbins, bint, fbins*binf).mean(1)
        dyn_binned = dyn_binned.reshape(tbins, fbins, binf).mean(-1)
        std = np.std(dyn_binned)
        mean = np.mean(dyn_binned)
        plt.figure(figsize=(12,18))
        plt.imshow(dyn_binned.T, aspect='auto', vmin=mean-2.3*std, vmax=mean+4*std,
                origin='lower', cmap='viridis')
        plt.colorbar()
        plt.show()

    return dynspec


def standard_psrfits(args):
    archives = args.inputfile
    factor=args.combinefacotr

    #seperate the input archives into several chunks because of memory error
    archives_segmentation = chunks(archives,factor)

    #analyse archives for each chunk
    switch = 1
    freq_lo_list = []
    freq_hi_list = []
    number = 0
    while number < len(archives_segmentation):
        #get basic information about observations
        name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec = archiveinfo(archives_segmentation[number])
        freq_lo_list.append(freq_lo)
        freq_hi_list.append(freq_hi)
        #get foldspec spectrum
        foldspec, Freq = foldspectrum(archives_segmentation[number])
        #remove RFI
        foldspec, flag, mask = clean_foldspec(foldspec, apply_mask=True, rfimethod='var')
        #get dynamic spectrum
        dynspec = getdynspec(foldspec, profsig=5., bint=1, plot=False, plotbin=4, sigma=10.)
        #combine all dynspec spectrum, just like psradd -R , combining dynamic spectrum at frequency
        if switch:
            dynspeccombine = np.copy(dynspec)
            switch = 0
        else:
            dynspeccombine = np.concatenate((dynspeccombine, dynspec), axis=1)
        number += 1
    freq_lo = np.min(freq_lo_list)
    freq_hi = np.max(freq_hi_list)

    #save dynspec 
    full_name =  '%s_%.2f_%.2f_dyn.txt' % (name, mjd_start, (freq_lo+freq_hi)/2)
    np.savetxt(full_name, dynspeccombine, fmt='%.9f')
    with open(full_name, "a") as myfile:
        myfile.write("\n # %s %s %f %f %f %f %s %s" %
                     (name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec)) 

    return dynspeccombine, name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec


def main():
    parser = argparse.ArgumentParser(description='Select the archive.')
    parser.add_argument('inputfile', nargs='+', help='the input archive')
    parser.add_argument('-c','--combinefacotr', type=int, default=1, help='Combine factor')
    args = parser.parse_args()
    dynspeccombine, name, site, freq_lo, freq_hi, mjd_start, mjd_end, ra, dec = standard_psrfits(args)

if  __name__=="__main__":
    ignorewarning()
    main()
