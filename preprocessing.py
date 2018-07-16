import numpy as np
import os
import h5py
import re
import time

import util_worm as util


def eigenAmps_clean(eigenAmps):
    """
    Remove leading/trailing NaNs
    """
    if np.isnan(eigenAmps[0,0]):
        # get the end of the starting NaN segment
        afternan = np.where(np.isnan(eigenAmps))[0][-1] + 1
        np.where(eigenAmps[afternan:] == 1)[0][0]
        # drop these values
        eigenAmps[:, 0:nanEnd] = [];
    if np.isnan(eigenAmps[0, -1]):
        # get the start of the final NaN segment
        nanVals =np.where(np.isnan(eigenAmps[0, :]))[0]
        nanStart = np.where(np.diff(nanVals)!=1)[0][-1] + 1
        # drop these values
        eigenAmps[:, nanStart:] = []
    # linearly interpolate over missing values
    eigenAmpsNoNaN = util.nan_interpolate(eigenAmps)
    return eigenAmpsNoNaN

def worm_covariance(angleArray):
    """
    Calculate covariance
    where eigenvecs are the columns
    """
    ### Select timeframes (angles) without nan
    omit_rows = np.isnan(angleArray).sum(1)
    print('Skipping %d angles'%(np.sum(omit_rows>1)))
    angleArrayNoNaN = angleArray[omit_rows<1,:]
    C = np.cov(angleArrayNoNaN.T)
    #plt.imshow(C,cmap='viridis')
    eigval, eigvec = np.linalg.eig(C)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    return eigval,eigvec


def resample_data(fpath,X,Y,fname_xyresampled='xy_resampled.npz'):
    """
    Resample X,Y
    """
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    fpath = os.path.join(fpath,fname_xyresampled)
    if os.path.exists(fpath):
        print('\nLoading resampled data from {}'.format(fpath))
        Xi = np.load(fpath)['Xi']
        Yi = np.load(fpath)['Yi']
    else:
        t0 = time.time()
        Xi,Yi = util.XY_resample(X,Y)
        t1 = time.time()
        print('Run time of {}'.format(t1-t0))
        print('\nStoring resampled data as %s'%(fpath))
        np.savez(fpath,Xi=Xi,Yi=Yi)
    return Xi, Yi


def extract_worm(fname_in, fname_dout,store_en=True,
        fname_pdata='pdata.npz'):
    """
    Extract activity of single worm
    """
    # check if file exists:
    if os.path.isfile(os.path.join(fname_dout,fname_pdata)):
        print('File\n{}\nalready exists!\n'.format(
            os.path.join(fname_dout,fname_pdata)))
        au = np.load(os.path.join(fname_dout,fname_pdata))['au']
        return au

    print('\n File does not exist\n')
    print('Loading data {}'.format(fname_in))
    f = h5py.File(fname_in,'r')

    # Get worm posture and length
    X = f['worm']['posture']['skeleton']['x'][()].T
    Y = f['worm']['posture']['skeleton']['y'][()].T
    meanL = np.nanmean(f['worm']['morphology']['length'])

    # Resample data and normalize wrt length
    Xi,Yi = resample_data(fname_dout,X,Y)
    Xi, Yi =Xi/meanL , Yi/meanL

    # Interpolate each position
    if True:
        x = util.nan_interpolate(Yi)
        y = util.nan_interpolate(Xi)
    else:
        x , y = Xi, Yi

    # Get angle from x,y coordinates (zero mean angle)
    angleArray, meanAngles = util.get_arc2angle(x.T,y.T)
    # Rotate worm wrt ventral vs dorsal (invert all angles)
    if '--L_--' in fname_dout:
        angleArray = angleArray*-1

    # Apply PCA to angle covariance
    eigval, eigvec = worm_covariance(angleArray)

    # Calculate amplitudes of motion along PCs
    # Projections of worm shape (angle) onto the low dim space of eigenworms
    eigenAmps = np.dot(eigvec.T,angleArray.T)

    # Remove nans and normalize
    eigenAmpsNoNaN = eigenAmps_clean(eigenAmps)
    au = util.eig_normalization(eigenAmpsNoNaN)

    if store_en:
        np.savez(os.path.join(fname_dout,fname_pdata),
                au=au, eigval=eigval,x=x, y=y,
                eigvec=eigvec, angleArray=angleArray,
                meanAngles=meanAngles,
                eigenAmpsNoNaN=eigenAmpsNoNaN)
    return  au
