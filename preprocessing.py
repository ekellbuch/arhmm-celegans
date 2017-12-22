import numpy as np
import os
import h5py
import re
import time

# Other
import tmputil as util


def eigenAmps_clean(eigenAmps):
    """
    # drop leading and trailing NaN values
    # (don't want to extrapolate to
    # extreme values)
    """
    if np.isnan(eigenAmps[0,0]):
        # get the end of the starting NaN segment
        afternan = np.where(np.isnan(eigenAmps))[0][-1] + 1
        np.where(eigenAmps[afternan:] == 1)[0][0]
        # drop these values
        eigenAmps[:, 0:nanEnd] = [];
    # is the last point NaN?
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
    Each column is the eigenvector
    """
    ### Select timeframes (angles) without nan
    omit_rows = np.isnan(angleArray).sum(1)
    print('We are going to omit {} angles'.format(
        np.sum(omit_rows>1))) # rows which contain nans
    angleArrayNoNaN = angleArray[omit_rows<1,:]
    C = np.cov(angleArrayNoNaN.T)
    #plt.imshow(C,cmap='viridis')
    eigval, eigvec = np.linalg.eig(C)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    return eigval,eigvec


def resample_data(fpath,X,Y):
    # check if directory exists
    if not os.path.isdir(fpath):
        os.mkdir(fpath)
    ### Resample data
    # check if file has been processed (data resampled?)
    fname_xyresampled ='xy_resampled.npz'

    if os.path.exists(os.path.join(fpath,fname_xyresampled)):
        print('\nLoading resampled data from {}'.format(fpath))
        Xi = np.load(os.path.join(fpath,fname_xyresampled))['Xi']
        Yi = np.load(os.path.join(fpath,fname_xyresampled))['Yi']
    else:
        t0 = time.time()
        Xi,Yi = util.XY_resample(X,Y)
        t1 = time.time()
        print('Run time {}'.format(t1-t0))
        print('\nStoring resampled data in {}'.format(fpath))
        np.savez(os.path.join(fpath,fname_xyresampled),Xi=Xi,Yi=Yi)
    return Xi, Yi


def extract_worm(fname_in, fname_dout,store_en=True,fname_pdata='pdata.npz'):

    # check if file already exists:
    if os.path.isfile(os.path.join(fname_dout,fname_pdata)):
        print('File\n{}\nalready exists!\n'.format(
            os.path.join(fname_dout,fname_pdata)))
        au = np.load(os.path.join(fname_dout,fname_pdata))['au']
        return au

    print('\n File does not exist\n')
    print('Loading data {}'.format(fname_in))
    f = h5py.File(fname_in,'r')

    ### get worm posture
    X = f['worm']['posture']['skeleton']['x'][()].T
    Y = f['worm']['posture']['skeleton']['y'][()].T

    ### worm length
    meanL = np.nanmean(f['worm']['morphology']['length'])

    # interpolate wrt angle array
    # might have to look at video-tracking?
    interp_pos = 1 # always

    ### Resample data
    Xi,Yi = resample_data(fname_dout,X,Y)
    ### Normalize worm wrt total length
    Xi, Yi =Xi/meanL , Yi/meanL

    ### Interpolate the positions to find the angle array
    if interp_pos:
        x = util.nan_interpolate(Yi)
        y = util.nan_interpolate(Xi)
    else:
        x , y = Xi, Yi

    ### Get tangent angle for each frame and rotate to have zero mean angle
    angleArray, meanAngles = util.get_arc2angle(x.T,y.T)
    ### ROTATE WORM if the worm in the ventral vs dorsal (invert all angles)
    if '--L_--' in fname_dout:
        angleArray = angleArray*-1

    ### Calculate C(s,s'), eigvectors and eigvals and sort them
    eigval, eigvec = worm_covariance(angleArray)

    ### Calculate amplitudes of motion along PCs
    # Projections of worm shape (angle) onto the low dim space of eigenworms
    eigenAmps = np.dot(eigvec.T,angleArray.T)

    ### Clean eigenAmps
    eigenAmpsNoNaN = eigenAmps_clean(eigenAmps)
    ### Normalize
    au = util.eig_normalization(eigenAmpsNoNaN)

    if store_en:
        np.savez(os.path.join(fname_dout,fname_pdata), au=au, eigval=eigval,
                x=x, y=y, eigvec=eigvec, angleArray=angleArray, meanAngles=meanAngles,
                eigenAmpsNoNaN=eigenAmpsNoNaN)

    return  au
