import numpy as np
from scipy import interpolate
import math
from pyhsmm.util.general import rle


def XY_resample(X, Y):
    """
    Resample X,Y such that
    all samples are equally spaced and
    first order derivative (arc) is continuous
    """
    dim1, dim2 = X.shape
    Xi = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan
    Yi = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan

    # Interpolate st dx is continuous
    for ii in range(0, dim2):
        if ii % 1000 == 0:
            print('Interpolating sample %d'%ii)
        if ~np.isnan(X[0, ii]):
            dx = np.diff(X[:, ii])
            dy = np.diff(Y[:, ii])
            dxy = np.append(0, dx ** 2 + dy ** 2)
            t = np.cumsum(np.sqrt(dxy))
            # equally spaced sampling
            ti = np.linspace(t[0], t[-1], dim1)
            f = interpolate.PchipInterpolator(t, X[:, ii])
            Xi[:, ii] = f(ti)
            f = interpolate.PchipInterpolator(t, Y[:, ii])
            Yi[:, ii] = f(ti)
    return Xi, Yi


def nan_interpolate(Xin):
    """
    Linearly interpolate missing (NaN) frames
    """
    X = Xi.copy()
    dim1, dim2 = X.shape
    Xout = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan
    t = np.arange(dim2)
    for ii in range(0, dim1):
        xi = X[ii, :]
        nanx = np.where(np.isnan(xi))
        nnanx = np.where(~np.isnan(xi))
        xi[nanx] = np.interp(t[nanx], t[nnanx], xi[nnanx])
        Xout[ii, :] = xi
    return Xout


def get_arc2angle(x, y):
    """
    Calculate angle between x,y coordinates
    """
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    # account for discontinuity
    angles = list(map(np.arctan2, dy, dx))
    angles = np.unwrap(angles, axis=1)
    #rotate angles st mean orientation is zero
    meanAngle = angles.mean(1, keepdims=True)
    AngleArray = angles - meanAngle
    return AngleArray, meanAngle


def reconstructedAngle(eigenAmpsNoNaN,eigenWorms, numDimensions):
    """
    Reconstruct angle
    """
    reconstructedAngle = eigenWorms[:, :numDimensions].dot(eigenAmpsNoNaN[:numDimensions, :])
    return reconstructedAngle.T


def angle2skel(angleArray, meanAngle, arclength):
    """
    Calculate skeleton given angle array
    """
    numAngles, numFrames = angleArray.T.shape
    skX0 = np.cumsum(np.cos(angleArray+meanAngle)*arclength/numAngles, axis=1)
    skY0 = np.cumsum(np.sin(angleArray+meanAngle)*arclength/numAngles, axis=1)
    skelX = np.column_stack((np.zeros(numFrames), skX0)).T
    skelY = np.column_stack((np.zeros(numFrames), skY0)).T
    return skelX, skelY


def eig_normalization(eigenworms):
    """
    Estimator of the second moment : for mean = 0, var = 1
    """
    norm_eworms = (eigenworms-eigenworms.mean(1, keepdims=True)) \
        / eigenworms.std(1, keepdims=True)
    return norm_eworms


def rotate_origin(x1, y1):
    """
    Rotate coordinates wrt origin
    """
    angle = -1*np.arctan2(y1[-1]-y1[0],x1[-1]-x1[0])
    U = np.asarray([[math.cos(angle),-1*math.sin(angle)],
        [math.sin(angle),math.cos(angle)]])
    a , b = U.dot(np.asarray([x1-x1[0],y1-y1[0]]))
    return a,b


def switch_state_ts(z, state=0, min_dur=30,delta=10):
    """
    Draw aligned all which start at 0
    """
    z2, dr = rle(z)

    duration=np.cumsum(dr)
    state_dr = np.insert(duration,0,0)

    # Extract timestamps @ there is a state change
    idx = np.where(z2==state)[0]
    idx = idx[idx>0] # only for first state
    print('There are %d transitions to state %d'%(len(idx),state))
    # Enforce min state duration
    idx = idx[dr[idx]>min_dur]
    print('There are %d > %d transitions to state %d'%(len(idx),min_dur,state))
    # Extract idx of state transitions (+- delta)
    trigger_tf = state_dr[idx]
    trigger_tf = trigger_tf[trigger_tf>delta]
    trigger_tf = trigger_tf[trigger_tf+delta<len(z)]
    print('Final: %d transitions to state %d'%(len(trigger_tf),state))
    return trigger_tf


def get_coord_tstamps(x_0,y_0,trigger_tf,tfrp=None,delta=10):
    """
    Extract coordinates before/after trigger (index)
    """
    n_tstamps = len(trigger_tf)
    x_0s = np.zeros(shape=(n_tstamps,x_0.shape[0],delta*2+1))
    y_0s = np.zeros(shape=(n_tstamps,y_0.shape[0],delta*2+1))
    # check timestamp
    for ii, tstamp in enumerate(trigger_tf):
        x_0s[ii] = x_0[:,tstamp-delta:tstamp+delta+1]
        y_0s[ii] = y_0[:,tstamp-delta:tstamp+delta+1]
    if (not (tfrp==None)):
        if tfrp <= x_0s.shape[0]:
            x_0s = x_0s[:tfrp,:,:]
            y_0s = y_0s[:tfrp,:,:]
        else:
            print('Error: Exceeded %d available frames'%n_tstamps)
    return x_0s, y_0s
