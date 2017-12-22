import numpy as np
from scipy import interpolate
import math
from pyhsmm.util.general import rle

def XY_resample(X, Y):
    # Resample wrt arclength st all samples are equally spaced
    # xi = signal.resample(X,numFrames,t)
    # but need to guarantee continuity
    dim1, dim2 = X.shape
    # upsample to 100
    # dim1 = 100
    # Xi = np.zeros([dim1, dim2], dtype=float) * np.nan
    Xi = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan
    Yi = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan
    # Yi = np.zeros([dim1, dim2], dtype=float) * np.nan
    # Interpolate st dx is continuous
    for ii in range(0, dim2):
        if ii % 1000 == 0:
            print('ii =', ii)
        if ~np.isnan(X[0, ii]):
            dx = np.diff(X[:, ii])
            dy = np.diff(Y[:, ii])
            dxy = np.append(0, dx ** 2 + dy ** 2)
            t = np.cumsum(np.sqrt(dxy))
            # equally spaced ts
            ti = np.linspace(t[0], t[-1], dim1)
            f = interpolate.PchipInterpolator(t, X[:, ii])
            Xi[:, ii] = f(ti)
            f = interpolate.PchipInterpolator(t, Y[:, ii])
            Yi[:, ii] = f(ti)
    return Xi, Yi


def nan_interpolate(Xin):
    """
    """
    X = np.copy(Xin)
    # interpolate all nan across worm
    dim1, dim2 = X.shape
    # Xout = np.empty([dim1,dim2],dtype=float) * np.nan
    Xout = np.zeros(shape=(dim1, dim2), dtype=float) * np.nan
    t = np.arange(dim2)
    for ii in range(0, dim1):
        xi = X[ii, :]
        nanx = np.where(np.isnan(xi))  # are nan
        nnanx = np.where(~np.isnan(xi))  # are not nan
        # interpolate nan given not nan
        xi[nanx] = np.interp(t[nanx], t[nnanx], xi[nnanx])
        Xout[ii, :] = xi
    return Xout


def get_arc2angle(x, y):
    """
    """
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    # account for discontinuity
    # atan = np.vectorize(np.arctan2)
    # angles = atan(dy,dx)
    # angles = list(map(np.arctan2, dy, dx))
    angles = list(map(np.arctan2, dy, dx))
    #angles = list(map(np.unwrap,angles))
    #angles = np.apply_along_axis(np.unwrap,1,angles)
    angles = np.unwrap(angles, axis=1) #-1) #-1)
    #rotate angles st mean orientation is zero
    meanAngle = angles.mean(1, keepdims=True)
    AngleArray = angles - meanAngle
    return AngleArray, meanAngle

def reconstructedAngle(eigenAmpsNoNaN,eigenWorms, numDimensions):
    """
    skelX = np.zeros(shape=(numAngles+1, numFrames))
    skelY = np.zeros(shape=(numAngles+1, numFrames))

    reconstructedAngle=np.zeros(shape=(eigenWorms.shape[1],eigenAmpsNoNaN.shape[1]))
    for j in np.arange(eigenAmpsNoNaN.shape[1]):
        for k in np.arange(K1):
            a = eigenAmpsNoNaN[k,j]*eigenWorms[k,:]
            reconstructedAngle[:,j] = reconstructedAngle[:,j]+a.T

    """
    reconstructedAngle = eigenWorms[:, :numDimensions].dot(eigenAmpsNoNaN[:numDimensions, :])
    return reconstructedAngle.T

def angle2skel(angleArray, meanAngle, arclength):
    """
    for frame in np.arange(numFrames):
        skelX[:,frame] = np.insert(np.cumsum(
        np.cos(angleArray[frame,:]+flatAngle[frame]) *arclength/numAngles),0,0)
        skelY[:,frame] = np.insert(np.cumsum(
        np.sin(angleArray[frame,:]+flatAngle[frame]) *arclength/numAngles),0,0)
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


# Rotate coordinates wrt origin
def rotate_origin(x1, y1):
    angle = -1*np.arctan2(y1[-1]-y1[0],x1[-1]-x1[0])#+np.pi
    U = np.asarray([[math.cos(angle),-1*math.sin(angle)],
        [math.sin(angle),math.cos(angle)]])
    a , b = U.dot(np.asarray([x1-x1[0],y1-y1[0]]))
    return a,b


def switch_state_ts(z, state=0, min_dur=30,delta=10):
    # we want to draw aligned all which start at 0 
    
    z2, dr = rle(z)

    duration=np.cumsum(dr)
    state_dr = np.insert(duration,0,0)

    # 1) Get all timestamps at which there is a state change for a given state
    idx = np.where(z2==state)[0]
    idx = idx[idx>0] # only for first state
    print('There are {} transitions to state {}'.format(len(idx),state))
    # 2) check if state lasts at least  min_dur
    idx = idx[dr[idx]>min_dur]
    print('There are {} >{} transitions to state {}'.format(len(idx),min_dur,state))
    # 3) Define state switching timestamps
    trigger_tf = state_dr[idx]
    # trump wrt delta
    trigger_tf = trigger_tf[trigger_tf>delta]
    trigger_tf = trigger_tf[trigger_tf+delta<len(z)]
    print('Output: {} transitions to state {}'.format(len(trigger_tf),state))

    return trigger_tf


def get_coord_tstamps(x_0,y_0,trigger_tf,tfrp=None,delta=10):
    # 4) Get all coordinates of worm some timeframes before and after trigger
    n_tstamps = len(trigger_tf)
    x_0s = np.zeros(shape=(n_tstamps,x_0.shape[0],delta*2+1))
    y_0s = np.zeros(shape=(n_tstamps,y_0.shape[0],delta*2+1))
    # check timestamp
    for ii, tstamp in enumerate(trigger_tf):
        x_0s[ii] = x_0[:,tstamp-delta:tstamp+delta+1]
        y_0s[ii] = y_0[:,tstamp-delta:tstamp+delta+1]
        # center wrt each timeframe - x_0s[ii,0,0]
    if (not (tfrp==None)):
        if tfrp <= x_0s.shape[0]:
            x_0s = x_0s[:tfrp,:,:]
            y_0s = y_0s[:tfrp,:,:]
        else:
            print('Error max number of time stamps is {}'.format(n_tstamps))
    return x_0s, y_0s

