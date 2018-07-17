import numpy as np

from pyhsmm.util.general import relabel_by_permutation, rle
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


def relabel_model_z(model, index=0, plot_en=False):
    """
    Relabel state from model by permutation
    """
    Nmax = model.num_states
    perm = np.argsort(model.state_usages)[::-1]
    z = relabel_by_permutation(model.states_list[index].stateseq,
            np.argsort(perm))
    if plot_en:
        plt.bar(np.arange(Nmax), np.bincount(z, minlength=Nmax))
        plt.show()
    return z , perm


def plot_pcs_slice(self,data_in,large_slice,plot_slice=None,
        num_pcs=4,indiv=0, color_array=None,fs=30,sz=4):
    """
    Plot PC slices and states
    """
    if color_array == None:
        color_array = self._get_colors()
    # Plot params
    fig = plt.figure(figsize=(sz,6))
    gs = GridSpec(sz+len(self.states_list),1)
    feature_ax = plt.subplot(gs[:sz,:])
    data_in =  data_in[:num_pcs,large_slice][::-1,:]
    max_ = ceil(data_in.max()-data_in.min()) + 1
    ttime = np.arange(data_in.shape[1])
    for ii in range(0,num_pcs):
        feature_ax.plot(ttime,data_in[ii,:]+ii*max_,'k')
    feature_ax.set_yticks(np.arange(num_pcs)*max_)
    feature_ax.set_yticklabels('')

    feature_ax.set_ylim((data_in.min()-1,num_pcs*max_-1))

    xlabel_= np.linspace(0,data_in.shape[1],5,dtype='int')
    feature_ax.set_xticks(xlabel_)
    feature_ax.set_xlim((xlabel_[0],xlabel_[-1]))
    feature_ax.set_xticklabels(list(map(str,xlabel_ // fs)))

    if not (plot_slice is None):
        feature_ax.axvline(plot_slice[0], color=color_array[0],linestyle=':',lw=2)
        feature_ax.axvline(plot_slice[-1], color=color_array[0],linestyle=':',lw=2)
        plot_pcs_slice_sub(self,data_in,large_slice,plot_slice,indiv,color_array)
    return


def plot_pcs_slice_sub(self,data_in,large_slice,plot_slice,
        indiv=0,color_array=None,sz=8):
    """
    Plot short PC slice
    """
    fig = plt.figure(figsize=(sz,6))
    gs = GridSpec(sz+len(self.states_list),1)
    feature_ax = plt.subplot(gs[:sz,:])
    stateseq_ax = plt.subplot(gs[sz+1])

    if color_array is None:
        color_array = self._get_colors()

    r_plot_slice = list(map(lambda x: large_slice[0] + x, plot_slice))
    z, perm = relabel_model_z(self,index=indiv)
    z = z[r_plot_slice]
    stateseq_norep, durations = rle(z)

    max_ = ceil(data_in.max()-data_in.min()) +1
    data_in=data_in[:,plot_slice]
    ttime = np.arange(data_in.shape[1])
    for ii in range(0,data_in.shape[0]):
        feature_ax.plot(ttime,data_in[ii,:] + ii*max_,'k')

    feature_ax.set_xlim((0,len(plot_slice)))
    feature_ax.set_ylim((data_in.min()-1,data_in.shape[0]*max_-1))
    feature_ax.set_yticks([])
    feature_ax.set_xticks([])

    stateseq_ax.imshow(z[:,np.newaxis].T,aspect='auto',
            cmap=ListedColormap(color_array),vmin=0,vmax=len(perm))
    stateseq_ax.set_yticks([])
    stateseq_ax.set_xticks([])

    for ii, pos in enumerate(durations.cumsum()):
        if durations[ii] >=1:
            feature_ax.axvline(pos,
                    color=color_array[stateseq_norep[ii]],
                    linestyle=':')
    return


def state_correlation(z1,z2):
    """
    Calculate state correlation
    """
    dim1, dim2= z1.max()+1, z2.max()+1
    fig = plt.figure(figsize=(dim1,dim2))
    C = np.zeros(shape=(dim1,dim2))

    for i in np.arange(dim1):
        for j in np.arange(dim2):
            C[i,j]=np.logical_and(z1==i,z2==j).sum()

    C_pr = C.copy()
    for col in np.arange(dim2):
        if C[:,col].sum()!=0:
            C_pr[:,col]=C[:,col]/C[:,col].sum()

    avg_ind = np.sum(C_pr * np.arange(dim1)[:,None], axis=0)
    perm = np.argsort(avg_ind)
    ends_ = (avg_ind==0).sum()
    new_l = perm.copy()
    new_l[0:dim2-ends_] = perm[ends_:]
    new_l[dim2-ends_:] = perm[0:ends_]
    ax = fig.add_subplot(111)
    Cpr = C_pr[:,new_l]
    im = ax.imshow(Cpr,cmap='Reds',
            vmin=Cpr.min(),vmax=Cpr.max())
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel(r'$K = %d$'%dim1)
    ax.set_xlabel(r'$K = %d$'%dim2)

    divider = make_axes_locatable(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal",
            spacing='uniform', format='%.2f',
            ticks=np.linspace(Cpr.min(),Cpr.max(),5))
    plt.tight_layout()
    plt.show()
    return
