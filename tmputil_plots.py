## Plot PCs and subset of segmentation
import numpy as np

from pyhsmm.util.general import relabel_by_permutation, rle
import math
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


#####################
# STATE DYNAMICS
#####################

def relabel_model_z(model, index=0, plot_en=False):
    """
    Relabel model by permutation

    """
    Nmax = model.num_states
    perm = np.argsort(model.state_usages)[::-1]
    z = relabel_by_permutation(model.states_list[index].stateseq, np.argsort(perm))
    if plot_en:
        plt.bar(np.arange(Nmax), np.bincount(z, minlength=Nmax))
        plt.show()
    return z , perm


def plot_pcs_slice(self,data_in,large_slice,plot_slice=None,num_pcs=4,indiv=0, color_array=None):
    """
    Plot PC slides and states according to tile
    """
    fs = 30
    print('Assuming {} Hz'.format(fs))
    if color_array == None:
        color_array = self._get_colors()
    # Plot params
    sz = 4
    fig = plt.figure(figsize=(sz,6))
    gs = GridSpec(sz+len(self.states_list),1)
    feature_ax = plt.subplot(gs[:sz,:])
    # Plot input data
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


def plot_pcs_slice_sub(self,data_in,large_slice,plot_slice,indiv=0,color_array=None):
    """
    self model
    data_in
    large_slice
    plot_slice
    indiv data size
    """
    sz = 8
    fig = plt.figure(figsize=(sz,6))
    gs = GridSpec(sz+len(self.states_list),1)
    feature_ax = plt.subplot(gs[:sz,:])
    stateseq_ax = plt.subplot(gs[sz+1])

    #update = True
    if color_array is None:
        color_array = self._get_colors()

    r_plot_slice = list(map(lambda x: large_slice[0] + x, plot_slice))
    # slice wrt original slice
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

    stateseq_ax.imshow(z[:,np.newaxis].T,aspect='auto',cmap=ListedColormap(color_array),vmin=0,vmax=len(perm))
    stateseq_ax.set_yticks([])
    stateseq_ax.set_xticks([])

    # Add vertical lines
    for ii, pos in enumerate(durations.cumsum()):
            if durations[ii] >=1:
                feature_ax.axvline(pos,color=color_array[stateseq_norep[ii]],linestyle=':')
    return

#####################
# EIGENWORMS
#####################

def plot_joint_pd_au(au,cmap='viridis'):
    """
    Plot joint pd of au values
    """
    numBins = 100
    bins = np.linspace(-2, 2,numBins+1)
    marginals = []
    for pc1 in range(0,4):
        for pc2 in range(pc1+1, 4):
            H, xedges, yedges= np.histogram2d(au[pc1],au[pc2],normed=True,bins=bins)
            plt.title('PC {} and PC {}'.format(pc1+1,pc2+1))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            imgplot=plt.imshow(H, interpolation='nearest', clim=(0,0.3),#H.min(), H.max()) ,\
                       extent=[-2.5, 2.5, -2.5, 2.5] )
            marginals.append(H.sum(1))
            imgplot.set_cmap(cmap)
            plt.colorbar()
            plt.axis((-2,2,-2,2))
            plt.xlabel("$a_{%d}$"%(pc1+1))
            plt.ylabel("$a_{%d}$"%(pc2+1))
            plt.show()
    plt.figure()
    plt.plot(marginals[0])
    plt.plot(marginals[1])
    plt.plot(marginals[2])
    return


def state_correlation(z1,z2):
    """
    Distribution of data with respect to two different states
    """
    dim1 = z1.max()+1
    dim2 = z2.max()+1

    fig = plt.figure(figsize=(dim1,dim2))

    C = np.zeros(shape=(dim1,dim2))

    for i in np.arange(dim1):
        for j in np.arange(dim2):
            C[i,j]=np.logical_and(z1==i,z2==j).sum()

    #C_pr = C / C.sum(0,keepdims=True)
    C_pr = C.copy()
    for col in np.arange(dim2):
        if C[:,col].sum()!=0:
            C_pr[:,col]=C[:,col]/C[:,col].sum()

    avg_ind = np.sum(C_pr * np.arange(dim1)[:,None], axis=0)
    perm = np.argsort(avg_ind)
    #print(perm)
    ends_ = (avg_ind==0).sum()
    new_l = perm.copy()
    new_l[0:dim2-ends_] = perm[ends_:]
    new_l[dim2-ends_:] = perm[0:ends_]
    #print(new_l)
    perm = new_l
    ax = fig.add_subplot(111)
    Cplot = C_pr # C
    Cplot = Cplot[:,perm]
    im = ax.imshow(Cplot,cmap='Reds',\
            vmin=Cplot.min(),vmax=Cplot.max())
    labels= list(map(str,perm))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel(r'$K = %d$'%dim1)
    ax.set_xlabel(r'$K = %d$'%dim2)

    divider = make_axes_locatable(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal", \
        spacing='uniform', format='%.2f',ticks=np.linspace(Cplot.min(),Cplot.max(),5))
    plt.tight_layout()
    #print(Cplot.sum(0).sum())
    #print(Cplot.sum(1).sum())
    return
