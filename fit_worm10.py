# Read in worm files
# Modified from https://github.com/slinderman/pyhsmm_spiketrains/

import os
import time
import gzip
import pickle

import numpy as np
from collections import namedtuple

import brewer2mpl
import itertools

from pybasicbayes.util.text import progprint_xrange
import pyhsmm
import autoregressive.distributions as d
import autoregressive.models as m
from pybasicbayes.distributions import AutoRegression
#
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

def load_aus_data(fpath, num_pcs=4,trainfrac=0.8):
    """
    Load aus from fpath
    """
    data_sets = np.load(fpath)['aus']
    time_frames = np.load(fpath)['time_frames']
    number_datasets = len(time_frames)

    shortest_data   = min(time_frames)
    train_frames    = int(trainfrac*shortest_data)
    train_data, test_data = [] , []
    for data in data_sets:
        train_data.append(data[:num_pcs,:train_frames].T)
        test_data.append(data[:num_pcs,train_frames:shortest_data].T)
    return train_data, test_data


Results = namedtuple( "Results", ["name", "loglikes", "predictive_lls", "predictive_lls2",
                "N_used", "alphas", "gammas",
                "samples", "timestamps"])


def get_empirical_ar_params(train_datas, params):
    """
    Estimate the parameters of an AR observation model
    by fitting a single AR model to the entire dataset.
    """
    assert isinstance(train_datas, list) and len(train_datas) > 0
    datadimension = train_datas[0].shape[1]
    assert params["nu_0"] > datadimension + 1

    # Initialize the observation parameters
    obs_params = dict(nu_0=params["nu_0"],
                      S_0=params['S_0'],
                      M_0=params['M_0'],
                      K_0=params['K_0'],
                      affine=params['affine'])

    # Fit an AR model to the entire dataset
    obs_distn = AutoRegression(**obs_params)
    obs_distn.max_likelihood(train_datas)

    # Use the inferred noise covariance as the prior mean
    obs_params["S_0"] = obs_distn.sigma * (params["nu_0"] - datadimension - 1)
    obs_params["M_0"] = obs_distn.A.copy()

    return obs_params

def fit(name, model, test_data, N_iter=1000, init_state_seq=None):
    def evaluate(model):
        ll = model.log_likelihood()
        if isinstance(test_data, list) and len(test_data) > 0:
            pll = 0
            pll2 = []
            for cdata in range(len(test_data)):
                pll += model.log_likelihood(test_data[cdata])
                pll2.append(model.log_likelihood(test_data[cdata]))
        N_used = len(list(model.used_states))
        trans = model.trans_distn
        alpha = trans.alpha
        gamma = trans.gamma if hasattr(trans, "gamma") else None
        return ll, pll, pll2, N_used, alpha, gamma

    def sample(model):
            tic = time.time()
            model.resample_model()
            timestep = time.time() - tic
            return evaluate(model), timestep

    #### Initialize with given state seq
    if init_state_seq is not None:
        model.states_list[0].stateseq = init_state_seq
        for _ in xrange(100):
            model.resample_obs_distns()

    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(N_iter)])
    lls, plls, plls2, N_used, alphas, gammas = \
            zip(*((init_val,) + vals))
    timestamps = np.cumsum((0.,) + timesteps)
    return Results(name, lls, plls, plls2, N_used, alphas, gammas,
            model.copy_sample(), timestamps)


def make_joint_models(train_datas, Nmax = 10):
    # Define a sequence of models
    if isinstance(train_datas, list) and len(train_datas) > 0:
        data = train_datas
        num_worms = len(train_datas)
    else:
        data = [train_datas]
        num_worms = 1
    print('Making models')
    names_list = []
    fnames_list = []
    hmm_list = []
    color_list = []
    method_list = []
    # Standard AR model (Scale resampling)
    D_obs = data[0].shape[1]
    print('D_obs shape {}'.format(data[0].shape[1]))
    affine = True
    nlags = 1
    init_state_distn = 'uniform'

    # Construct a standard AR-HMM for fitting
    # with just one worm
    obs_hypers = dict(
        nu_0 = D_obs+2,
        S_0 = np.eye(D_obs),
        M_0 = np.hstack((np.eye(D_obs), np.zeros((D_obs, D_obs*(nlags-1)+affine)))),
        K_0 = np.eye(D_obs*nlags+affine),
        affine = affine)

    # Joint model - fitting all worm at a time
    # Fit range of parameters for each state
    state_array = [6,1,8,10,12,15,2,4]
    alpha_array = [5.0, 10.0, 100.0]
    gamma_array = [5.0, 10.0, 100.0]
    kappa_array = 10**np.arange(2,12)[::2]

    # Vary the hyperparameters of the scale resampling model
    for num_states, alpha_a_0, gamma_a_0, kappa_a_0 in itertools.product(state_array,
            alpha_array,gamma_array,kappa_array):
        # using data of all worms
        obs_hypers = get_empirical_ar_params(data, obs_hypers)
        obs_distns = [d.AutoRegression(**obs_hypers) for state in range(num_states)]
        names_list.append("AR-HMM (Scale)")
        fnames_list.append("ar_scale_wormall_states%.1f_alpha%.1f_gamma%.1f_kappa%.1f"%(num_states,
            alpha_a_0, gamma_a_0, kappa_a_0))
        color_list.append(allcolors[1])
        # Init Model Param
        hmm = m.ARWeakLimitStickyHDPHMM(
                # sampled from 1d finite pmf
                alpha=alpha_a_0, gamma=gamma_a_0,
                init_state_distn=init_state_distn,
                # create A, Sigma
                obs_distns=obs_distns,
                kappa = kappa_a_0 # kappa
                )
        # Add data of each worm
        for cworm in np.arange(num_worms):
            hmm.add_data(data[cworm])
        # Append model  and store
        hmm_list.append(hmm)
        method_list.append(fit)
    print('Finished making baseline models')
    return names_list, fnames_list, color_list, hmm_list, method_list

def run_experiment():
    # change directory to andrebrown
    parent_directory = os.getcwd()
    # Set output directory
    dir_result = 'results_strains'
    fname_dout = os.path.join(parent_directory,dir_result)
    runnum = 10
    output_dir = os.path.join(fname_dout, "run%03d" % runnum)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created directory %s'%(output_dir))
    # Set output parameters
    num_pcs = 4
    trainfrac = 0.7
    # Load the data
    # INPUT FILE
    file_in = os.path.join(fname_dout,'joint_aus_N2.npz')
    #
    print('Loading data {}'.format(file_in))
    train_data , test_data = load_aus_data(file_in, num_pcs=num_pcs, trainfrac=trainfrac)
    print ("Running Experiment")
    if (isinstance(train_data, list)) and len(train_data)> 0:
        num_worms = len(train_data)
        print('Number of files: %d'%(num_worms))
        print ("T_train: {}".format(train_data[0].shape[0]))
        print ("T_test: {}".format(test_data[0].shape[0]))
    else:
        num_worms = 1
        print ("T_train: {}\t".format(train_data.shape[0]))
        print ("T_test: {}\t".format(test_data.shape[0]))

    # Parameters

    Nmax = 6
    # Define a set of HMMs
    names_list = []
    fnames_list = []
    color_list = []
    model_list = []
    method_list = []

    # Add HDP_HMMs
    nl, fnl, cl, ml, mthdl = make_joint_models(train_data, Nmax = Nmax)
    names_list.extend(nl)
    fnames_list.extend(fnl)
    color_list.extend(cl)
    model_list.extend(ml)
    method_list.extend(mthdl)
    # Fit the models with Gibbs sampling
    N_iter = 1000

    ttotal = len(model_list)
    counter = 0
    for model_name, model_fname, model, method in \
            zip(names_list, fnames_list, model_list, method_list):
        print("Looking at model {} out of {}".format(counter,ttotal))
        print ("Model: ", model_name)
        print ("File:  ", model_fname)
        print ("")
        output_file = os.path.join(output_dir, model_fname + ".pkl.gz")

        # Check for existing results
        if os.path.exists(output_file):
            print ("Results already exist at: ", output_file)
            print ("")
        else:
            res = method(model_name, model, test_data, N_iter=N_iter)
            # Store results
            with gzip.open(output_file, "w") as f:
                print ("Saving results to: ", output_file)
                pickle.dump(res, f, protocol=-1)
                print ("")
        counter+=1

if __name__ == "__main__":
    run_experiment()
