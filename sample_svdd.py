# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:31:41 2016
This script runs under python3.
Code written to accompany sampling based svdd paper.
"""

#  Some general comments.
# 1. One Class SVM formulation (OCSVM) is identical to the SVDD formulation for the Gaussian Kernel.
# 2. The feasible set  for the optimization in SVDD/OCSVM computation is
#        0 <= alpha_i <= 1/(n * f),
#        \sum a\lpha_i = 1,
#    which is equivalent to
#       0 <= alpha_i <= min(1,1/(n*f)),
#       \sum alpha_i = 1.
#    So a value of f less than 1/n can be replaced by 1/n. For some reason explicitly replacing f
#    gives much better results than passing in tiny values of f.
# For the paper we used the C++ SVDD implementation from LIBSVM here:
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#libsvm_for_svdd_and_finding_the_smallest_sphere_containing_all_data.
# Even though Scikit-learn's OCSVM implementation is also based on LIBSVM, their performance characteristics are different.
#  LIBSVM probably uses different solvers for OCSVM and SVDD, and in may cases this python OCSVM implmentation outperformed
# the SVDD one significantly.


from collections import namedtuple
import numpy as np
from numpy.random import choice
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel

#Compute the radius and center from a svdd result
def _compute_radius_center(clf, method=1):
    sv, coef = clf.support_vectors_, clf.dual_coef_
    sv_pos = np.where((coef < 1)[0, ...])[0]
    coef.shape = (coef.shape[1], )
    coef = coef/np.sum(coef)
    center = np.dot(coef, sv)
    #method 1 is a fast approximation of the radius which is good enough for our purpose
    if method == 0:
        m = rbf_kernel(sv, sv, gamma=clf.gamma)
        radius = 1 - 2 * np.dot(m[sv_pos[0], ...], coef) + np.dot(coef, np.dot(m, coef))
    else:
        v = sv[sv_pos[0], ...].reshape(1, sv.shape[1])
        m = rbf_kernel(v, sv, gamma=clf.gamma)
        radius = 1 - np.dot(m, coef)
    return radius, center

#compute svdd given the indices of the sample 
def _do_one_class_svm_sample(gamma, nu, x_train, sample_indices, compute_rc=True):
    x_train_sample = x_train[sample_indices, ...]
    nsample = x_train_sample.shape[0]
    nu_1 = nu  if nu * nsample > 1 else 1/nsample
    clf = svm.OneClassSVM(gamma=gamma, nu=nu_1)
    clf.fit(x_train_sample)
    if compute_rc:
        radius, center = _compute_radius_center(clf)
        return sample_indices[clf.support_], radius, center
    else:
        return sample_indices[clf.support_]

# draw a random sample from the original data and peform svdd on it
def _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=True):
    sample = choice(x_train.shape[0], sample_size)
    return _do_one_class_svm_sample(gamma, nu, x_train, sample, compute_rc=compute_rc)


# the sampling svdd implementation, see the __main__ section for an example
def sample_svdd(x_train,
                outlier_fraction=0.001,
                kernel_s=2,
                maxiter=1000,
                sample_size=10,
                resample_n=3,
                stop_tol=1e-6,
                n_iter=30,
                iter_history=True,
                seed=2513646):
    """
    Perform sampling based approximate svdd.
    Input Parameters:
        x_train : input data to train, must be a two-dim numpy array 
        kernel_s: the bandwidth for the Gaussian kernel, the Gaussian kernel is 
                  assumed to be of the form exp( -||x - y||^2 / (2 *kernel_s^2))
        sample_size: the size of each random sample 
        resample_n: take these many samples in each iteration, and merge the union of their support vectors with the
                    master, the method documented in the paper corresponds to resample_n = 1
        stop_tol: the tolerance value to detect convergence
        n_iter: the raidus and center must be close to each other for this many consecutive iterations
                for convergence to be declared
        iter: flag to determine whether convergence history will be stored
        seed: seed value for the random number generator    
    Output:
        The output is a named tuple. If the output is denoted by res then:
            res.IterHist: a named tuple containing the iteration history
                res.IterHist.niter_ : number of iterations till convergence
                res.IterHist.radius_history_ : the iteration history for the radius
                res.IterHist.center_history_: the iteration history of the center
                res.IterHist.converged_ : convergence status flag
            res.Params: a named tuple containing the output parameters of the suggested SVDD 
                res.Params.sv_: the indices of the fitted support vectors
                res.Params.center_: final center point
                res.Params.radius_ : final radius
            res.OneClassSVM:
                A sklearn.svm.OneClassSVM instance corresponding to the result. Can be used for scoring.                                
    """    
    
    # Only matrix input allowed
    if len(x_train.shape) != 2:
        print("ERROR: invalid x_train input found, expecting a matrix")
        raise ValueError

    #sanity checks
    if maxiter <= 0:
        print("ERROR: maxiter must be positive integer")
        raise ValueError

    nobs = x_train.shape[0]

    if nobs <= sample_size:
        print("ERROR: sample size must be strictly smaller than number of observations in input data")
        raise ValueError

    # convert kernel_s to gamma
    gamma, nu = 0.5/(kernel_s*kernel_s), outlier_fraction

    if np.isfinite(gamma) != True or np.isfinite(nu) != True or (nu < 0) or (nu > 1):
        print("ERROR: Invalid kernel_s or outlier_fraction input")
        raise ValueError

    #if negative seed is provided use a system chosen seed
    np.random.seed(seed=seed if seed >= 0 else None)

    if iter_history:
        radius_history, center_history = np.empty(maxiter+1), list()

    clf = None
    sv_ind_prev, radius_prev, center_prev = _do_one_class_svm_random(gamma, nu, x_train, sample_size)

    if iter_history:
        radius_history[0] = radius_prev
        center_history.append(center_prev)

    i, converged, iter_n = 0, 0, 0
    while i < maxiter:
        if converged: break

        sv_ind_local = _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=False)
        for dummy1 in range(resample_n-1):
            sv_ind_locals = _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=False)
            sv_ind_local = np.union1d(sv_ind_locals, sv_ind_local)

        sv_ind_merge = np.union1d(sv_ind_local, sv_ind_prev)
        sv_ind_master, radius_master, center_master = _do_one_class_svm_sample(gamma, nu, x_train, sv_ind_merge)


        if iter_history:
            radius_history[i+1] = radius_master
            center_history.append(center_master)

        iter_n = iter_n + 1 if np.fabs(radius_master - radius_prev) <= stop_tol * np.fabs(radius_prev) else 0
        if iter_n >= n_iter:
            converged = 1
        else:
            sv_ind_prev, center_prev, radius_prev = sv_ind_master, center_master, radius_master
        i += 1

    if iter_history:
        radius_history = radius_history[0:i+1]
    niter = i + 1

    SampleSVDDRes      = namedtuple("SampleSVDDRes", "Params  IterHist OneClassSVM")
    SampleSVDDParams   = namedtuple("SampleSVDDParams", "sv_ center_ radius_")
    SampleSVDDIterHist = namedtuple("SampleSVDDIterHist", "niter_ radius_history_ center_history_ converged_")

    params = SampleSVDDParams(sv_ind_master, center_master, radius_master)

    iterhist = None
    if iter_history:
        iterhist = SampleSVDDIterHist(niter, radius_history, center_history, converged)

    nsv = sv_ind_master.shape[0]
    clf = svm.OneClassSVM(gamma=gamma, nu=nu if nu * nsv > 1 else 1./nsv)
    clf.fit(x_train[sv_ind_master, ...])

    return SampleSVDDRes(params, iterhist, clf)

if __name__ == "__main__":
    def run_main():
        import matplotlib.pyplot as plt
        import time
        #create a donut data.
        def one_donut(rmin, rmax, origin, nobs):
            """
                rmin: inner radius
                rmax: outer radis
                origin: origin
                nobs: number of observations in the data
            """
            r = np.sqrt(rmin*rmin + (rmax - rmin) * (rmax + rmin) * np.random.ranf(nobs))
            theta = 2 * np.pi * np.random.ranf(nobs)
            res = np.array([(r_*np.cos(theta_), r_*np.sin(theta_)) for r_, theta_ in zip(r, theta)])
            return res + origin

        seed = 24215125
        np.random.seed(seed)
        
        #store time taken by the two methods
        tsample, tfull = list(),list()
        
        #run the method over data sets of these sizes
        dsize_list = [5000,10000,100000,500000,1000000,1250000,2000000] 
        
        #this will take about 10mins to run
        for ndat in dsize_list:
            
            #parameters of the two donuts           
            r_min1, r_max1, origin1, nobs1 = 3, 5, (0, 0), np.floor(0.75 * ndat)
            r_min2, r_max2, origin2, nobs2 = 2, 4, (10, 10), ndat - nobs1
    
            #create the training data        
            test_data = np.append(one_donut(r_min1, r_max1, origin1, nobs1), one_donut(r_min2, r_max2, origin2, nobs2), axis=0)
            
            print('the test data has {0} observations'.format(test_data.shape[0]))
            
            #parameters of the training SVDD. Tweak for performance/accuracy.
            outlier_fraction, kernel_s = 0.0001, 1.3
            sample_size, resample_n, n_iter = 10, 2, 15
            stop_tol, maxiter = 1e-4, 5000
            
            #train using sampling svdd
            start = time.time()
            result = sample_svdd(test_data,
                                 outlier_fraction=outlier_fraction, 
                                 kernel_s=kernel_s, 
                                 resample_n=resample_n, 
                                 maxiter=maxiter, 
                                 sample_size=sample_size, 
                                 stop_tol=stop_tol, 
                                 n_iter=n_iter, 
                                 iter_history=True, 
                                 seed=seed)
            end = time.time()
            tsample.append( end-start )
            print("sample svdd took {0} seconds to train, iteration history stored".format(end-start))
            radius_history = result.IterHist.radius_history_
            sv_indices = result.Params.sv_
    
            #train using full svdd
            start = time.time()
            clf1 = svm.OneClassSVM(nu=outlier_fraction if test_data.shape[0] * outlier_fraction > 1 else 1./test_data.shape[0], kernel="rbf", gamma=0.5/(kernel_s*kernel_s))
            clf1.fit(test_data)
            end = time.time()
            tfull.append(end-start)
            print("full svdd took {0} seconds to train".format(end-start))
    
            
            #plot the support vectors
            plt.figure(1)
            plt.grid(True)
            plt.title('Support Vectors (Sampling Method)')
            plt.scatter(test_data[sv_indices, 0], test_data[sv_indices, 1])
            plt.show()
    
            plt.figure(2)
            plt.grid(True)
            plt.title('Support Vectors (Full SVDD))')
            plt.scatter(clf1.support_vectors_[..., 0], clf1.support_vectors_[..., 1])
            plt.show()
    
            plt.figure(3)
            plt.title('Iteration History for Sampling Method')
            plt.plot(radius_history)
            plt.show()
    
            #create a 200 x 200 grid on the bounding rectangle of the training data
            # for scoring
            ngrid=200
            max_x, max_y = np.amax(test_data, axis=0)
            min_x, min_y = np.amin(test_data, axis=0)
    
            x_ = np.linspace(min_x, max_x, ngrid)
            y_ = np.linspace(min_y, max_y, ngrid)
    
            x, y = np.meshgrid(x_, y_)
    
            score_data = np.array([(x1, y1) for x1, y1 in zip(x.ravel(), y.ravel())])
    
            #the OneClasSVM result corresponding to the sample data
            clf2 = result.OneClassSVM
    
            scores1 = clf1.predict(score_data)
            scores2 = clf2.predict(score_data)
            
            #plot the scored data
            plt.figure(4)
            p2 = np.where(scores2 == 1)
            plt.grid(True)
            plt.title("Scoring Results : Inside Points Colored green (using sampling svdd)")
            plt.scatter(score_data[p2, 0], score_data[p2, 1], color='g', s=0.75)
            plt.show()
    
            plt.figure(5)
            p1 = np.where(scores1 == 1)
            plt.grid(True)
            plt.title("Scoring Results : Inside Points Colored (using full svdd)")
            plt.scatter(score_data[p1, 0], score_data[p1, 1], color='g', s=0.75)
            plt.show()

        plt.figure(6)
        plt.grid(True)
        plt.title("Sampling SVDD Performance")
        plt.xlabel("Sample Size")
        plt.ylabel("Time Taken (in seconds)")
        plt.plot(dsize_list,tsample)
        
        plt.figure(7)
        plt.grid(True)
        plt.title("Full SVDD Performance")
        plt.xlabel("Sample Size")
        plt.ylabel("Time Taken (in seconds)")
        plt.plot(dsize_list,tfull)
    
    run_main()    
    
    
        
        
        
        
    
    

