#!/usr/bin/env python3
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

np.random.seed(seed=0)


def input2feature(input_vec, normalization_const=5.0):
    '''
    Args:
        input_vec: Input vector of shape (1,2)
        
    Returns:
        feat_vec: Feature vector of shape(6,1)
    '''
    
    input_vec = np.array(input_vec) / normalization_const
    
    feat_vec = np.zeros((6,1))
    
    feat_vec[0,0] = input_vec[0]
    feat_vec[1,0] = input_vec[0]**2
    feat_vec[2,0] = input_vec[1]
    feat_vec[3,0] = input_vec[1]**2
    feat_vec[4,0] = input_vec[0] * input_vec[1]
    feat_vec[5,0] = 1.0
    
    return feat_vec


def lambda_func(xi):
    return (expit(xi) - 0.5) / (2*xi + 1e-24)


def gamma_expectation(a_N, b_N):
    '''Returns the expected value of a Gamma distribution defined by parameters 'a_N' and 'b_N'.
    
    Ref: Bishop Eq 10.182 (p.505)
    '''
    return a_N / b_N


def gaussian_quadratic_expectation_wrt_w(mu_N):
    '''Returns the expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
    
    Because we are taking the expectations w.r.t. weights, other unrelated terms can be dropped from the general expectation.
    
    Ref: Bishop Eq 10.183 (p.505)
    '''
    return np.matmul(mu_N.T, mu_N)


def maximization_mu(phis, ts, sigma_N):
    '''Maximizes the weight vector distribution 'q(w)' w.r.t. mean vector.
    
    Args:
        phis:   : Feature matrix of shape (D,N)
        ts      : Label vector of shape (1,N)
        sigma_N : Covariance matrix for 'q(w)' of shape (D,D)
    
    Returns:
        mu_N : Vector of shape (D,1)
    
    Ref: Bishop Eq 10.175 (p.504)
    '''
    # Vector container for summing individual samples
    sample_vec_sum = np.zeros((phis.shape[0], 1))
    
    # Compute and sum each sample individually (TODO: vectorization)
    for i in range(phis.shape[1]):
        
        # Vectors for sample 'i'
        phi_i = phis[:,i:i+1]  # [D,1]
        t_i = ts[:,i].item()   # scalar
        
        sample_vec_sum += (t_i - 0.5) * phi_i
    
    mu_N = np.matmul(sigma_N, sample_vec_sum)
    
    return mu_N


def maximization_sigma_inv(expectation_alpha, xis, phis):
    '''Maximizes the weight vector distribution 'q(w)' w.r.t. variance of data.
    
    Args:
        expectation_alpha : Expected value of Gamma distribution modeling variance of 'q(w)'.
        xis               : Variational variable related to the class certainty of a sample (I think).
        phis              : Feature matrix of shape (D,N).
        
    Returns:
        sigma_inv : Precision of q(w) which optimizes the approximation.
    
    Ref: Bishop Eq 10.176 (p.504)
    '''
    D = phis.shape[0]
    
    # Vector container for summing individual samples
    sample_vec_sum = np.zeros((D,D))
    
    # Compute and sum each sample individually (TODO: vectorization)
    for i in range(phis.shape[1]):
        
        # Vectors for sample 'i'
        phi_i = phis[:,i:i+1]   # [D,1]
        xi_i = xis[:,i].item()  # scalar
        
        sample_vec_sum += lambda_func(xi_i) * np.matmul(phi_i, phi_i.T)
        
    sigma_inv = expectation_alpha * np.eye(D) + 2.0 * sample_vec_sum

    return sigma_inv


def maximization_an(a_0, D):
    '''Maximizes the covariance matrix 'q(alpha)' w.r.t. the 'a_N' parameter.
    
    Args:
        a_0 : Value of initial variational hyperpriors.
        D   : Number of features.
    '''
    return a_0 + D


def maximization_bn(b_0, expectation_quadratic_w):
    '''Maximizes the covariance matrix 'q(alpha)' w.r.t. the 'a_N' parameter.
    
    Args:
        b_0                     : Value of initial variational hyperpriors.
        expectation_quadratic_w : The expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
    '''
    return b_0 + 0.5 * expectation_quadratic_w


def xi_update(phi, mu_N, sigma_N):
    '''Computes the optimal variational variable related to class certainity (I think) of one sample, according to weight distribution 'q(w)'.
    
    Args:
        phi     : Sample feature vector of shape (D,1)
        mu_N    : Mean vector for weight distribution 'q(w)' of shape (D,1)
        sigma_N : Covariance matrix for weight distribution 'q(w)'  of shape (D,D)
    
    Returns : 'xi' value for one sample given 'q(w)'
    '''
    # xi = phi.T * (sigma_N + mu_N * mu_N.T) * phi
    A = sigma_N + np.matmul(mu_N, mu_N.T)
    xi = np.matmul(np.matmul(phi.T, A), phi)
    xi = np.sqrt(xi)
    return xi


def xis_update(phis, mu_N, sigma_N):
    '''Computes the optimal variational variable related to class certainity (I think) of all samples.
    
    Args:
        phi     : Sample feature matrix of shape (D,N)
        mu_N    : Mean vector for weight distribution 'q(w)' of shape (D,1)
        sigma_N : Covariance matrix for weight distribution 'q(w)' of shape (D,D)
    '''
    N = phis.shape[1]
    
    xis_new = np.zeros((1,N))
    
    # Compute each sample individually
    for i in range(N):
        
        # Vector for sample 'i'
        phi_i = phis[:,i:i+1]   # [D,1]
        
        xis_new[0,i] = xi_update(phi_i, mu_N, sigma_N)
    
    return xis_new


def comp_lower_bound(sigma_0, sigma_N, sigma_N_inv, mu_N, xis, skip_determinants=False):
    sigma_N_det = np.linalg.det(sigma_N)
    sigma_0_det = np.linalg.det(sigma_0)
    
    if skip_determinants:
        term_1 = 0.
    else:
        term_1 = 0.5 * np.log(sigma_N_det / sigma_0_det)
    
    term_2 = 0.5 * np.matmul(mu_N.T, np.matmul(sigma_N_inv, mu_N)).item()
    
    term_3 = 0.0  # As mu_0 = zero vector
    
    term_4 = 0.

    for idx in range(xis.shape[1]):
        
        xi = xis[0,idx]
        term_4 += np.log(expit(xi)) - 0.5*xi - lambda_func(xi) * xi**2 
    
    return term_1 - term_2 + term_3 + term_4


def EM_iteration_func(phi_train, y_train, basis_N, samples_train_N, max_iter, conv_criteria=1e-4, print_interval=1):
    # Hyperpriors for Gamma distribution
    a_0 = 1.0
    b_0 = 1.0

    a_N = a_0
    b_N = b_0

    mu_N = np.random.random((basis_N,1))
    sigma_N = np.random.random((basis_N,basis_N))

    # Sample-specific latent variables 'xi'
    xis = np.ones((1, samples_train_N))

    prev_lower_bound = -np.inf

    for iter_idx in range(max_iter):
        
        # Expectation step
        expectation_alpha = gamma_expectation(a_N, b_N)                             # scalar
        expectation_quadratic_w_wrt_w = gaussian_quadratic_expectation_wrt_w(mu_N)  # scalar
        
        # Maximization step
        a_N = maximization_an(a_0, basis_N)                                         # scalar
        b_N = maximization_bn(b_0, expectation_quadratic_w_wrt_w)                   # scalar
        sigma_N_inv = maximization_sigma_inv(expectation_alpha, xis, phi_train)     # [D,D]
        sigma_N = np.linalg.inv(sigma_N_inv)                                        # [D,D]
        mu_N = maximization_mu(phi_train, y_train, sigma_N)                         # [D,1]
        xis = xis_update(phi_train, mu_N, sigma_N)                                  # [1,N]
        
        # Compute lower bound
        sigma_0 = expectation_alpha * np.eye(basis_N)
        
        lower_bound =-comp_lower_bound(sigma_0, sigma_N, sigma_N_inv, mu_N, xis, skip_determinants=False)
        
        if iter_idx % print_interval == 0:
            print(f"{iter_idx}: {lower_bound}")

        lower_bound_diff = lower_bound - prev_lower_bound
        if lower_bound_diff < conv_criteria:
            break

        prev_lower_bound = lower_bound
        
    return mu_N, sigma_N, iter_idx


def train_var_log_reg_model(dataset, max_iter, conv_criteria=1e-4):
    '''Trains a VB logistic regression on the supplied dataset and returns model parameters.

    Args:
        dataset:
        max_iter:
        conv_criteria:

    Returns:
        model: Tuple consisting of a (mean vectors, covariance matrix, basis_scaler).
            basis_scaler: sklearn preprocessing object used to normalize basis vectors.
    '''
    # Number of samples
    N = dataset.shape[0]
    
    ############################################
    #  TRANSFORM 'FEATURE VEC' -> 'BASIS VEC'
    ############################################
    phis = np.zeros((6, N))

    for n in range(N):
        phis[:,n:n+1] =input2feature(dataset[n,:])

    # Number of features
    D = phis.shape[0]

    ts = np.zeros((1,N))
    ts[0,:] = dataset[:,2]
    
    print(f"Feature array 'phis' : {phis.shape}")
    print(f"Label array 'ts'     : {ts.shape}")
    print(f"    #Samples  : {D}")
    print(f"    #Features : {D}")

    #################
    #  TRAIN MODEL
    #################
    model_mu, model_sigma, iterations = EM_iteration_func(phis, ts, D, N, max_iter, conv_criteria)

    print("Model converged:")
    print(f"  iteration: {iterations}")
    print(f"  convergence criteria: {conv_criteria}")

    return (model_mu, model_sigma)


def predictive_posterior_prob(x, model_mu, model_sigma):
    '''Computes the probability of a new input vector 'x' belonging to class 1 using the learned variational parameters.
    
    Args:
        phi         : Input normalized basis vector (or list) of shape (1,D) TODO: check this
        model_mu    : Learned mean vector for weight distribution 'q(w)'
        model_sigma : Learned covariance matrix for weight distribution 'q(w)'

    Returns:
        p: Probability of proposition (Y = 1 | phi)
    '''
    phi =input2feature(x)
    
    mu_a = np.matmul(model_mu.T, phi).item()
    sigma_a_2 = np.matmul(np.matmul(phi.T, model_sigma), phi).item()

    kappa = np.power(1.0 + np.pi/8.0 * sigma_a_2,-0.5)

    p = expit(kappa*mu_a)
    
    return p


def visualize_model(gaussian_1, gaussian_2, model):
    '''
    Args:
        dataset: Tuple with (X, y) representing the feature matrix and label vector of a dataset.
        model:   Tuple consisting of a (mean vectors, covariance matrix, basis_scaler).
    '''
    mu_N = model[0]
    sigma_N = model[1]
    
    # Compute posterior class probability over the whole input space.
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)

    xx, yy = np.meshgrid(x, y, sparse=False)

    p_array = np.zeros(xx.shape)

    for i in range(p_array.shape[0]):
        for j in range(p_array.shape[1]):

            x = (xx[i,j], yy[i,j])

            p_array[i,j] = predictive_posterior_prob(x, mu_N, sigma_N)
            
    plt.figure(figsize=(10,10))
    plt.imshow(p_array, origin='lower', extent=[-10,10,-10,10])
    plt.scatter(gaussian_1[:,0], gaussian_1[:,1], c='r')
    plt.scatter(gaussian_2[:,0], gaussian_2[:,1], c='g')

    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    
if __name__ == "__main__":

    ###################
    #  TRAINING DATA
    ###################
    mean_1 = [-6, 3]
    cov_1 = [[1,0.5],[0.5,3]]
    gaussian_1 = np.random.multivariate_normal(mean_1, cov_1, 50)

    mean_2 = [6, -3]
    cov_2 = [[1,0.5],[0.5,3]]
    gaussian_2 = np.random.multivariate_normal(mean_2, cov_2, 50)
    
    data_1 = np.zeros((gaussian_1.shape[0], 3))
    data_1[:,0:2] = gaussian_1

    data_2 = np.ones((gaussian_2.shape[0], 3))
    data_2[:,0:2] = gaussian_2
    
    # Dataset is stored as a (N,2) array
    dataset = np.vstack((data_1, data_2))
    dataset = np.random.permutation(dataset)

    #################
    #  TRAIN MODEL
    #################
    model = train_var_log_reg_model(dataset, 500, 1e-6)

    ####################
    #  VISUALIZE MODEL
    ####################
    visualize_model(gaussian_1, gaussian_2, model)
    