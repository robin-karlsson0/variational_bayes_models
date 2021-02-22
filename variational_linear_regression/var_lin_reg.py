#!/usr/bin/env python3
import numpy as np
from scipy import special
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

np.random.seed(seed=0)


class VarLinRegModel():
    '''
    '''
    def __init__(self, basis_func_deg, iter_max:int=200, conv_criteria:float=1e-6, log_iter_freq:int=1):
        '''

        mu: (D,1) vector
        sigma: (D,D) matrix

        Args:
            basis_func: 
            iter_max:
            conv_criteria:
            log_iter_freq:
        '''
        self.iter_max = iter_max
        self.conv_criteria = conv_criteria
        self.log_iter_freq = log_iter_freq
    
        # Stores the lower bound value of each iteration
        self.lower_bound_list = []

        # Model parameters
        self.mu = None
        self.sigma = None
        self.beta = None

        # Basis feature generation object
        self.basis_func_deg = basis_func_deg
        self.basis_func = PolynomialFeatures(degree=basis_func_deg)

        # Input and basis feature normalization object
        self.input_scaler = None
        self.basis_scaler = None


    def train(self, dataset):
        '''
        Args:
            dataset: Dataset tuple (X, Y) consisting of
                     Data matrix X: (#samples N, #input features M)
                     Target vector Y: (#samples N, 1)
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Number of samples
        samples_N = X.shape[0]

        # Basis feature matrix (#samples N, #basis features D)
        phis = self.convert_input_to_basis_matrix(X)

        D = phis.shape[1]

        print("\nSummary of training data")
        print(f"  Basis matrix 'phis' : {phis.shape}")
        print(f"  Label vector 'Y'    : {Y.shape}")
        print(f"    #Samples        : {samples_N}")
        print(f"    #Input features : {X.shape[1]}")
        print(f"    #Basis feat.    : {D}")

        ###################
        #  Training loop
        ###################

        # Initial variables - hyperpriors for Gamma distributions
        a_0 = 1.
        b_0 = 1.
        c_0 = 1.
        d_0 = 1.

        a_N = a_0
        b_N = b_0
        c_N = c_0
        d_N = d_0

        mu_N = np.random.random((D,1))
        sigma_N = np.random.random((D,D))

        lower_bound_prev = np.inf
    
        print("\nTraining model")
        for iter_idx in range(1, self.iter_max+1):

            # Expectation step
            exp_alpha = self.alpha_expectation(a_N, b_N)
            exp_beta = self.beta_expectation(c_N, d_N)
            exp_quad_w = self.gaussian_quadratic_expectation_wrt_w(mu_N, sigma_N)
            
            # Maximization step
            mu_N, sigma_N = self.estim_qw_distr(phis, Y, exp_alpha, exp_beta)
            a_N, b_N = self.estim_qalpha_distr(a_0, b_0, exp_quad_w, D)
            c_N, d_N = self.estim_qbeta_distr(c_0, d_0, phis, Y, mu_N, sigma_N, samples_N)
            
            # Lower bound
            lower_bound = self.comp_lower_bound(phis, Y, mu_N, sigma_N, exp_beta, a_0, b_0, c_0, d_0, a_N, b_N, c_N, d_N, D, samples_N)
            self.lower_bound_list.append(lower_bound)

            if iter_idx % self.log_iter_freq == 0:
                print(f"  iter {iter_idx} | {lower_bound} | beta: {self.beta_expectation(c_N, d_N):.4f}")

            # Check convergence
            if (np.abs(lower_bound - lower_bound_prev) < self.conv_criteria):
                print("\n  Model converged:")
                print(f"     iteration: {iter_idx}")
                print(f"     convergence criteria: {self.conv_criteria}")
                break
            else:
                lower_bound_prev = lower_bound

        # Store optmized model parameters
        self.mu = mu_N
        self.sigma = sigma_N
        self.beta = self.beta_expectation(c_N, d_N)

        return self.lower_bound_list

    def predictive_posterior_distr(self, x):
        '''Computes the probability distribution representing the output value of a feature vector.

        mu = mu_N^T * phi
        sigma^2 = 1/beta + phi^T * Sigma_N * phi
        
        Args:
            x: Input vector of shape (m,1).
        
        Returns:
            mu, sigma^2: Gaussian distribution parameters representing the predicted data generating process (output value 'y' given input 'x').
        '''
        # Basis function require (1,m) input vector
        x = x.T
        # Transform 'input vec' (1,m) --> 'feat vec' (1,d)
        phi = self.convert_input_to_basis_matrix(x)
        # Need 'phi' as (D,1) vector
        phi = phi.T
        
        mu = self.mu.T.dot(phi)
        sigma2 = 1./self.beta + phi.T.dot(self.sigma).dot(phi)
        return mu[0,0], sigma2[0,0]

    @staticmethod
    def comp_lower_bound(phis, ts, mu_N, sigma_N, beta, a_0, b_0, c_0, d_0, a_N, b_N, c_N, d_N, D, N, eps=1e-14):
        # Exp[ log p(t|w) ]_{w}
        term_1 = 0.5 * N * np.log(beta / (2.*np.pi) + eps) - 0.5 * beta * ts.T.dot(ts) + beta * mu_N.T.dot(phis.T).dot(ts) - 0.5 * beta * np.trace(phis.T.dot(phis).dot( mu_N.dot(mu_N.T) + sigma_N ))
        # Exp[ log p(w|alpha) ]_{w, alpha}
        term_2 = -0.5 * D * np.log(2.*np.pi) + 0.5*D*(special.digamma(a_N) - np.log(b_N + eps)) - (a_N / (2.*b_N)) * (mu_N.T.dot(mu_N) + np.trace(sigma_N))
        # Exp[ log p(alpha) ]_{alpha}
        term_3 = a_0 * np.log(b_0 + eps) + (a_0 - 1.)*(special.digamma(a_N) - np.log(b_N + eps)) - b_0*(a_N/b_N) - np.log(special.gamma(a_N) + eps)
        # Exp[ log p(beta) ]_{beta}
        term_4 = c_0 * np.log(d_0 + eps) + (c_0 - 1.)*(special.digamma(c_N) - np.log(d_N + eps)) - d_0*(c_N/d_N) - np.log(special.gamma(c_N) + eps)
        # Exp[ log q(w) ]_{w}
        term_5 = -0.5*np.log(np.linalg.det(sigma_N) + eps) - 0.5*D*(1. + np.log(2.*np.pi))
        # Exp[ log q(alpha) ]_{alpha}
        term_6 = -np.log(special.gamma(a_N) + eps) + (a_N - 1.)*special.digamma(a_N) + np.log(b_N + eps) - a_N
        # Exp[ log q(beta) ]_{beta}
        term_7 = -np.log(special.gamma(c_N) + eps) + (c_N - 1.)*special.digamma(c_N) + np.log(d_N + eps) - c_N
        lower_bound = term_1 + term_2 + term_3 + term_4 - term_5 - term_6 - term_7
        # Return scalar
        return lower_bound.item()

    def evaluate(self, dataset):
        '''Evaluates a trained model on a dataset according to errors and MSE.
        Args:
            dataset: Tuple (X, Y) consisting of
                Data matrix X:   (#samples N, #input features M)
                Target vector Y: (#samples N, 1)
        Returns:
            Tuple of errors and MSE
                errors: Array of signed errors (prediction - target)
                MSE:    Scalar
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Number of samples
        samples_N = X.shape[0]

        #####################
        #  Evaluation loop
        #####################
        errors = []
        MSE = []
        for sample_idx in range(samples_N):

            x = X[sample_idx:sample_idx+1, :].T
            mu_out, _ = self.predictive_posterior_distr(x)

            y_target = Y[sample_idx,0]

            diff = mu_out - y_target
            errors.append(diff)
            MSE.append(diff**2)
        
        errors = np.array(errors)
        MSE = np.sum(np.array(MSE)) / len(MSE)

        return errors, MSE

    def convert_input_to_basis_matrix(self, X):
        '''Returns the basis feature matrix 'phis' and fits basis cunc
        Args:
            X: Input matrix of shape (#samples N, #input features M)
        '''
        # Normalize 'input matrix' X
        if self.input_scaler == None:
            self.input_scaler = preprocessing.StandardScaler().fit(X)

        X_norm = self.input_scaler.transform(X)

        # Transform 'input vectors' x --> 'feature vectors' phi 
        phis = self.basis_func.fit_transform(X_norm)

        # Normalize 'feature matrix' phis
        if self.basis_scaler == None:
            self.basis_scaler = preprocessing.StandardScaler().fit(phis)

        phis = self.basis_scaler.transform(phis)
          
        # Recover bias term (as it gets normalized by the scaler)
        phis[:,0] = 1.

        return phis

    ##################
    #  Expectations
    ##################
    
    @staticmethod
    def alpha_expectation(a_N, b_N):
        '''Returns the expected value of a Gamma distribution defined by parameters 'a_N' and 'b_N'.
        
        Ref: Bishop Eq 10.102 (p.488)
        '''
        return a_N / b_N


    @staticmethod
    def beta_expectation(c_N, d_N):
        '''Returns the expected value of a Gamma distribution defined by parameters 'c_N' and 'd_N'.
        
        Ref: Equiv. to Bishop Eq 10.102 (p.488)
        '''
        return c_N / d_N


    @staticmethod
    def gaussian_quadratic_expectation_wrt_w(mu_N, sigma_N):
        '''Returns the expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
        
        Because we are taking the expectations w.r.t. weights, other unrelated terms can be dropped from the general expectation.
        
        Ref: Bishop Eq 10.183 (p.505)
        '''
        val = np.dot(mu_N.T, mu_N) + np.trace(sigma_N)
        return val.item() 

    ############################################
    #  Maximization (re-estimation) equations
    ############################################

    @staticmethod
    def estim_qw_distr(phis, ts, exp_alpha, exp_beta):
        '''Returns parameters for the re-estimated distribution which maximizes the lower bound.
        
        Args:
            phis:   : Feature matrix of shape (N, D).
            ts      : Label vector of shape (N, 1).
            exp_alpha : Expected value of Gamma distribution modeling variance of 'p(w)'.
            exp_beta  : Expected value of Gamma distribution modeling variance of 'p(t)'.
            
        Returns:
            mu_N, sigma_N : Mean vector and covariance matrix of shapes (D,1) and (D,D).
        
        Ref: Bishop Eq. 10.96-10.98 (p.487)
        '''
        feature_dim = phis.shape[1]
        sigma_N = exp_beta * phis.T.dot(phis) + exp_alpha * np.eye(feature_dim)
        sigma_N = np.linalg.inv(sigma_N)

        mu_N = exp_beta * sigma_N.dot(phis.T).dot(ts)
        
        return mu_N, sigma_N

    @staticmethod
    def estim_qalpha_distr(a_0, b_0, exp_quad_w, D):
        '''Returns parameters for the re-estimated distribution which maximizes the lower bound.
        
        Args:
            a_0 : Value of initial variational hyperpriors for alpha distr.
            b_0 : 
            exp_quad_w : The expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
            D : Number of model parameters (i.e. length of mu_N).
            
        Returns:
            a_N, b_N : Scalar parameters.
            
        Ref: Bishop Eq. 10.93-10.95 (p.487)
        '''
        a_N = a_0 + 0.5*D
        b_N = b_0 + 0.5*exp_quad_w
        
        return a_N, b_N

    @staticmethod
    def estim_qbeta_distr(c_0, d_0, phis, ts, mu_N, sigma_N, N):
        '''Returns parameters for the re-estimated distribution which maximizes the lower bound.
        
        Args:
            c_0     : Value of initial variational hyperpriors for beta distr.
            d_0     : 
            phis    : Feature matrix of shape (D,N).
            ts      : Label vector of shape (1,N).
            mu_N    : Mean vector for weight distribution 'q(w)' of shape (D,1).
            sigma_N : Covariance matrix for weight distribution 'q(w)' of shape (D,D).
            N       : Number of samples (i.e. observations).
        
        Returns:
            c_N, d_N : Scalar parameters.
        '''
        c_N = c_0 + 0.5*N
        d_N = d_0 + 0.5*(np.linalg.norm(phis.dot(mu_N) - ts)**2 + np.trace(phis.T.dot(phis).dot(sigma_N)))
        
        return c_N, d_N

    def get_basis_func_deg(self):
        return self.basis_func_deg


if __name__ == "__main__":

    basis_func_deg = 3
    model = VarLinRegModel(basis_func_deg, iter_max=200)

    #################################
    #  Generate dataset D = (X, Y)
    #################################

    obs_N = 40
    x_range_min = 0.
    x_range_max = 2.
    x = np.random.random(obs_N) * (x_range_max - x_range_min)

    y_var = 0.4
    y_noise = np.random.normal(scale=y_var, size=obs_N)

    y = np.zeros(obs_N)

    for i in range(obs_N):
        y[i] = np.sin(x[i] * np.pi) + y_noise[i]

    x_dense = np.linspace(x_range_min, x_range_max, 200)
    y_true = np.sin(np.pi*x_dense)

    X = np.expand_dims(x, 0).T
    Y = np.expand_dims(y, 1)

    print("\nSummary of dataset")
    print(f"  Data matrix X : {X.shape}")
    print(f"  Label vector Y: {Y.shape}")

    dataset = (X, Y)

    #################
    #  Train model
    #################
    
    model.train(dataset)

    errors, MSE = model.evaluate(dataset)
    print(f"errors: {errors}")
    print(f"RMSE: {np.sqrt(MSE)}")

    #####################
    #  Model inference
    #####################

    N = 200
    x_array = np.linspace(x_range_min, x_range_max, N)
    mu_array = np.zeros(N)
    sigma2_array = np.zeros(N)

    for i in range(x_array.shape[0]):

        x_norm = np.array([[x_array[i]]])
        mu_i, sigma2_i = model.predictive_posterior_distr(x_norm)
        
        mu_array[i] = mu_i
        sigma2_array[i] = sigma2_i

    sigma_array = np.sqrt(sigma2_array)

    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.plot(x_dense, y_true)
    plt.fill_between(x_dense, y_true - y_var, y_true + y_var, alpha=.1, color="blue")
    plt.plot(x, y, 'o')
    plt.plot(x_array, mu_array)
    plt.fill_between(x_array, mu_array - sigma_array, mu_array + sigma_array, alpha=0.2, color="green")
    plt.ylim(-2,2)
    plt.title("'Predicted distribution' and 'true distribution' overlayed by observations")

    plt.subplot(1,2,2)
    plt.plot(x_array, sigma2_array)
    plt.title("Combined 'parameter' and 'model output' variance")
    plt.show()

    print(f"\nModel mean vector (shape: {model.mu.shape}")
    print(model.mu)

    print(f"\nWeight distribution covariance matrix (shape: {model.sigma.shape})")
    print(model.sigma)  
