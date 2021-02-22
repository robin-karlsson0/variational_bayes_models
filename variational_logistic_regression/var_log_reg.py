    #!/usr/bin/env python3
import numpy as np
from scipy.special import expit
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

np.random.seed(seed=0)


class VarLogRegModel():
    '''Class for training and using the variational logistic regression model.

    How to use:
        1. Initialize the model
            model = VarLogRegModel()
        2. Train the model using a dataset
            dataset = ('data_matrix X', 'target labels Y')
            model.train(dataset)
        3. Do inference on new 'input feature vectors x'
            p(Y=True) = model.predictive_posterior_distr(x)
    '''
    def __init__(self, basis_func_deg, iter_max:int=600, conv_criteria:float=1e-6, log_iter_freq:int=1):
        '''
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

        # Basis feature generation object
        self.basis_func = PolynomialFeatures(degree=basis_func_deg)

        # Basis feature normalization object
        self.input_scaler = None
        self.basis_scaler = None

        # Feature dimensions
        self.input_dim = None
        self.basis_dim = None
            

    def train(self, dataset):
        '''Trains a VB logistic regression on the supplied dataset and returns model parameters.
        Args:
            dataset: Tuple (X, Y) consisting of
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
        a_0 = 1.0
        b_0 = 1.0

        a_N = a_0
        b_N = b_0

        mu_N = np.random.random((D,1))
        sigma_N = np.random.random((D,D))

        # Sample-specific latent variables 'xi'
        xis = np.ones((1, samples_N))

        lower_bound_prev = np.inf

        print("\nTraining model")
        for iter_idx in range(1, self.iter_max+1):

            # Expectation step
            expectation_alpha = self.gamma_expectation(a_N, b_N)                                      # scalar
            expectation_quadratic_w_wrt_w = self.gaussian_quadratic_expectation_wrt_w(mu_N, sigma_N)  # scalar
            
            # Maximization step
            a_N = self.maximization_an(a_0, D)                                       # scalar
            b_N = self.maximization_bn(b_0, expectation_quadratic_w_wrt_w)           # scalar
            sigma_N_inv = self.maximization_sigma_inv(expectation_alpha, xis, phis)  # [D,D]
            sigma_N = np.linalg.inv(sigma_N_inv)                                     # [D,D]
            mu_N = self.maximization_mu(phis, Y, sigma_N)                            # [D,1]
            xis = self.xis_update(phis, mu_N, sigma_N)                               # [1,N]
            
            # Lower bound
            sigma_0 = expectation_alpha * np.eye(D)
            lower_bound =-self.comp_lower_bound(sigma_0, sigma_N, sigma_N_inv, mu_N, xis, skip_determinants=False)
            self.lower_bound_list.append(lower_bound)
            
            if iter_idx % self.log_iter_freq == 0:
                print(f"{iter_idx}: {lower_bound}")
            
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

        return self.lower_bound_list


    def predictive_posterior_distr(self, x):
        '''Computes the probability of a new input vector 'x' belonging to class 1.
    
        Args:
            x: Input vector of shape (M,1)

        Returns:
            p: Probability of proposition (Y = 1 | phi)
        '''
        # Basis function require (1,M) input vector
        x = x.T
        # Normalize 'input matrix' X
        x_norm = self.input_scaler.transform(x)
        # Transform 'input vec' (1,M) --> 'feat vec' (1,D)
        phi = self.basis_func.transform(x_norm)
        # Normalize 'feat vec' similar to training
        phi = self.basis_scaler.transform(phi)
        # Recover bias term (as it gets normalized by the scaler)
        phi[:,0] = 1.
        # Need 'phi' as (D,1) vector
        phi = phi.T

        mu = np.matmul(self.mu.T, phi).item()
        sigma2 = np.matmul(np.matmul(phi.T, self.sigma), phi).item()

        kappa = np.power(1.0 + np.pi/8.0 * sigma2, -0.5)

        p = expit(kappa*mu)
        
        return p


    def comp_lower_bound(self, sigma_0, sigma_N, sigma_N_inv, mu_N, xis, skip_determinants=False):
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
            term_4 += np.log(expit(xi)) - 0.5*xi - self.lambda_func(xi) * xi**2 
        
        return term_1 - term_2 + term_3 + term_4


    def evaluate(self, dataset):
        '''Evaluates a trained model on a dataset according to F1 score.
        Args:
            dataset: Tuple (X, Y) consisting of
                  Data matrix X: (#samples N, #input features M)
                  Target vector Y: (#samples N, 1)
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Number of samples
        samples_N = X.shape[0]

        #####################
        #  Evaluation loop
        #####################
        y_true = []
        y_pred = []
        for sample_idx in range(samples_N):

            x = X[sample_idx:sample_idx+1, :].T
            y_out = self.predictive_posterior_distr(x)

            y_target = Y[sample_idx,0]

            y_true.append(y_target)
            if y_out >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        return f1_score(y_true, y_pred)


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
    def lambda_func(xi):
        return (expit(xi) - 0.5) / (2*xi + 1e-24)


    @staticmethod
    def gamma_expectation(a_N, b_N):
        '''Returns the expected value of a Gamma distribution defined by parameters 'a_N' and 'b_N'.
        
        Ref: Bishop Eq 10.182 (p.505)
        '''
        return a_N / b_N


    @staticmethod
    def gaussian_quadratic_expectation_wrt_w(mu_N, sigma_N):
        '''Returns the expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
        
        Because we are taking the expectations w.r.t. weights, other unrelated terms can be dropped from the general expectation.

        Args:
            mu_N: (D,1)
            sigma_N: (D,D)
        
        Ref: Bishop Eq 10.183 (p.505)
        '''
        expecation = np.matmul(mu_N.T, mu_N) + np.trace(sigma_N)
        return expecation.item()


    ############################################
    #  Maximization (re-estimation) equations
    ############################################

    @staticmethod
    def maximization_mu(phis, ts, sigma_N):
        '''Maximizes the weight vector distribution 'q(w)' w.r.t. mean vector.
        
        Args:
            phis:   : Feature matrix of shape (N,D)
            ts      : Label vector of shape (N,1)
            sigma_N : Covariance matrix for 'q(w)' of shape (D,D)
        
        Returns:
            mu_N : Vector of shape (D,1)
        
        Ref: Bishop Eq 10.175 (p.504)
        '''
        # Vector container for summing individual samples
        sample_vec_sum = np.zeros((phis.shape[1], 1))  # (D, 1)
        
        # Compute and sum each sample individually (TODO: vectorization)
        for i in range(phis.shape[0]):
            
            # Vectors for sample 'i'
            phi_i = phis[i:i+1,:].T  # [D,1]
            t_i = ts[i,0]            # scalar
            
            sample_vec_sum += (t_i - 0.5) * phi_i

        mu_N = np.matmul(sigma_N, sample_vec_sum)

        return mu_N


    def maximization_sigma_inv(self, expectation_alpha, xis, phis):
        '''Maximizes the weight vector distribution 'q(w)' w.r.t. variance of data.
        
        Args:
            expectation_alpha : Expected value of Gamma distribution modeling variance of 'q(w)'.
            xis               : Variational variable related to the class certainty of a sample (I think) (1,N).
            phis              : Feature matrix of shape (N,D).
            
        Returns:
            sigma_inv : Precision of q(w) which optimizes the approximation.
        
        Ref: Bishop Eq 10.176 (p.504)
        '''
        D = phis.shape[1]

        # Vector container for summing individual samples
        sample_vec_sum = np.zeros((D,D))
        
        # Compute and sum each sample individually (TODO: vectorization)
        for i in range(phis.shape[0]):
            
            # Vectors for sample 'i'
            phi_i = phis[i:i+1,:].T  # [D,1]
            xi_i = xis[0,i]          # scalar
            
            sample_vec_sum += self.lambda_func(xi_i) * np.matmul(phi_i, phi_i.T)
            
        sigma_inv = expectation_alpha * np.eye(D) + 2.0 * sample_vec_sum

        return sigma_inv

    @staticmethod
    def maximization_an(a_0, D):
        '''Maximizes the covariance matrix 'q(alpha)' w.r.t. the 'a_N' parameter.
        
        Args:
            a_0 : Value of initial variational hyperpriors.
            D   : Number of features.
        '''
        return a_0 + D


    @staticmethod
    def maximization_bn(b_0, expectation_quadratic_w):
        '''Maximizes the covariance matrix 'q(alpha)' w.r.t. the 'a_N' parameter.
        
        Args:
            b_0                     : Value of initial variational hyperpriors.
            expectation_quadratic_w : The expected value of the quadratic weight vectors for a Gaussian distribution w.r.t. weights.
        '''
        return b_0 + 0.5 * expectation_quadratic_w


    @staticmethod
    def xi_update(phi, mu_N, sigma_N):
        '''Computes the optimal variational variable related to class certainity (I think) of one sample, according to weight distribution 'q(w)'.
        
        Args:
            phi     : Sample feature vector of shape (D,1)
            mu_N    : Mean vector for weight distribution 'q(w)' of shape (D,1)
            sigma_N : Covariance matrix for weight distribution 'q(w)'  of shape (D,D)
        
        Returns : 'xi' value for one sample given 'q(w)'
        '''
        A = sigma_N + np.matmul(mu_N, mu_N.T)

        xi = np.matmul(np.matmul(phi.T, A), phi)
        xi = np.sqrt(xi)

        return xi


    def xis_update(self, phis, mu_N, sigma_N):
        '''Computes the optimal variational variable related to class certainity (I think) of all samples.
        
        Args:
            phis    : Sample feature matrix of shape (N,D)
            mu_N    : Mean vector for weight distribution 'q(w)' of shape (D,1)
            sigma_N : Covariance matrix for weight distribution 'q(w)' of shape (D,D)

        Returns:
            xis_new : (1,N)
        '''
        N = phis.shape[0]
        
        xis_new = np.zeros((1,N))
        
        # Compute each sample individually
        for i in range(N):
            
            # Vector for sample 'i'
            phi_i = phis[i:i+1,:].T   # [D,1]
            
            xis_new[0,i] = self.xi_update(phi_i, mu_N, sigma_N)
        
        return xis_new


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
    X = dataset[:,:2]
    Y = dataset[:,2:]
    dataset = (X, Y)

    #################
    #  TRAIN MODEL
    #################

    basis_func_deg = 3
    model = VarLogRegModel(basis_func_deg)

    model.train(dataset)

    score = model.evaluate(dataset)
    print(score)

    ####################
    #  VISUALIZE MODEL
    ####################
    
    # Compute posterior class probability over the whole input space.
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)

    xx, yy = np.meshgrid(x, y, sparse=False)

    p_array = np.zeros(xx.shape)

    for i in range(p_array.shape[0]):
        for j in range(p_array.shape[1]):

            x = np.empty((2,1))
            x[0] = xx[i,j]
            x[1] = yy[i,j]

            p_array[i,j] = model.predictive_posterior_distr(x)
            
    plt.figure(figsize=(10,10))
    plt.contourf(xx, yy, p_array, levels=20)
    plt.scatter(gaussian_1[:,0], gaussian_1[:,1], c='r')
    plt.scatter(gaussian_2[:,0], gaussian_2[:,1], c='g')

    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()