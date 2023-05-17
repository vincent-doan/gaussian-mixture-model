import numpy as np
import scipy.stats as sp

class GaussianMixtureModel:
  def __init__(self, X, k):
    # Random seed
    np.random.seed(2023)
    
    # data : 2D numpy array
    self.data = X.copy()
    
    # m : number of records in dataset
    # n : number of features in dataset
    self.m, self.n = X.shape
    
    # k : number of mixture components
    self.k = k    

    # Initialize model parameters
    # Row of metrix : mean of one mixture component
    self.means = np.asmatrix(np.random.random((self.k, self.n)) + np.mean(self.data))
    # Array element : covariance of one mixture component
    self.covariances = np.array([np.asmatrix(np.identity(self.n)) for _ in range(self.k)])
    # Array element : weight (scalar) of one mixture component
    self.weights = np.ones(self.k)/self.k
    # Row of matrix : responsibility of each mixture component for one data point
    self.R = np.asmatrix(np.empty((self.m, self.k), dtype=float))

  def log_likelihood(self):
    """
    Calculates the log-likelihood of dataset 
    For each data point, evaluate the PDF of each mixture component,
    and sum up the logarithms of those probabilities.
    Iterate over all data points.
    """
    logl = 0
    for i in range(self.m):
      tmp = 0
      for j in range(self.k):
        tmp += sp.multivariate_normal.pdf(self.data[i, :], 
                                          self.means[j, :].A1, # .A1 converts into flattened 1D array
                                          self.covariances[j]) 
      logl += np.log(tmp)
    return logl
  
  def e_step(self):
    """
    --CALCULATE self.R--
    Calculates the responsibility of each mixture component for each data point,
    using the current parameter estimates,
    stored in self.R (a [m x k] matrix)
    """
    for i in range(self.m):
      evd = 0
      for j in range(self.k):
        llh = sp.multivariate_normal.pdf(self.data[i, :],
                                         self.means[j, :].A1,
                                         self.covariances[j])
        pr = self.weights[j]
        evd += llh * pr
        self.R[i, j] = llh * pr
      self.R[i, :] /= evd
      # Assert that for each row of matrix self.R, elements sum to 1
      assert self.R[i, :].sum() - 1 < 1e-4
  
  def m_step(self):
    """
    --UPDATE self.means, self.covariances, self.weights--
    Maximizes the expected complete data log likelihood,
    by updating the means, covariances, and mixture component weights.
    """
    for j in range(self.k):
      # Sums over all responsibilities of jth mixture component for each data point
      mixture_responsibilities = self.R[:, j].sum()
      
      # Calculate new mixture component weight for jth mixture component
      self.weights[j] = mixture_responsibilities/self.m
      
      """
      # Initialize new mean for jth mixture component
      updated_mean = np.zeros(self.n)
      for i in range(self.m):
        updated_mean += self.R[i, j] * self.data[i, :]
      self.means[j, :] = updated_mean / mixture_responsibilities
      
      # Initialize new covariance for jth mixture component
      updated_covariance = np.zeros((self.n, self.n))
      for i in range(self.m):
        updated_covariance += self.R[i, j] * self.data[i, :].T.dot(self.data[i, :])
      self.covariances[j] = updated_covariance / mixture_responsibilities - self.means[j, :].T.dot(self.means[j, :])
      """
      updated_mean = np.zeros(self.n)
      updated_covariance = np.zeros((self.n, self.n))
      for i in range(self.m):
        updated_mean += (self.data[i, :] * self.R[i, j])
        updated_covariance += self.R[i, j] * ((self.data[i, :] - self.means[j, :]).T * (self.data[i, :] - self.means[j, :]))

      self.means[j] = updated_mean / mixture_responsibilities
      self.covariances[j] = updated_covariance / mixture_responsibilities

  def fit(self, tol=1e-4):
    """
    --PERFORM EM algorithm--
    EM algorithm on loop until change in log likelihood falls below threshold
    """
    num_iters = 0
    logl = 1
    previous_logl = 0
    while (logl - previous_logl > tol):
      previous_logl = self.log_likelihood()
      self.e_step()
      self.m_step()
      num_iters += 1
      logl = self.log_likelihood()
      print('Iteration %d: log-likelihood is %.6f'%(num_iters, logl))
    print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, logl))