"""
Restricted Boltzmann Machine (RBM)
"""
import numpy as np

class rbm:
    """
    This class implements a Bernoulli Restricted Boltzmann Machine with binary visible and
    hidden units.

    Parameters:
    -----------
    n_hidden : int
        Number of hidden units.

    learning_rate : float
        The learning rate for stochastic gradient decent algorithm.

    n_gibbs_iter : float
        Number of Gibbs sampling iterations used in Contrastive Divergence (CD) approximation.

    n_epoch : int
        Number of iterations over a minipatch in during the training.

    batch_size : int
        Number of data samples in a minibatch.

    rng: numpy.RandomState
        The random number generator used to generate all random samples
        needed by the algorithm.
        
    h_phv_generator: function
        A function of form gen_h_phv(v, W, b, rng) which generates samples of visible variables
        given values of hidden variables, where v is a matrix of visible units, W and b are RBM weight
        and hidden bias variables and rng is the random generator state.
        This function can be used to generate arbitrary distributions for p(h|v), e.g., exponential
        family distribution. If no generator is provided, the default gen_h_phv_bernoulli(v, W, b, rng)
        is used. 

    v_pvh_generator: function
        A function of form gen_v_pvh(h, W, a, rng) which generates samples of visible variables
        given values of hidden variables, where v is a matrix of visible units, W and a are RBM weight
        and hidden bias variables and rng is the random generator state.
        This function can be used to generate arbitrary distributions for p(v|h), e.g., exponential
        family distribution. If no generator is provided, the default gen_v_pvh_bernoulli(h, W, a, rng)
        is used. 
    """
    def __init__(self, n_hidden, learning_rate, n_gibbs_iter=2, n_epoch=100, 
                 batch_size=10, rng=None, h_phv_generator=None, v_pvh_generator=None, verbose=False):
        self.n_h = n_hidden
        self.learning_rate = learning_rate
        self.n_gibbs_iter = n_gibbs_iter
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.rng = rng
        self.verbose = verbose
        
        if (h_phv_generator is None):
            self.gen_h_phv_ = rbm.gen_h_phv_Bernoulli
        else:
            self.gen_h_phv_ = h_phv_generator

        if (v_pvh_generator is None):
            self.gen_v_pvh_ = rbm.gen_v_pvh_Bernoulli
        else:
            self.gen_v_pvh_ = v_pvh_generator

    
    @staticmethod
    def logistic(x):
        return (1 + np.exp(-x)) ** -1
    
    @staticmethod
    def gen_h_phv_Bernoulli(v, W, b, rng):
        """ Default gen_h_phv function using Bernoulli distribution for p(h|v)
        """ 
        x = W.T.dot(v) + b.dot(np.ones((1, v.shape[1])))
        # Generate Bernoulli distributed RVs according to Bernoulli(logistic(W'*v+b)).
        return rng.random_sample((x.shape[0], x.shape[1])) < rbm.logistic(x)
    
    def gen_h_phv(self, v, W, b):
        """Generate samples for hidden variables from distribution p(h|v).

        Parameters:
        ----------
        v : array of shape (n_v, L)
            A batch of size L of visible variables.

        W : array of shape (n_v, n_h)
            Weight matrix.
        
        b : array of shape (n_h, 1)
            Bias variables for hidden components.
        
        Return value:
        --------------
            array of shape (n_h, L)
            A batch of size L of hidden variables sampled from p(h|v).
        """
        return self.gen_h_phv_(v, W, b, self.rng)

    @staticmethod
    def gen_v_pvh_Bernoulli(h, W, a, rng):
        """ Default gen_v_pvh function using Bernoulli distribution for p(v|h)
        """ 
        x = W.dot(h) + a.dot(np.ones((1, h.shape[1])))
        # Generate Bernoulli distributed RVs according to Bernoulli(logistic(W*h+a)).
        return rng.random_sample((x.shape[0], x.shape[1])) < rbm.logistic(x)
    
    def gen_v_pvh(self, h, W, a):
        """Generate samples for visible variables from distribution p(v|h).

        Parameters:
        ----------
        h : array of shape (n_h, L)
            A batch of size L of hidden variables.

        W : array of shape (n_v, n_h)
            Weight matrix.
        
        a : array of shape (n_v, 1)
            Bias variables for visible components.
        
        Return value:
        --------------
            array of shape (n_v, L)
            A batch of size L of visible variables sampled from p(v|h).
        """
        return self.gen_v_pvh_(h, W, a, self.rng)

    def set_v_pvh_generator(self, v_pvh_generator):
        """ Setter function for v_pvh_generator.     
        """
        self.gen_v_pvh_ = v_pvh_generator

    def set_h_phv_generator(self, h_phv_generator):
        """ Setter function for h_phv_generator.     
        """
        self.gen_h_phv_ = h_phv_generator

    def gradient(self, v, h):
        """ Calculate the gradient for a mini-batch needed for one step of the
            stochastic gradient decent algorithm.
            
        Parameters:
        ----------
        v : array of shape (n_v, batch_size)
            A mini-batch of visible variables.
            
        h : array of shape (n_h, batch_size)
            A mini-batch of hidden variables.
        
        Return values:
            array of shape (n_v * n_h + n_v + n_h, 1)
            Gradient for the input mini-batch.
        """
        # Gradients for W, a, and b
        W_grad = np.reshape(-v.dot(h.T), (self.n_v * self.n_h, 1), 'F')
        a_grad = -v.dot(np.ones((self.batch_size, 1)))
        b_grad = -h.dot(np.ones((self.batch_size, 1)))
        
        return 1.0 / self.batch_size * np.vstack((W_grad, a_grad, b_grad)) 

    def pretrain_(self, v):
        """ Pre-train the model for one mini-batch.

        Parameters:
        ----------
        v : array of shape (n_v, batch_size)
            A mini-batch of training samples.
        """
        n_v = self.n_v
        n_h = self.n_h
        
        # Obtain W, a, b from theta
        W = np.reshape(self.theta[0:n_v * n_h], (n_v, n_h), 'F')
        a = self.theta[n_v * n_h: n_v * n_h + n_v]
        b = self.theta[n_v * n_h + n_v: n_v * n_h + n_v + n_h]
        
        # Perform Contrastive Divergence approximation to obtain approximate samples for
        # p(h|v, W, a, b) and p(v|h, W, a, b).
        h_data = self.gen_h_phv(v, W, b)
        h_free = h_data
        
        # Perform self.n_gibbs_iter iterations of Gibbs sampling.
        for __ in range(self.n_gibbs_iter):
            v_free = self.gen_v_pvh(h_free, W, a)
            h_free = self.gen_h_phv(v_free, W, b)
        
        # Updated parameter vector theta using stochastic gradient algorithm.
        gradient_t = -self.gradient(v, h_data) + self.gradient(v_free, h_free)
        self.theta += self.learning_rate * gradient_t
        self.gradient_vec.append(np.mean(gradient_t ** 2) ** 0.5)
   
    def pretrain(self, d_n):
        """Pre-train the model using data set d_n.

        Parameters:
        ----------
        d_n : array of shape (n_v, n_samples)
            The data set to train the model on.

        Return value:
        -------
        model: Dictionary
            A dictionary containing fitted W, a, and b variables.
        
        gradient_vec: list
            Vector of gradient values generated during pre-training.
        """    
        n_v = d_n.shape[0]
        n_samples = d_n.shape[1]
        self.n_v = n_v
        n_h = self.n_h
        n_epoch = self.n_epoch
        batch_size = self.batch_size
        
        # Initialize W, a, and b using Gaussian RVs.
        W = self.rng.normal(0, 0.05, (n_v, n_h))
        a = self.rng.normal(-0.5, 0.05, (n_v, 1))
        b = self.rng.normal(-0.2, 0.05, (n_h, 1))
        
        self.theta = np.vstack((np.reshape(W, (n_v * n_h, 1), 'F'), a, b))

        # Vector of gradient values to keep track of the convergence of the algorithm.
        self.gradient_vec = []
        
        for k in range(n_epoch):
            
            # Generate Bernoulli RV based on the intensity of image pixels.
            d_n_sample = self.rng.random_sample((n_v, n_samples)) < (d_n + 0.5) / 256
            
            # Shuffle data set samples to obtain a random order.
            d_n_rand_order = d_n_sample[:, self.rng.permutation(n_samples)]
            
            # Divide data set into mini-batches and perform pre-trainig on each mini-patch.
            n_batch = 0
            gradient_sum = 0                  
            for n in range(int(np.floor(float(n_samples) / batch_size))):
                v = d_n_rand_order[:, n * batch_size : (n + 1) * batch_size]
                self.pretrain_(v)
                gradient_sum += self.gradient_vec[-1]
                n_batch += 1
                    
            if (self.verbose):
                print "epoch %d, average gradient magnitude: %f" % (k, float(gradient_sum)/n_batch)
                
        model = {}
        W = np.reshape(self.theta[0:n_v * n_h], (n_v, n_h), 'F')
        a = self.theta[n_v * n_h: n_v * n_h + n_v]
        b = self.theta[n_v * n_h + n_v: n_v * n_h + n_v + n_h]
        model['W'] = W
        model['a'] = a
        model['b'] = b
        return model, self.gradient_vec
