import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# Let's begin with the P operators
def PO_operator(Amat, setO):
    """
    Set all elements that are not in setO to 0. It assumes that setO
    comes from applying np.nonzero(A==condition) to a matrix.
    """
    Anew = np.zeros_like(Amat)
    Anew[setO] = Amat[setO]

    return Anew

def POcomp_operator(Amat, setO):
    """
    The complement of PO_operator.
    Set all elements that are in setO to 0. It assumes that setO
    comes from applying np.nonzero(A==condition) to a matrix.
    """
    Anew = np.copy(Amat)
    Anew[setO] = 0

    return Anew

# Lets code the shrink operator
def shrink(Amat, lamb=0, doprint=False):
    """
    This generates a reduced version of A given by the singular value decomposition.
    It only takes the singular above lamb.
    """
    U, Sigma, VT = np.linalg.svd(Amat, full_matrices=False)
    
    if(doprint): print(Sigma)
    
    Sigma[Sigma < lamb] = 0
    
    return U@np.diag(Sigma)@VT

# The loss function
def mcnnm_loss(y_true, y_pred, setO, doprint=False):
    Ocardinality = len(setO[0])
    diffmat = y_true - y_pred
    if(doprint): print(diffmat)
    outmat = PO_operator(diffmat, setO)**2
    return (outmat.sum().sum()/Ocardinality)**0.5

# The "Matrix-Completion with Nuclear Norm Minimization" estimator
class MatrixCompletion_NNM(BaseEstimator, TransformerMixin):
    """
    This implements the iterative procedure to estimate L. 
    Since L is really the matrix Y, but with the missing values
    imputed, we chose the Transformer 'Mixin'.

    Parameters
    ----------
    setOk : array-like
        This comes, for example, from applying np.nonzero(A==condition) to a matrix
    
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
        
    lamb : int (default=0)
        This is the regularization parameter, which should be non-negative since
        it is a minimization procedure.
        
    epsilon : float (default=0.001)
        Desired accuracy of the estimation.
    
    max_iters : integer (default=100)
    
    doprint : boolean, optional (default=False)
        Prints different results of the estimation steps.
        
    printbatch : integer, optional if doprint==True (default=10)
        After how many iterations to print the loss function.
        
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    

    Attributes
    ----------
    Lest_ : array-like, shape (N, T)
        This is the estimate of L.
    
    loss_ : float
        The root-square of the mean square error of the observed elements.
        
    iters_ : int
        Number of iterations it took the algorithm to get the desired
        precision given by the parameter 'epsilon'.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1,2,np.nan, 0],[np.nan,2,2,1],[3,0,1,3],[0,0,2,np.nan]])
    >>> observedset = np.nonzero(~np.isnan(data))
    >>> my_mcnnm = MatrixCompletion_NNM(setOk=observedset, lamb=2.5, epsilon=10**(-6), doprint=False)
    >>> print(data)
    [[  1.   2.  nan   0.]
    [ nan   2.   2.   1.]
    [  3.   0.   1.   3.]
    [  0.   0.   2.  nan]]
    >>> print(my_mcnnm.fit(data))
    MatrixCompletion_NNM(copy=True, doprint=False, epsilon=1e-06, lamb=2.5,
               max_iters=100, printbatch=10,
               setOk=(array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int64), 
               array([0, 1, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=int64)))
    >>> print(my_mcnnm.transform(data))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    >>> print(my_mcnnm.fit_transform(data))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    >>> print(my_mcnnm.transform([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]))
    [[ 0.98082397  2.00915981  3.49022202  0.01813745]
     [ 1.55601129  1.33063721  2.38547663  0.97025289]
     [ 3.00567286  0.33134019  0.80761928  3.00948113]
     [ 0.0419737   0.87320627  1.48553996 -0.39839039]]
    
    Notes
    -----
    
    
    """
    

    def __init__(self, setOk=None, #missing_values=np.nan, 
                 lamb=0, epsilon=0.001, max_iters=100, 
                 doprint=False, printbatch=10, copy=True):
        self.setOk = setOk 
        #self.missing_values = missing_values 
        self.lamb = lamb 
        self.epsilon = epsilon 
        self.max_iters = max_iters 
        self.doprint = doprint 
        self.printbatch = printbatch
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the estimator to the matrix X (which is 
        really the matrix Y in Athey et al.'s paper).
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : Ignored
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
            
        Returns
        -------
        self : object
            Returns self.
        """
        
        # First, I should check that the missing values are right
        #Ynew = np.zeros_like(self.Ymat)
        #Ynew = self.missing_values
        #Ynew[self.setOk] = self.Ymat[self.setOk]

        assert (len(self.setOk) > 0), "setOk should be a non-empty array"
        assert (self.lamb > 0), "lamb is the lambda parameter which should be larger than zero"

        N, T = X.shape

        # Initialize L to the observed (non-missing) values of Y given by the set setOk
        Lprev = PO_operator(X, self.setOk)

        # Initialization of error with a highvalue and the iteration
        error = N*T*10**3
        iteration = 0

        while((error > self.epsilon) and (iteration < self.max_iters)):
            Lnext = shrink(PO_operator(X, self.setOk) + 
                           POcomp_operator(Lprev, self.setOk), lamb = self.lamb)

            # Updating values
            Lprev = Lnext.copy()
            error = mcnnm_loss(X, Lprev, self.setOk)
            iteration = iteration + 1

            if(self.doprint and (iteration%self.printbatch==0 or iteration==1)):
                print("Iteration {}\t Current loss: {}".format(iteration, error))

        if(self.doprint):
            print("")
            print("Final values:")
            print("Iteration {}\t Current loss: {}".format(iteration, error))
            print("")
            print(X)
            print(np.round(Lnext, 2))
        
        self.iters_ = iteration
        self.loss_ = error
        self.Lest_ = Lnext
        
        # Return the transformer
        return self

    def transform(self, X):
        """ 
        Actually returning the estimated matrix, in which we have 
        imputed the missing values of X (matrix Y).
        
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (N, T)
            The input data to complete.
            
        Returns
        -------
        X_transformed : array, shape (N, T)
            The array the completed matrix.
        """
        try:
            getattr(self, "Lest_")
        except AttributeError:
            raise RuntimeError("You must estimate the model before transforming the data!")
        
        return self.Lest_

def simul_adopt(Ymat, treated_units, t0):
    """
    The function assumes that t0 are integer values that go
    from 1 to T, where T is the wide length of the matrix Ymat.
    """
    Ynew = Ymat.copy()
    Ynew[treated_units, t0:] = np.nan
    return Ynew

def stag_adopt(Ymat, treated_units, t0):
    """
    Here we assume that units get treated at times 
    t after t0, chosen uniformly at random from the 
    remaining times.
    """
    N, T = Ymat.shape
    ts = np.random.choice(np.arange(t0,T), size = len(treated_units))
    Ynew = Ymat.copy()
    for i, unit in enumerate(treated_units):
        Ynew[unit,ts[i]:] = np.nan
    return Ynew

# FUNCTIONS FOR CROSS-VALIDATION
def mcnnm_kfold_CV(Yreal, Yobs, observedset, lambda_cv, num_splits=10):
    kfold = KFold(n_splits=num_splits, shuffle=True)
    vec_errors = np.zeros(num_splits)
    for i, train_test_tuple in enumerate(kfold.split(observedset)):
        O_train_i = tuple(observedset[train_test_tuple[0]].T)
        O_valid_i = tuple(observedset[train_test_tuple[1]].T)

        # Initializing the MC-NNM object
        my_mcnnm_i = MatrixCompletion_NNM(setOk=O_train_i, 
                                    lamb=lambda_cv, epsilon=10**(-6))

        # Fitting (/transforming)
        Lest = my_mcnnm_i.fit_transform(Yobs)

        # Error on validation set
        vec_errors[i] = mcnnm_loss(Yreal, Lest, O_valid_i)
        
    return vec_errors

def mcnnm_GridSearch(Yreal, Yobs, observedset, lambda_vec, num_splits=10, doprint=True):
    mat_errors = np.zeros((len(lambda_vec), num_splits))
    for i, lambda_cv in enumerate(lambda_vec):
        mat_errors[i,:] = mcnnm_kfold_CV(Yreal, Yobs, observedset, lambda_cv, num_splits=num_splits)
    
    
    
    df_results = pd.DataFrame(mat_errors,
                             index = ["Lambda = {}".format(np.round(l, 2)) for l in lambda_vec],
                             columns = ["Fold {}".format(i+1) for i in range(num_splits)])
    
    vec_rmse = df_results.mean(axis=1)
    best_lambda = lambda_vec[np.argmin(np.array(vec_rmse))]
    
    if doprint:
        print("\nMatrix of errors:")
        print("-----------------")
        print(df_results)
        print("\nAverage RMSE across folds:")
        print("--------------------------")
        print(vec_rmse)
        print("\nBest parameter value (returned):")
        print("--------------------------------")
        print(np.round(best_lambda, 3))
    
    return best_lambda
    



