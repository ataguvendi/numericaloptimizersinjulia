import numpy as np
import jax
import jax.numpy as jnp
from jax import hessian
from jax import grad


def sgd(w: jnp.array, gradient_size: int, fn, X: jnp.array, z: jnp.array):
    """
    Implements stochastic gradient descent using JAX for gradient computation.

    Parameters:
        w (jnp.array): Current weights.
        gradient_size (int): Number of samples to use for gradient estimation.
        fn (function): Loss function that takes (X_row, w, z) and returns a scalar loss.
        X (jnp.array): Feature matrix.
        z (jnp.array): Target vector.

    Returns:
        jnp.array: The averaged gradient for the sampled mini-batch.
    """
    #Random indices
    sampled_indices = jnp.array(np.random.choice(X.shape[0], gradient_size, replace=False))
    gradients = jnp.zeros((gradient_size, w.shape[0])) #preallocation
    for i, index in enumerate(sampled_indices):
        sampled_row = X[index]
        target_value = z[index]
        loss_fn = lambda w: fn(sampled_row, w, target_value) #setup lambda function to use jax differentiation.
        gradients = gradients.at[i].set(grad(loss_fn)(w))
    return jnp.mean(gradients, axis=0) #division by gradient size step, take row-wise element average to vectorize process.


def she(w:np.array, hessian_size:float, fn, X:np.array, z:np.array):
    """
    Implements stochastic hessian for the SQN algorithm.

    Parameters:
        w (jnp.array): Current weights.
        hessian_size (int): Number of samples to use for hessian estimation.
        fn (function): Loss function that takes (X_row, w, z) and returns a scalar loss.
        X (jnp.array): Feature matrix.
        z (jnp.array): Target vector.

    Returns:
        jnp.array: The averaged hessian for the sampled mini-batch.
    """
    #sample indices
    sampled_indices = np.random.choice(X.shape[0], hessian_size, replace=False)
    sampled_rows = X[sampled_indices]
    #get the hessian using Jax based on weight as the independent variables.
    hessian_w_fn = hessian(lambda X_row, w: fn(X_row, w, z), argnums=1) #function to set up.
    hessians_w = [hessian_w_fn(X_row, w) for X_row in sampled_rows] #compute hessians by comprehension
    return np.mean(np.array(hessians_w),axis=0) #convert to np and divide by sample count by mean method.


def hessian_updating(t:int, M:int, s:np.array, y:np.array, eps:float = 1e-6):
    """
    Implements algorithm 2. Provides an approximate for H, inverse hessian using LBGFS.

    Inputs:
    Updating counter t
    Memory parameter M
    s and y in order to draw the correction pairs (sj,yj) NOTE: Each index in this algorithm is expected to be stored in a row where s and t are matrices. 
    So s[j] should be the jth row of s (1-base indexing fashions)
    eps: Epsilon, small tolerance to prevent 0 division, especially for BCEL
    Outputs:
    new H_t
    """

    #decrease t by one to combat OBO
    t -= 1
    #Step 1: Define algorithm parameters
    mbar = min(t,M)
    H = np.dot(s[t], y[t])/(np.dot(y[t],y[t])+eps) #prevent 0 div
    H = H * np.eye(len(s[0]))
    
    #loop
    for j in range(t-mbar+1,t+1):
        p = 1/(np.dot(y[j], s[j])+eps) #prevent 0 div
        #Apply BGFS formula
        H = (np.eye(H.shape[0])-p*np.outer(s[j], y[t])) @ H @ (np.eye(H.shape[1]) - p *np.outer(y[j], s[j])) + p*(np.outer(s[j], s[j]))
    return H


def sqn_algorithm(w:np.array, M:int, L:int, alpha:np.array, gradient_size:int, hessian_size:int, maxit:int, fn, X:np.array, z:np.array, verbose=False):
    """
    Implements the SQN Method (algorithm 1) of Bryd et al.
    Author: Ata Guvendi

    Inputs:
        w: Parameter vector w. W1 (W[0]) stores the initial parameter vector. W must be maxit x d, where d is the input dimension of the objective function PARAMETERS, f(.).
        M: Integer memory parameter passed to Hessian updater
        L: Parameter defines how often to caluclate correction pairs. If L is n, then we compute every n iterations.
        alpha: Vector of step sizes for gradient descent iteration. Could be a vector of the same numbers.
        gradient_size: Size of gradient sample for SGD
        hessian_size: Size of hessian for stochastic hessian updating.
        maxit: Maximum iteration limiter
        fn: Function object returns evaluation, f(X,w,z)
        X: NxM matrix where every column is a feature
        z: Nx1 vector where every row is a label
        verbose: If true, will print out iteration count at the start of every iteration
    
    
    Outputs:
        W so that every iteration is the next row of weights.
    """
    #Process inputs and initialize parameters
    assert w.shape[0] == maxit
    alpha = alpha.reshape(-1)
    z = z.reshape(-1,1)
    t = -1
    s = []
    y = []
    wbar = None
    wbar_previous = None
    #Mainloop
    for k in range(maxit-1):
        if verbose: print(f"Iteration {k}")
        elif k%10==0: print(f"Iteration {k}")
        stochastic_gradient = sgd(w[k], gradient_size,fn,X,z)
        #If sufficient correction pairs do not exist, default to SGD.
        if t < 1: 
            w[k+1] = w[k] - (alpha[k] * stochastic_gradient)
        else: 
            w[k+1] = w[k] - alpha[k]*(hessian_updating(t,M,s,y) @ stochastic_gradient)
        #correction pairs need to be computed every L iterations:
        if k % L == 0:
            t += 1 #set off hessian based iterations instead of SGD
            wbar_previous = wbar
            wbar = 0
            for i in range(k-L+2, k+1): wbar += w[i]
            wbar /= L #where wbar is the average of the last L weight vectors. (every entry is an entry-wise average.)
            if t>0:
                stochastic_hessian = she(wbar,hessian_size,fn,X,z) #retrieve stochastic hessian
                #calculate correction pairs:
                st = (wbar-wbar_previous).reshape(-1)
                yt = (stochastic_hessian @ st.reshape(-1,1)).reshape(-1)
                #add to collection:
                s.append(st)
                y.append(yt)
    return w