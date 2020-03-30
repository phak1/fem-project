import numpy as np
import matplotlib.pyplot as plt

# See:
# https://people.eecs.berkeley.edu/~demmel/cs267/lecture25/lecture25.html

def restrict(x, sparse=False):
    """Applies the restriction operator to vector x and grid points t. 

    Parameters
    ----------
    t : numpy.array
        gridpoints on fine grid
    x : numpy.array
        vector on finer grid

    Returns
    -------
    numpy.array
        grid points on coarse grid    
    numpy.array
        vector on coarse grid
    """
    n = x.shape[0]
    nr = int((n-1)/2)
    R = np.zeros((nr, n))
    R[:,0] = 0.25
    R[:,1] = 0.5
    R[:,2] = 0.25

    # Roll rows of R
    r = np.arange(0, 2*R.shape[0], 2)
    rows, column_indices = np.ogrid[:R.shape[0], :R.shape[1]]
    r[r < 0] += R.shape[1]
    column_indices = column_indices - r[:, np.newaxis]
    R = R[rows, column_indices]
    x_restricted =  np.matmul(R, x)

    return x_restricted

    
def interpolate(x, sparse=False):
    """Applies the interpolation operator to vector x

    Parameters
    ----------
    t : numpy.array
        gridpoints on coarse grid
    x : numpy.array
        vector on coarse grid

    Returns
    -------
    numpy.array
        grid points on fine grid    
    numpy.array
        vector on fine grid
    """
    # Pad with zeros
    x_padded = np.concatenate(([0], x, [0]))
    ni = 2*x.shape[0]+1
    I = np.zeros((ni, x_padded.shape[0]))
    I[::2,0:2] = [0.5, 0.5]
    I[1::2,0] = 1

    # Roll rows of I
    r = np.zeros(I.shape[0]+1)
    r[::2] = np.arange((r.shape[0])/2)
    r[1::2] = np.arange(r.shape[0]/2)
    r = r[1:]

    for k in range(r.shape[0]):
        I[k,:] = np.roll(I[k,:], int(r[k]))

    return np.matmul(I, x_padded)



def system_matrix_1d(i, sparse=False):
    """Returns system matrix for solving 1D poisson equation.
    i is the number of grid points and matrix dimension is (2^i-1)x(2^i-1)

    Parameters
    ----------
    i : int
        The dimension of the matrix is (2^i-1)x(2^i-1)

    sparse : bool, optional
        set to True to use sparse matrix
    """
    if not sparse:
        A = np.diag(np.repeat(2, 2**i-1))
        A += np.diag(np.repeat(-1, 2**i-2), 1) + np.diag(np.repeat(-1, 2**i-2), -1)
        return A*(4**(i))

def smooth(x, A, b, n_iter=1, sparse=False):
    """Applies the weighted Jacobi in 1d on the approximate 
    solution x for Ax=b.

    Parameters
    ----------
    x : numpy.ndarray
        approximate solution
    A : numpy.ndarray
        system matrix
    b : numpy.ndarray
        rhs of equation

    Returns
    -------
    numpy.ndarray
        column vector of the smoothed solution
    """
    if not sparse:
        for i in range(n_iter):
            w = 2.0/3.0
            D = np.diag(np.diag(A)) 
            D_inv = np.diag(1/np.diag(A))
            L_U = A - D
            x1 = w*np.matmul(D_inv, b - np.matmul(L_U, x)) + (1-w)*x
        return x1
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    i = 4
    n = 2**i-1
    t = np.linspace(0.0, 1.0, 2**i+1)[1:-1]
    x = -4*t**2 + 4*t

    xr1 = restrict(x)
    h = 1 / (len(xr1) + 1)
    tr1 = np.arange(h, 1, h)

    xr0 = interpolate(xr1)


    plt.figure()
    plt.plot(t, x)
    plt.plot(tr1, xr1)
    plt.plot(t, xr0, 'rx')
    plt.show()

