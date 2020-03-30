import numpy as np
import matplotlib.pyplot as plt

def assembly_1d(f, x, e):
    """FEM assembly in 1d

    Parameters
    ----------
    f : load function

    p : numpy.array
        grid points
    e : numpy.ndarray
        elements as array of shape (k, 2), where k is number of elements
    """
    N = x.shape[0]
    k = e.shape[0]

    # Linear basis functions and derivatives
    def phi1(x):
        return 1-x

    def phi2(x):
        return x

    def dphi1(x):
        return -np.ones_like(x)

    def dphi2(x):
        return np.ones_like(x)

    # Use midpoint rule for integration
    t = np.array([0.5])
    w = np.array([1.0])

    # Evaluate basis functions at quadrature points
    phi_t = np.zeros((2, t.shape[0]))
    phi_t[0,:] = phi1(t)
    phi_t[1,:] = phi2(t)
    dphi_t = np.zeros((2, t.shape[0]))
    dphi_t[0,:] = dphi1(t)
    dphi_t[1,:] = dphi2(t)

    # System matrix and load vecor
    print(N)
    b = np.zeros(N)
    print(b)
    A = np.zeros((N, N))

    for i in range(k):
        # Element grid points
        n1 = x[e[i, 0]] 
        n2 = x[e[i, 1]]

        # Element size
        h = abs(n2-n1)

        # Mapped quadrature point
        te = n1 + (n2-n1)*t

        # Compute entries of the load vector b
        for p in range(2):
            for k in range(len(t)):
                b[e[i, p]] = b[e[i, p]] + f(te[k])*phi_t[p, k]*h*w[k]

        Ae = np.zeros((2, 2))
        # Compute entries of A
        for p in range(2):
            for q in range(2):
                for k in range(len(t)):
                    Ae[p, q] = Ae[p, q] + (1/h)*dphi_t[p, k]*(1/h)*dphi_t[q, k]*h*w[k]

        # Move to A
        for p in range(2):
            for q in range(2):
                # print(e[i, p])
                A[e[i, p], e[i, q]] = A[e[i, p], e[i, q]] + Ae[p, q]

    return (A, b)
     

if __name__ == '__main__':
    # Load function
    def f(x):
        return 8*np.ones_like(x)

    # Uniform grid test
    p = np.linspace(0.0, 1.0, 5)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])

    A, b = assembly_1d(f, p, e)
    print('Uniform test:')
    print('============================')    
    print('System matrix A:')
    print(A)
    print('Load vector b:')
    print(b)

    # Non uniform grid test
    p1 = np.linspace(0.0, 0.5, 3)
    p2 = np.linspace(0.5, 1.0, 5)
    p = np.concatenate((p1, p2[1:]))
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    A, b = assembly_1d(f, p, e)
    print('Non uniform test:')
    print('============================')
    print('System matrix A:')
    print(A)
    print('Load vector b:')
    print(b)