import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mesh
import time

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
    b = np.zeros(N)
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
        for k in range(len(t)):
            b[e[i, :]] = b[e[i, :]] + f(te[k])*phi_t[:, k]*h*w[k]

        Ae = np.zeros((2, 2))
        # Compute entries of A
        for p in range(2):
            for q in range(2):
                for k in range(len(t)):
                    Ae[p, q] = Ae[p, q] + (1/h)*dphi_t[p, k]*(1/h)*dphi_t[q, k]*h*w[k]

        # Move to A
        for p in range(2):
            for q in range(2):
                A[e[i, p], e[i, q]] = A[e[i, p], e[i, q]] + Ae[p, q]

    return A, b

def assembly_2d(f, x, e):
    """FEM assembly in 2d with linear basis functions. Based on the example
     implementation on the course.

    Parameters
    ----------
    f : load function

    p : numpy.array
        grid points as array of shape (n, 2), where n is number of points
    e : numpy.ndarray
        elements as array of shape (k, 3), where k is number of elements
    """
    n_nodes = x.shape[0]

    # Basis functions on the reference element
    phi_1 = lambda x: 1 - x[:,0] - x[:,1]
    phi_2 = lambda x: x[:,0]
    phi_3 = lambda x: x[:,1]

    # Quadrature weights and positions for reference
    w = (1/6)*np.ones(3)
    q = np.array([[0.5, 0], [0.5, 0.5], [0, 0.5]])

    # Values of basis functions and derivatives on reference element
    phis = np.array([phi_1(q), phi_2(q), phi_3(q)])
    d_phis = np.array([[-1, -1], [1, 0], [0, 1]])

    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for elem in e:
        # Get nodes of element
        ns = x[elem,:]

        # Calculate matrix and bias for affine tranform
        M = np.array([ns[1,:]-ns[0,:], ns[2,:]-ns[0,:]])
        d = ns[0,:]

        # Precalculate M^(-1) and B = M^(-1)*M(-T)
        inv_M = np.linalg.inv(M)
        B = np.dot(inv_M, inv_M.T)
        
        # Calculate the local A matrix
        A_loc = abs(np.linalg.det(M)) * np.sum(w) * np.dot(d_phis, np.dot(B, d_phis.T))
        
        # Add to global A matrix
        for i in range(3):
            for j in range(3):
                A[elem[i], elem[j]] += A_loc[i,j]

        # Quadrature points on element
        q_g = np.dot(q, M) + d
        
        # Calculate local b vector
        b_loc = np.dot(phis, w * f(q_g.T)) * abs(np.linalg.det(M))

        # Add to global b vector
        b[elem] += b_loc

    return A, b

def apply_dirichlet(A, b, boundary_idx, interior_idx, g=0):
    A_ii = A[np.ix_(interior_idx, interior_idx)]
    #A_ib = A[interior_idx, boundary_idx]
    b_i = b[interior_idx]
    return A_ii, b_i

if __name__ == '__main__':
    
    def f(x):
        return np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])

    mesh = mesh.uniform_mesh(5)
    start = time.time()
    A, b = assembly_2d(f, mesh.nodes, mesh.elements)
    end = time.time()
    A_ii, b_i = apply_dirichlet(A, b, mesh.boundary_node_idx, mesh.interior_node_idx)
    u = np.zeros(mesh.n_nodes)
    u[mesh.interior_node_idx] = np.linalg.solve(A_ii, b_i)
    print(f'Time: {end-start}')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(mesh.nodes[:,0], mesh.nodes[:,1], u)
    plt.show()
    

    """
    # Load function
    def f(x):
        return 8*np.ones_like(x)

    def sol_fun(t):
        return -4*t**2+4*t

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
    p2 = np.linspace(0.5, 1.0, 4000)
    p = np.concatenate((p1, p2[1:]))
    n = np.arange(p.shape[0])
    n1 = n[0:n.shape[0]-1]
    n2 = n[1:]
    e = np.transpose(np.vstack((n1, n2)))
    # e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    start = time.time()
    A, b = assembly_1d(f, p, e)
    end = time.time()
    print(end - start)
    # print('Non uniform test:')
    # print('============================')
    # print('System matrix A:')
    # print(A)
    # print('Load vector b:')
    # print(b)

    N = A.shape[0]
    A = A[1:(N-1), 1:(N-1)]
    b = b[1:(N-1)]

    u = np.linalg.solve(A, b)
    u = np.concatenate(([0.0], u, [0.0]))

    g = np.linspace(0.0, 1.0, 1000)
    u_real = sol_fun(g)

    plt.figure()
    plt.plot(g, u_real)
    plt.plot(p, u)
    plt.show()
    """