import numpy as np
import matplotlib.pyplot as plt

import utils

if __name__ == '__main__':


    def u(t):
        return -4*t**2+4*t
    def f(t):
        return np.ones_like(t)*8

    """
    ###############################
    # Test restriction operator
    ###############################
    i = 5
    n = 2**i-1
    t = np.linspace(0.0, 1.0, 2**i+1)[1:-1]

    x = -4*t**2 + 4*t
    tr = t
    xr = x

    plt.figure()
    plt.plot(np.concatenate(([0], tr, [1])), np.concatenate(([0], xr, [0])))

    for k in range(2):
        tr, xr = utils.restrict(tr, xr)
        print(xr)
        plt.plot(np.concatenate(([0], tr, [1])), np.concatenate(([0], xr, [0])))    

    plt.title('Test of restriction operator')
    plt.savefig('restrict.png')
    
    ###############################
    # Test interpolation operator
    ###############################
    i = 3
    n = 2**i-1
    t = np.linspace(0.0, 1.0, 2**i+1)[1:-1]

    x = -4*t**2 + 4*t

    ti = t
    xi = x

    plt.figure()
    plt.plot(ti, xi)

    ti, xi = utils.interpolate(ti, xi)
    plt.plot(ti, xi, 'rx')

    plt.title('Test of interpolation operator')
    plt.savefig('interpolate.png')

    # ###############################
    # # Test solution operator
    # ###############################
    i = 6
    n = 2**i-1
    t = np.linspace(0.0, 1.0, 2**i+1)[1:-1]

    # System matrix and RHS
    A = utils.system_matrix_1d(i)
    b = f(t)

    # Correct solution
    u = u(t)

    # Initial guess x = 0
    x0 = np.zeros_like(b)
    x0 = u + (np.random.random(size=u.shape)-0.5)*1.0 + np.sin(4*np.pi*t)*0.6

    # Allocate array for solutions
    N = 50001
    uk = np.zeros((N, u.shape[0]))
    uk[0,:] = x0

    plt.figure()
    # plt.plot(t, u)
    plt.plot(t, uk[0,:]-u)

    for k in range(1, N):
        uk[k,:] = utils.smooth(uk[k-1,:], A, b)
        if k == 1:
            plt.plot(t, uk[k,:]-u)
        if k == 2:
            plt.plot(t, uk[k,:]-u)

    plt.title('Error (red) smooth after small number of iterations (blue)')
    plt.savefig('smooth.png')
    """

    # ###############################
    # # Test Multi-grid V-cycle
    # ###############################
    i = 6
    n = 2**i-1
    t = np.linspace(0.0, 1.0, 2**i+1)[1:-1]

    # System matrix and RHS
    A = utils.system_matrix_1d(i)
    b = f(t)

    # Correct solution
    u = u(t)

    # Initial guess x = 0
    x0 = np.zeros_like(b)
    x0 = u + (np.random.random(size=u.shape)-0.5)*1.0 + np.sin(4*np.pi*t)*0.6

    x = utils.v_cycle(x0, b, i)

    plt.figure()
    plt.plot(t, u)
    plt.plot(t, x)

    plt.figure()
    plt.plot(t, x0 - u)
    plt.plot(t, x - u)
    plt.show()


    # ###############################
    # # Test full Multi-Grid
    # ###############################

    x = utils.full_multigrid(f, i)
    
    plt.figure()
    plt.plot(t, u)
    plt.plot(t, x)
    plt.show()

