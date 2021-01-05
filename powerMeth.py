import numpy as np

# The power method algorithm
# for computing lambda_1
def power_method(A, x0, maxiter=1):
    x = x0
    n = A.shape[0]
    for k in range(maxiter):
        y = np.matmul(A,x)
        normalization_factor = 1.0 / np.linalg.norm(y,2)
        x = normalization_factor* y
        Ax = np.matmul(A,x)
        rayleigh = np.matmul(x.T, Ax)
        print("iteration: ", k+1)
        print("y: ", y.T)
        print("x: ", x.T)
        print("rayleigh:", rayleigh)
        print("")
    return x, rayleigh

# define matrix A
# compute its egeinvalues and eigenvectors for comparison
A = np.float64(np.array([[12,2,-1],
                         [2,12,2],
                         [-1,2,12]]))
eigenvalues, v = np.linalg.eigh(A)

x0 = np.float64(np.array([0.5,0.5,-0.5])) # input vector
x, eigenvalue = power_method(A,x0,3)