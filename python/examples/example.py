import pympcxx as mpc
import numpy as np

# Create a new LMPC object
nx = 4
nu = 2
ndu = 1
ny = 4

N = 10

lmpc = mpc.LMPC(nx, nu, ndu, ny, N, N)

# Let's define the model matrices for a simple double integrator
# x_{k+1} = A x_k + B u_k
# y_k = C x_k
# with u_k made of two inputs
A = np.array([[1.0, 0.1, 0.0, 0.0],
              [0.0, 1.0, 0.1, 0.0],
              [0.0, 0.0, 1.0, 0.1],
              [0.0, 0.0, 0.0, 1.0]])

B = np.array([[0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]])

C = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

lmpc.setStateSpaceModel(A, B, C)

# the weights for the cost function
y_cost_weight = np.array([1.0, 1.0, 1.0, 1.0])
u_cost_weight = np.array([1.0, 1.0])
delta_u_cost_weight = np.array([1.0, 1.0])

lmpc.setObjectiveWeights(y_cost_weight, u_cost_weight, delta_u_cost_weight, mpc.HorizonSlice.all())

# define the initial state
x0 = np.array([2.0, 10.0, 0.0, 0.0])
u0 = np.array([0.0, 0.0])

# solve once
lmpc.optimize(x0, u0)

# now let's simulate the system
x = x0
u = u0

for i in range(1000):
    # solve the optimization problem
    res = lmpc.optimize(x, u)

    # apply the first input
    u = res.cmd

    # simulate the system
    x = A @ x + B @ u

    print("x = ", x)
    print("u = ", u)

stats = lmpc.getExecutionStats()

print("Min solution time: " + str(stats.minSolutionTime.total_seconds()) + " seconds")
print("Max solution time: " + str(stats.maxSolutionTime.total_seconds()) + " seconds")
print("Avg solution time: " + str(stats.averageSolutionTime.total_seconds()) + " seconds")
print("Total solution time: " + str(stats.totalSolutionTime.total_seconds()) + " seconds")