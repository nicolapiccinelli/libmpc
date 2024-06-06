import pympcxx as mpc
import numpy as np

# Create a new LMPC object
nx = 2
nu = 1
ny = 2
ph = 10
ch = 5
ineq_c = ph + 1
eq = 0

ts = 0.1

nlmpc = mpc.NLMPC(nx, nu, ny, ph, ch, ineq_c, eq)
nlmpc.setLoggerLevel(mpc.LoggerLevel.NORMAL)
nlmpc.setDiscretizationSamplingTime(ts)

def model(dx,x,u):
    try:
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0]
        dx[1] = x[0]
    except Exception as e:
        print("model ex: ", e)

def cost(x,y,u,slack):
    try:
        c = np.sum(np.square(x)) + np.sum(np.square(u))
        return c
    except Exception as e:
        print("cost ex:", e)

def state_space(x,u,i):
    try:
        dx = np.zeros(nx)
        model(dx, x, u)        
        return dx
    except Exception as e:
        print("state ex: ", e)

def ineq_con(x,y,u,slack):
    try:
        in_con = np.zeros(ineq_c)

        for i in range(ineq_c):
            in_con[i] = u[i, 0] - 0.5
            
        return in_con
    except Exception as e:
        print("ineq ex: ", e)

nlmpc.setStateSpaceFunction(state_space, 1e-10)
nlmpc.setObjectiveFunction(cost)
nlmpc.setIneqConFunction(ineq_con, 1e-10)

m_x = np.array([0.0, 1.0])
m_dx = np.array([0.0, 0.0])
m_u = np.array([0.0])

while True:
    # solve the optimization problem
    res = nlmpc.optimize(m_x, m_u)

    # apply the first input
    m_u = res.cmd

    model(m_dx, m_x, m_u)

    # update the state
    m_x = m_x + m_dx * ts

    if abs(m_x[0]) <= 1e-2 and abs(m_x[1]) <= 1e-1:
        break

print("m_x: ", m_x)

stats = nlmpc.getExecutionStats()

print("Min solution time: " + str(stats.minSolutionTime.total_seconds()) + " seconds")
print("Max solution time: " + str(stats.maxSolutionTime.total_seconds()) + " seconds")
print("Avg solution time: " + str(stats.averageSolutionTime.total_seconds()) + " seconds")
print("Total solution time: " + str(stats.totalSolutionTime.total_seconds()) + " seconds")