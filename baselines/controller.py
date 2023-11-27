import dynamics
from dynamics import g, A, B, kT
import numpy as np
import scipy
from scipy.integrate import odeint

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)

####################### The controller ######################
def u(x, goal):
    goal = np.array(goal)
    return K.dot([goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x) + [0, 0, g / kT]

######################## The closed_loop system #######################
def cl_nonlinear(x, t, goal):
    x = np.array(x)
    dot_x = dynamics.f(x, u(x, goal))
    return dot_x

# simulate
def simulate(x, goal, dt):
    curr_position = np.array(x)[[0, 4, 8]]
    error = goal - curr_position
    distance = np.sqrt((error**2).sum())
    if distance > 1:
        goal = curr_position + error / distance
    return odeint(cl_nonlinear, x, [0, dt], args=(goal,))[-1]