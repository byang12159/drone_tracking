import sys
import pickle5 as pickle
import numpy as np
from scipy.optimize import least_squares
from vision.cam2drone import get_T_DC
from utils import transformations

with open(sys.argv[1], 'rb') as handle:
    data = pickle.load(handle)

marker_gt = np.array([-0.07250241851806641, -0.009298406600952148, 0.]).reshape(1, -1)

pose_C = np.array([d[0] for d in data]) # N x 4 x 4
pose_W = np.array([d[4] for d in data]) # N x 4 x 4

# T_WD = transformations.quaternion_matrix(q)
# T_WD[:3, 3] = snap_state[:3]
T_WD = np.array([d[1] for d in data]) # N x 4 x 4



idx = np.where(((pose_W[:,:3,3] - marker_gt)**2).sum(axis=1)<0.1)[0]
print('# raw samples: %d, # filtered samples: %d'%(pose_W.shape[0], len(idx)))
pose_C = pose_C[idx,:,:]
pose_W = pose_W[idx,:,:]
T_WD = T_WD[idx,:,:]

def loss(_T_DC, verbose=False):
    q = _T_DC[:4]
    t = _T_DC[4:]
    T_DC = transformations.quaternion_matrix(q)
    T_DC[:3, 3] = t
    T_DC = T_DC.reshape(1, 4, 4)

    pose_W = np.matmul(T_WD, np.matmul(T_DC, pose_C))
    position_W = pose_W[:, :3, 3]

    if verbose:
        return position_W
    
    return ((position_W - marker_gt)**2).sum()

# Initial Guess
x0 = np.zeros(7)
x0[:4] = transformations.quaternion_from_matrix(get_T_DC())
x0[4:] = get_T_DC()[:3, 3]

print(f"Initial Loss: {loss(x0)}")
res = least_squares(loss, x0)
print(loss(res.x))
q=res.x[:4]
t=res.x[4:]
euler = transformations.euler_from_quaternion(q, 'syxz')

print(res.x.tolist())
# print(res.cost)
# print(res.optimality)