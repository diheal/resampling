import tensorflow as tf
import numpy as np
import random
from scipy.interpolate import interp1d
#Part of the code reference https://github.com/iantangc/ContrastiveLearningHAR

def resampling_fast(x,M,N):
    time_steps = x.shape[1]
    raw_set = np.arange(time_steps)
    interp_steps = np.arange(0, raw_set[-1] + 1e-1, 1 / (M + 1))
    x_interp = interp1d(raw_set, x, axis=1)
    x_up = x_interp(interp_steps)

    length_inserted = x_up.shape[1]
    start = random.randint(0, length_inserted - time_steps * (N + 1))
    index_selected = np.arange(start, start + time_steps * (N + 1), N + 1)
    return x_up[:, index_selected, :]

def resampling_fast_random(x):
    M, N = random.choice([[1, 0], [2, 1], [3, 2]])
    time_steps = x.shape[1]
    raw_set = np.arange(x.shape[1])
    interp_steps = np.arange(0, raw_set[-1] + 1e-1, 1 / (M + 1))
    x_interp = interp1d(raw_set, x, axis=1)
    x_up = x_interp(interp_steps)

    length_inserted = x_up.shape[1]
    start = random.randint(0, length_inserted - time_steps * (N + 1))
    index_selected = np.arange(start, start + time_steps * (N + 1), N + 1)
    return x_up[:, index_selected, :]

def resampling(x,M,N):
    '''
    :param x: the data of a batch,shape=(batch_size,timesteps,features)
    :param M: the number of  new value under tow values
    :param N: the interval of resampling
    :return: x after resampling，shape=(batch_size,timesteps,features)
    '''
    assert M>N,'the value of M have to greater than N'

    timesetps = x.shape[1]

    for i in range(timesetps-1):
        x1 = x[:,i*(M+1),:]
        x2 = x[:,i*(M+1)+1,:]
        for j in range(M):
            v = np.add(x1,np.subtract(x2,x1)*(j+1)/(M+1))
            x = np.insert(x,i*(M+1)+j+1,v,axis=1)

    length_inserted = x.shape[1]
    start = random.randint(0,length_inserted-timesetps*(N+1))
    index_selected = np.arange(start,start+timesetps*(N+1),N+1)
    return x[:,index_selected,:]
    return x

def resampling_random(x):
    import random
    M = random.randint(1, 3)
    N = random.randint(0, M - 1)
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]

    for i in range(timesetps - 1):
        x1 = x[:, i * (M + 1), :]
        x2 = x[:, i * (M + 1) + 1, :]
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            x = np.insert(x, i * (M + 1) + j + 1, v, axis=1)
    length_inserted = x.shape[1]
    num = x.shape[0]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    x_selected=x[0,index_selected,:][np.newaxis,]
    for k in range(1,num):
        start = random.randint(0, length_inserted - timesetps * (N + 1))
        index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
        x_selected = np.concatenate((x_selected,x[k,index_selected,:][np.newaxis,]),axis=0)
    return x_selected


def noise(x):
    x = tf.add(x,tf.multiply(x,tf.cast(tf.random.uniform(shape = (x.shape[0],x.shape[1],x.shape[2]),minval=-0.1,maxval=0.1),tf.float64)))
    return x

def rotate(x,angles=np.pi/12):
    t = angles
    f = angles
    r = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(t), -np.sin(t)],
                   [0, np.sin(t), np.cos(t)]])
    Ry = np.array([[np.cos(f), 0, np.sin(f)],
                   [0, 1, 0],
                   [-np.sin(f), 1, np.cos(f)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    c = x.shape[2]//3
    x_new = np.matmul(np.matmul(np.matmul(Rx,Ry),Rz),np.transpose(x[:,:,0:3],(0,2,1))).transpose(0,2,1)
    for i in range(1,c):
        temp = np.matmul(np.matmul(np.matmul(Rx,Ry),Rz),np.transpose(x[:,:,i*3:i*3+3],(0,2,1))).transpose(0,2,1)
        x_new = np.concatenate((x_new,temp),axis=-1)
    return x_new


def scaling(x):
    alpha = np.random.randint(7,10)/10
    # alpha = 0.9
    return tf.multiply(x,alpha)
#
def magnify(x):
    lam = np.random.randint(11,14)/10
    return tf.multiply(x,lam)


def inverting(x):
    return np.multiply(x,-1)
def reversing(x):
    return x[:,-1::-1,:]


def rotation(x):
    c = x.shape[2]//3
    x_new = rotation_transform_vectorized(x[:,:,0:3])
    for i in range(1,c):
        temp = rotation_transform_vectorized(x[:,:,i*3:(i+1)*3])
        x_new = np.concatenate((x_new,temp),axis=-1)
    return x_new
def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

    return np.matmul(X, matrices)

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

