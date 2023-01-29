import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=28)
mpl.rc('xtick', labelsize=24)
mpl.rc('ytick', labelsize=24)

import math
import time 
from mpl_toolkits.mplot3d import Axes3D
from halos_exchange import HaloExchange

###############################################
dt = 0.5
dx = 1.0
dy = 1.0
dz = 1.0
Re = 1/20
ub = 1
nx = 128
ny = 128
nz = 128
ratio = int(nx/nz)
# nlevel = int(math.log(nz, 2)) + 1 

pd1 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
pd2 = [[3/26, 6/26,  3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26,  3/26]]
pd3 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
w1 = np.zeros([1,3,3,3,1])
w1[0,0,:,:,0] = np.array(pd1)*dt*Re/dx**2
w1[0,1,:,:,0] = np.array(pd2)*dt*Re/dx**2 
w1[0,2,:,:,0] = np.array(pd3)*dt*Re/dx**2 

p_div_x1 = [[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]]
p_div_x2 = [[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]]
p_div_x3 = [[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]]

p_div_y1 = [[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]]
p_div_y2 = [[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]]
p_div_y3 = [[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]]

p_div_z1 = [[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]]
p_div_z2 = [[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]]
p_div_z3 = [[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]]
w2 = np.zeros([1,3,3,3,1])
w3 = np.zeros([1,3,3,3,1])
w4 = np.zeros([1,3,3,3,1])

w2[0,0,:,:,0] = np.array(p_div_x1)*dt/dx*0.5
w2[0,1,:,:,0] = np.array(p_div_x2)*dt/dx*0.5
w2[0,2,:,:,0] = np.array(p_div_x3)*dt/dx*0.5

w3[0,0,:,:,0] = np.array(p_div_y1)*dt/dx*0.5
w3[0,1,:,:,0] = np.array(p_div_y2)*dt/dx*0.5
w3[0,2,:,:,0] = np.array(p_div_y3)*dt/dx*0.5 

w4[0,0,:,:,0] = np.array(p_div_z1)*dt/dx*0.5 
w4[0,1,:,:,0] = np.array(p_div_z2)*dt/dx*0.5
w4[0,2,:,:,0] = np.array(p_div_z3)*dt/dx*0.5

pA1 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
pA2 = [[3/26, 6/26,  3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26,  3/26]]
pA3 = [[2/26, 3/26,  2/26],
       [3/26, 6/26,  3/26],
       [2/26, 3/26,  2/26]]
w5 = np.zeros([1,3,3,3,1])
w5[0,0,:,:,0] = -np.array(pA1)/dx**2
w5[0,1,:,:,0] = -np.array(pA2)/dx**2 
w5[0,2,:,:,0] = -np.array(pA3)/dx**2 

pctyu1 = [[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]]
pctyu2 = [[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]]
pctyu3 = [[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]]

pctyv1 = [[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]]
pctyv2 = [[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]]
pctyv3 = [[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]]

pctyw1 = [[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]]
pctyw2 = [[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]]
pctyw3 = [[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]]
w6 = np.zeros([1,3,3,3,1])
w7 = np.zeros([1,3,3,3,1])
w8 = np.zeros([1,3,3,3,1])
w9 = np.zeros([1,2,2,2,1])
w6[0,0,:,:,0] = np.array(pctyu1)/(dx*dt)*0.5
w6[0,1,:,:,0] = np.array(pctyu2)/(dx*dt)*0.5
w6[0,2,:,:,0] = np.array(pctyu3)/(dx*dt)*0.5
w7[0,0,:,:,0] = np.array(pctyv1)/(dx*dt)*0.5
w7[0,1,:,:,0] = np.array(pctyv2)/(dx*dt)*0.5
w7[0,2,:,:,0] = np.array(pctyv3)/(dx*dt)*0.5 
w8[0,0,:,:,0] = np.array(pctyw1)/(dx*dt)*0.5
w8[0,1,:,:,0] = np.array(pctyw2)/(dx*dt)*0.5
w8[0,2,:,:,0] = np.array(pctyw3)/(dx*dt)*0.5
w9[0,:,:,:,0] = 0.125

kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
kernel_initializer_3 = tf.keras.initializers.constant(w3)
kernel_initializer_4 = tf.keras.initializers.constant(w4)
kernel_initializer_5 = tf.keras.initializers.constant(w5)
kernel_initializer_6 = tf.keras.initializers.constant(w6)
kernel_initializer_7 = tf.keras.initializers.constant(w7)
kernel_initializer_8 = tf.keras.initializers.constant(w8)
kernel_initializer_9 = tf.keras.initializers.constant(w9)
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))

## Initialization
################# Numerical parameters ################
multi_itr = 5               # Iterations of multi-grid 
j_itr = 1                   # Iterations of Jacobi 
ntime = 10                 # Time steps (fewer timesteps)
n_out = 500                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver
LSCALAR = False              # Scalar transport 
LMTI = False                 # Non density for multiphase flows
LIBM = True                # Immersed boundary method 
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime   
save_fig = True             # Save results
Restart = False             # Restart
eplsion_k = 1e-03
################# Physical parameters #################
rho_l = 1000                # Density of liquid phase 
rho_g = 980                 # Density of gas phase 
g_x = 0;g_y = 0;g_z = -9.8   # Gravity acceleration (m/s2) 
#######################################################
# print('============== Numerical parameters ===============')
# print('Mesh resolution:', values_v.shape)
# print('Time step:', ntime)
# print('Initial time:', ctime)
#######################################################
# Eliminated codes that are not relevant
################# Only for IBM ########################
if LIBM == True:
    # mesh = np.load('INHALE_1280.npy')
    mesh = np.load('master/mesh_64_sk.npy')
    sigma = np.zeros((1,nz,nx,ny,1)).astype('float32')
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                if mesh[0][i][j][k][0] == 0:
                    sigma[0][k][j][i][0] = 1000000000  
    sigma = tf.convert_to_tensor(sigma)
    # print('Switch on IBM solver!')
    # print('===================================================')
    # plt.imshow(sigma[0,1,:,:,0], cmap='jet')
    # plt.colorbar()
    # plt.title('South Kensington area')
    # plt.gca().invert_yaxis()
#######################################################
he = HaloExchange(structured=True, halo_size=1, tensor_used=True,double_precision=True, corner_exchanged=True)
sub_nz, sub_nx, sub_ny, current_domain = he.initialization(sigma,topology=(1,4,4), is_periodic=False, is_reordered=False)
sub_x, sub_y, sub_z = sub_nx+2, sub_ny+2, sub_nz+2

current_domain = he.structured_halo_update_3D(current_domain)
current_domain = current_domain.numpy()
rank = he.rank  # get process rank
neighbors = he.neighbors

LEFT = 0
RIGHT = 1
FRONT = 2
BEHIND = 3
TOP = 4
BOTTOM = 5
###################################################################################

CNN3D_central_2nd_dif = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])

CNN3D_central_2nd_xadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_central_2nd_yadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_3,
                                bias_initializer=bias_initializer),
])

CNN3D_central_2nd_zadv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_4,
                                bias_initializer=bias_initializer),
])

CNN3D_Su = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_6,
                                bias_initializer=bias_initializer),
])

CNN3D_Sv = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_7,
                                bias_initializer=bias_initializer),
])

CNN3D_Sw = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(sub_z, sub_y, sub_x, 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',
                                kernel_initializer=kernel_initializer_8,
                                bias_initializer=bias_initializer),
])

# CNNs
# Multi-grid process: (128,32,32) -> (64,16,16) -> (32,8,8) -> (16,4,4) -> (8,2,2) -> (4,1,1)
nlevel = int(math.log(32,2))
# laplacian operators
for i in range(nlevel+1):
    locals()['CNN3D_A_'+str(2**(nlevel -i) + 2)] = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(int(sub_nz*(0.5**i) + 2), int(sub_ny*(0.5**i) + 2), int(sub_nx*(0.5**i) + 2), 1)),
         tf.keras.layers.Conv3D(1, kernel_size=3, strides=1, padding='VALID',         
                                kernel_initializer=kernel_initializer_5,
                                bias_initializer=bias_initializer)
])
    
# restriction layers    
for i in range(nlevel):
    locals()['CNN3D_res_'+str(2**(nlevel-i))] = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(int(sub_nz*0.5**(i)), int(sub_ny*0.5**(i)), int(sub_nx*0.5**(i)), 1)),
         tf.keras.layers.Conv3D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_9,
                                bias_initializer=bias_initializer),   
])

# prolongation layers
for i in range(nlevel):
    locals()['CNN3D_prol_'+str(2**i)] = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2**(i+2), 2**(i), 2**(i), 1)),
         tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
])
    
# functions linking to AI functions    
def boundary_condition_velocity(values_u,values_v,values_w,nx,ny,nz):
    """Define boundary conditions for velocity field"""
    global neighbors, FRONT, BEHIND, LEFT, RIGHT, TOP, BOTTOM

    tempu = tf.Variable(values_u)
    tempv = tf.Variable(values_v)   
    tempw = tf.Variable(values_w)
    
    print("1:",values_u.shape)
    print("2:",tempu[0,1:-1,1:-1,1:3,0].shape)
    print(tf.zeros((1,nz,nx))[0,:].shape)
    
    # print((nz,nx,ny))
    if neighbors[LEFT] == -2:
        tempu[0,1:-1,1:-1,1,0].assign(tf.Variable(tf.ones((1,nz,nx)))[0,:]*ub) 
        tempv[0,1:-1,1:-1,1,0].assign(tf.Variable(tf.zeros((1,nz,nx)))[0,:])    
        tempw[0,1:-1,1:-1,1,0].assign(tf.Variable(tf.zeros((1,nz,nx)))[0,:])    

    if neighbors[RIGHT] == -2:
        tempu[0,1:-1,1:-1,ny,0].assign(tf.Variable(tf.ones((1,nz,nx)))[0,:]*ub) 
        tempv[0,1:-1,1:-1,ny,0].assign(tf.Variable(tf.zeros((1,nz,nx)))[0,:])   
        tempw[0,1:-1,1:-1,ny,0].assign(tf.Variable(tf.zeros((1,nz,nx)))[0,:])  
        
    if neighbors[FRONT] == -2:
        tempu[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(values_u)[0, 1:-1, 2, 1:-1, 0])
        tempv[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nz, ny)))[0, :])
        tempw[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nz, ny)))[0, :])
    
    if neighbors[BEHIND] == -2:
        tempu[0, 1:-1, nx, 1:-1, 0].assign(tf.Variable(values_u)[0, 1:-1, nx-1, 1:-1, 0])
        tempv[0, 1:-1, nx, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nz, ny)))[0, :])
        tempw[0, 1:-1, nx, 1:-1 ,0].assign(tf.Variable(tf.zeros((1, nz, ny)))[0, :])
        
    if neighbors[BOTTOM] == -2:
        tempu[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nx, ny)))[0, :])
        tempv[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nx, ny)))[0, :])
        tempw[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(tf.zeros((1, nx, ny)))[0, :])

    if neighbors[TOP] == -2:
        tempu[0, nz, 1:-1, 1:-1, 0].assign(tf.Variable(values_u)[0, nz-1, 1:-1, 1:-1, 0])
        tempv[0, nz, 1:-1, 1:-1, 0].assign(tf.Variable(values_v)[0, nz-1, 1:-1, 1:-1, 0])
        tempw[0, nz, 1:-1, 1:-1,0].assign(tf.Variable(tf.zeros((1, nx, ny)))[0, :])

    return tempu,tempv,tempw

def boundary_condition_pressure(values_p,nx,ny,nz):
    'Define boundary conditions for pressure field'
    global neighbors, FRONT, BEHIND, LEFT, RIGHT, TOP, BOTTOM

    tempp = tf.Variable(values_p)   
    
    if neighbors[RIGHT] == -2:
        tempp[0, 1:-1, 1:-1, nx,0].assign(tf.Variable(tf.zeros((1, nz, ny)))[0, :])
    if neighbors[LEFT] == -2:
        tempp[0, 1:-1, 1:-1, 1, 0].assign(tf.Variable(values_p)[0, 1:-1, 1:-1, 2, 0])
        
    if neighbors[FRONT] == -2:
        tempp[0, 1:-1, 1, 1:-1, 0].assign(tf.Variable(values_p)[0, 1:-1, 2, 1:-1, 0])
    if neighbors[BEHIND] == -2:
        tempp[0, 1:-1, nx, 1:-1, 0].assign(tf.Variable(values_p)[0, 1:-1, nx - 1, 1:-1, 0])
        
    if neighbors[BOTTOM] == -2:
        tempp[0, 1, 1:-1, 1:-1, 0].assign(tf.Variable(values_p)[0, 2, 1:-1, 1:-1, 0])
    if neighbors[TOP] == -2:
        tempp[0, nx, 1:-1, 1:-1, 0].assign(tf.Variable(values_p)[0, nx-1, 1:-1, 1:-1, 0])
    
    # tempp[0,:,:,nx-1,0].assign(tf.Variable(tf.zeros((1,nz,ny)))[0,:]) 
    # tempp[0,:,:,0,0].assign(tf.Variable(values_p)[0,:,:,1,0])
    
    # tempp[0,:,0,:,0].assign(tf.Variable(values_p)[0,:,1,:,0])     
    # tempp[0,:,nx-1,:,0].assign(tf.Variable(values_p)[0,:,nx-2,:,0]) 
    
    # tempp[0,0,:,:,0].assign(tf.Variable(values_p)[0,1,:,:,0])     
    # tempp[0,nz-1,:,:,0].assign(tf.Variable(values_p)[0,nz-2,:,:,0])  
    return tempp

def boundary_condition_indicator(alpha,nx):   
    'Define boundary conditions for scalar field'
    tempa = tf.Variable(alpha)   
    tempa[0,:,:,nx-1,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:]) 
    tempa[0,:,:,0,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:])
    
    tempa[0,:,0,:,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:])     
    tempa[0,:,nx-1,:,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:]) 
    
    
    tempa[0,0,:,:,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:])     
    tempa[0,nx-1,:,:,0].assign(tf.Variable(tf.zeros((1,nx,nx)))[0,:])  
    return tempa

def boundary_condition_source(b,nx):  
    'Define inflow boundary conditions for source terms to'
    'avoid incorrect paddings caused by CNNs'
    global neighbors,LEFT
    
    tempb = tf.Variable(b)
    if neighbors[LEFT] == -2:
        tempb[0, :, :, 0, 0].assign(tf.Variable(b)[0, :, :, 1, 0])
        
    return tempb

def bluff_body(values_u,values_v,values_w,sigma):
    'Immersed Boundary for buildings'
    temp1 = values_u / (1+dt*sigma) 
    temp2 = values_v / (1+dt*sigma)
    temp3 = values_w / (1+dt*sigma) 
    return temp1,temp2,temp3

def save_data(n_out,itime):
    'Save field data'
    if itime % n_out == 0:  
        np.save("data_SK_Petrov_Galerkin_demo/uu"+str(itime), arr=values_u[0,:,:,:,0])
        np.save("data_SK_Petrov_Galerkin_demo/vv"+str(itime), arr=values_v[0,:,:,:,0])
        np.save("data_SK_Petrov_Galerkin_demo/ww"+str(itime), arr=values_w[0,:,:,:,0])
        np.save("data_SK_Petrov_Galerkin_demo/pp"+str(itime), arr=values_p[0,:,:,:,0])
        
def Petrov_Galerkin_dissipation(values_u, values_v, values_w, eplsion_k, sigma):    
    '''Turbulence modelling using Petrov-Galerkin dissipation       
    Input
    ---------------------------------------------------------------------
    values_u: u-component velocity 
    values_v: v-component velocity 
    values_w: w-component velocity 
    eplsion_k: Need to sufficiently large
    
    Output
    ---------------------------------------------------------------------
    k_x: Final diffusion matrix in x direction 
    k_y: Final diffusion matrix in y direction 
    k_z: Final diffusion matrix in z direction 
    '''
    values_u = values_u / (1+dt*sigma) 
    values_v = values_v / (1+dt*sigma)     
    values_w = values_w / (1+dt*sigma) 
    
    # extract valid domains
    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
      
    k_u = 8 * 0.25 * abs(1/3*
                            (abs(tempU) + abs(tempV) + abs(tempW)) * 
                               CNN3D_central_2nd_dif(values_u)) / (eplsion_k + 
                               (abs(CNN3D_central_2nd_xadv(values_u)) + 
                               abs(CNN3D_central_2nd_yadv(values_u)) + 
                               abs(CNN3D_central_2nd_zadv(values_u)))/3)
    
    k_v = 8 * 0.25 * abs(1/3*
                            (abs(tempU) + abs(tempV) + abs(tempW)) * 
                               CNN3D_central_2nd_dif(values_v)) / (eplsion_k + 
                               (abs(CNN3D_central_2nd_xadv(values_v)) + 
                               abs(CNN3D_central_2nd_yadv(values_v)) + 
                               abs(CNN3D_central_2nd_zadv(values_v)))/3)
    
    k_w = 8 * 0.25 * abs(1/3*
                            (abs(tempU) + abs(tempV) + abs(tempW)) * 
                               CNN3D_central_2nd_dif(values_w)) / (eplsion_k + 
                               (abs(CNN3D_central_2nd_xadv(values_w)) + 
                               abs(CNN3D_central_2nd_yadv(values_w)) + 
                               abs(CNN3D_central_2nd_zadv(values_w)))/3)
              
    # padding and halo update                 
    k_u = he.padding_block_halo_3D(k_u, 1).reshape(1,sub_z,sub_x,sub_y,1)
    k_v = he.padding_block_halo_3D(k_v, 1).reshape(1,sub_z,sub_x,sub_y,1)
    k_w = he.padding_block_halo_3D(k_w, 1).reshape(1,sub_z,sub_x,sub_y,1)
    k_u = he.structured_halo_update_3D(k_u)
    k_v = he.structured_halo_update_3D(k_v)
    k_w = he.structured_halo_update_3D(k_w)
       
    k_u = tf.math.minimum(k_u, tf.ones(input_shape)*0.5) / (1+dt*sigma) 
    k_v = tf.math.minimum(k_v, tf.ones(input_shape)*0.5) / (1+dt*sigma)     
    k_w = tf.math.minimum(k_w, tf.ones(input_shape)*0.5) / (1+dt*sigma)    
    
    # extract valid domains
    tempKU = tf.reshape(k_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
    tempKV = tf.reshape(k_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
    tempKW = tf.reshape(k_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_ny,sub_nx,1))
    
    # print(tempKU.shape,values_u.shape,k_u.shape,tempU.shape)
    k_x = 3*0.5*(tempKU*CNN3D_central_2nd_dif(values_u) +
                CNN3D_central_2nd_dif(values_u*k_u) -
                tempU*CNN3D_central_2nd_dif(k_u))


    k_y = 3*0.5*(tempKV*CNN3D_central_2nd_dif(values_v) + 
                CNN3D_central_2nd_dif(values_v*k_v) -
                tempV*CNN3D_central_2nd_dif(k_v))


    k_z = 3*0.5*(tempKW*CNN3D_central_2nd_dif(values_w) + 
                CNN3D_central_2nd_dif(values_w*k_w) -
                tempW*CNN3D_central_2nd_dif(k_w))

    return k_x, k_y, k_z


#################### Create field #####£###############
input_shape = (1,sub_z,sub_x,sub_y,1)
values_u = tf.zeros(input_shape)
values_v = tf.zeros(input_shape)
values_w = tf.zeros(input_shape)
values_p = tf.zeros(input_shape)


### AI-HFM Solver
start = time.time()
for itime in range(1,ntime+1): 
    ctime = ctime + dt 
    [values_u,values_v,values_w] = boundary_condition_velocity(values_u,values_v,values_w,sub_nx,sub_ny,sub_nz)
    break
    values_p = boundary_condition_pressure(values_p,sub_nx,sub_ny,sub_nz)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    values_p = he.structured_halo_update_3D(values_p)
    
    # extract the valid domain
    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    
    # Petrov-Galerkin dissipation
    [k_x,k_y,k_z] = Petrov_Galerkin_dissipation(values_u, values_v, values_w, eplsion_k, current_domain)
    
    # Momentum equation 
    a_u = k_x - \
    tempU*CNN3D_central_2nd_xadv(values_u) - \
    tempV*CNN3D_central_2nd_yadv(values_u) - \
    tempW*CNN3D_central_2nd_zadv(values_u)
    
    b_u = 0.5*a_u + tempU
    a_v = k_y - \
    tempU*CNN3D_central_2nd_xadv(values_v) - \
    tempV*CNN3D_central_2nd_yadv(values_v) - \
    tempW*CNN3D_central_2nd_zadv(values_v)
    
    b_v = 0.5*a_v + tempV
    a_w = k_z - \
    tempU*CNN3D_central_2nd_xadv(values_w) - \
    tempV*CNN3D_central_2nd_yadv(values_w) - \
    tempW*CNN3D_central_2nd_zadv(values_w) 
    b_w = 0.5*a_w + tempW
    
    # padding and update halos
    b_u = he.padding_block_halo_3D(b_u, 1).reshape(1,sub_z,sub_x,sub_y,1)
    b_v = he.padding_block_halo_3D(b_v, 1).reshape(1,sub_z,sub_x,sub_y,1)
    b_w = he.padding_block_halo_3D(b_w, 1).reshape(1,sub_z,sub_x,sub_y,1)

    [b_u,b_v,b_w] = boundary_condition_velocity(b_u,b_v,b_w,sub_nx,sub_ny,sub_nz)
    [kb_x,kb_y,kb_z] = Petrov_Galerkin_dissipation(b_u, b_v, b_w, eplsion_k,current_domain)
    
    # extract valid domains    
    tempBU = tf.reshape(b_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempBV = tf.reshape(b_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempBW = tf.reshape(b_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))

    c_u = kb_x - \
    tempBU*CNN3D_central_2nd_xadv(b_u) - \
    tempBV*CNN3D_central_2nd_yadv(b_u) - \
    tempBW*CNN3D_central_2nd_zadv(b_u)
    tempU = tempU + c_u      
    
    c_v = kb_y - \
    tempBU*CNN3D_central_2nd_xadv(b_v) - \
    tempBV*CNN3D_central_2nd_yadv(b_v) - \
    tempBW*CNN3D_central_2nd_zadv(b_v) 
    tempV = tempV + c_v
    
    c_w = kb_z - \
    tempBU*CNN3D_central_2nd_xadv(b_w) - \
    tempBV*CNN3D_central_2nd_yadv(b_w) - \
    tempBW*CNN3D_central_2nd_zadv(b_w) 
    tempW = tempW + c_w    
    
    # padding and update
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)

    # Ommit other solvers for now.
    # Passive tracer scalar transport
    # if LSCALAR == True:
    #     a_t = CNN3D_central_2nd_dif(alpha) - \
    #     values_u*CNN3D_central_2nd_xadv(alpha) - \
    #     values_v*CNN3D_central_2nd_yadv(alpha) - \
    #     values_w*CNN3D_central_2nd_zadv(alpha)
    #     alpha = alpha + a_t
    #     alpha = boundary_condition_indicator(alpha,nx)
    # if LMTI == True:
    #     rho = alpha*rho_l + (1-alpha)*rho_g
    
    # IBM
    # if LIBM == True:
    [values_u,values_v,values_w] = bluff_body(values_u,values_v,values_w,current_domain)
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    
    # gradp            
    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))

    # padding and update    
    tempU = tempU - CNN3D_central_2nd_xadv(values_p)
    tempV = tempV - CNN3D_central_2nd_yadv(values_p)  
    tempW = tempW - CNN3D_central_2nd_zadv(values_p)
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
         
    [values_u,values_v,values_w] = boundary_condition_velocity(values_u,values_v,values_w,sub_nx,sub_ny,sub_nz)
    
    # halo update
    values_u = he.structured_halo_update_3D(values_u)
    values_v = he.structured_halo_update_3D(values_v)
    values_w = he.structured_halo_update_3D(values_w)
    
    # possion equation (multi-grid) A*P = Su
    b = -(CNN3D_Su(values_u) + CNN3D_Sv(values_v) + CNN3D_Sw(values_w))
    b = boundary_condition_source(b,sub_nx)
    
    # NOTE: the multi-grid process have been modified for computation field 128x128x128, modify it to the correct field size if needed
    if mgsolver == True:
        for multi_grid in range(multi_itr): 
            # compute the residual
            # w_1 = tf.zeros([1,1,ratio,ratio,1]) # coarest size (4,1,1) in this case
            w_1 = tf.zeros([1,4,1,1,1])
            r = CNN3D_A_34(values_p) - b
            if neighbors[RIGHT] == -2:
                r = tf.Variable(r)[0,:,:,sub_nx-1,0].assign(tf.Variable(tf.zeros((1,sub_nz,sub_ny)))[0,:]) 
            
            r_16 = CNN3D_res_32(r) # (128,32,32) -> (64,16,16) 
            r_8 = CNN3D_res_16(r_16) # (64,16,16) -> (32,8,8) 
            r_4 = CNN3D_res_8(r_8) # (32,8,8) -> (16,4,4) 
            r_2 = CNN3D_res_4(r_4) # (16,4,4) -> (8,2,2)
            r_1 = CNN3D_res_2(r_2) # (8,2,2) -> (4,1,1)
            
            w_t = he.padding_block_halo_3D(w_1, 1)
            w_t = he.structured_halo_update_3D(w_t)             
            for Jacobi in range(j_itr):
                temp1 = CNN3D_A_3(w_t)
                w_1 = (w_1 - temp1/w5[0,1,1,1,0] + r_1/w5[0,1,1,1,0])

            w_2 = CNN3D_prol_1(w_1)
            w_t1 = he.padding_block_halo_3D(w_2, 1)
            w_t1 = he.structured_halo_update_3D(w_t1)             
            for Jacobi in range(j_itr):
                temp2 = CNN3D_A_4(w_t1)
                w_2 = (w_2 - temp2/w5[0, 1, 1, 1, 0] + r_2/w5[0, 1, 1, 1, 0])

            w_4 = CNN3D_prol_2(w_2) 
            w_t2 = he.padding_block_halo_3D(w_4, 1)
            w_t2 = he.structured_halo_update_3D(w_t2)
            for Jacobi in range(j_itr):
                temp4 = CNN3D_A_6(w_t2)
                w_4 = (w_4 - temp4/w5[0, 1, 1, 1, 0] + r_4/w5[0, 1, 1, 1, 0])

            w_8 = CNN3D_prol_4(w_4) 
            w_t3 = he.padding_block_halo_3D(w_8, 1)
            w_t3 = he.structured_halo_update_3D(w_t3)
            for Jacobi in range(j_itr):
                temp8 = CNN3D_A_10(w_t3)
                w_8 = (w_8 - temp8/w5[0, 1, 1, 1, 0] + r_8/w5[0, 1, 1, 1, 0])

            w_16 = CNN3D_prol_8(w_8) 
            w_t4 = he.padding_block_halo_3D(w_16, 1)
            w_t4 = he.structured_halo_update_3D(w_t4)
            for Jacobi in range(j_itr):
                temp16 = CNN3D_A_18(w_t4)
                w_16 = (w_16 - temp16/w5[0, 1, 1, 1, 0] + r_16/w5[0, 1, 1, 1, 0])
                
            w_32 = CNN3D_prol_16(w_16)
            w_t5 = he.padding_block_halo_3D(w_32, 1)
            w_t5 = he.structured_halo_update_3D(w_t5)
            for Jacobi in range(j_itr):
                temp32 = CNN3D_A_34(w_t5)
                w_32 = (w_32 - temp32/w5[0, 1, 1, 1, 0] + r/w5[0, 1, 1, 1, 0])

            w_32 = he.padding_block_halo_3D(w_32,1)
            w_32 = he.structured_halo_update_3D(w_32)
            values_p = values_p - w_32
            if neighbors[RIGHT] == -2:
                values_p = tf.Variable(values_p)[0,1:-1,1:-1,sub_ny,0].assign(tf.Variable(tf.zeros((1,sub_nz,sub_nx)))[0,:])

            values_p = he.structured_halo_update_3D(values_p)       
            tempVal = tf.reshape(values_p[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
            tempVal = tempVal - CNN3D_A_34(values_p)/w5[0, 1, 1, 1, 0] + b/w5[0, 1, 1, 1, 0]
            values_p = he.padding_block_halo_3D(tempVal,1)
            values_p = he.structured_halo_update_3D(values_p)

            # values_p = (values_p - CNN3D_A_256(values_p)/w5[0,1,1,1,0] + b/w5[0,1,1,1,0])        
              
    # correction
    values_p = boundary_condition_pressure(values_p,sub_nx,sub_ny,sub_nz)
    values_p = he.structured_halo_update_3D(values_p)   # halo update
    
    tempU = tf.reshape(values_u[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempV = tf.reshape(values_v[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    tempW = tf.reshape(values_w[0,1:-1,1:-1,1:-1,0],(1,sub_nz,sub_nx,sub_ny,1))
    
    tempU = tempU - CNN3D_central_2nd_xadv(values_p)
    tempV = tempV - CNN3D_central_2nd_yadv(values_p)  
    tempW = tempW - CNN3D_central_2nd_zadv(values_p)
    
    values_u = he.padding_block_halo_3D(tempU, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_v = he.padding_block_halo_3D(tempV, 1).reshape(1,sub_z,sub_x,sub_y,1)
    values_w = he.padding_block_halo_3D(tempW, 1).reshape(1,sub_z,sub_x,sub_y,1)
    
    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update
    
    [values_u,values_v,values_w] = boundary_condition_velocity(values_u,values_v,values_w, sub_nx,sub_ny,sub_nz)
    
    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update
    
    # if LIBM == True:
    [values_u,values_v,values_w] = bluff_body(values_u,values_v,values_w,current_domain)
    
    values_u = he.structured_halo_update_3D(values_u) # halo update
    values_v = he.structured_halo_update_3D(values_v) # halo update
    values_w = he.structured_halo_update_3D(values_w) # halo update
    
    # output   
    print('Time step:', itime) 
    print('Pressure error:', np.max(w_32), 'cty equation residual:', np.max(r))
    print('========================================================')
    if np.max(np.abs(w_32)) > nsafe:
        print('Not converged !!!!!!')
        break
    if save_fig == True:
        save_data(n_out,itime+nrestart)
end = time.time()
print('[TOTAL RUNTIME]',(end-start))
