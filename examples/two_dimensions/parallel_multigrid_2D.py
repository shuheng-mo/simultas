
# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import math
import time

# import MPI subdomain
import halo_exchange

# Parameters to be defined for different grid size and conductivity
dt = 10      # Time step (s)
dx = 1       # Grid size in x    
dy = 1       # Grid size in y   
Dx = 0.1     # Conductivity in x   
Dy = 0.1     # Conductivity in y   
# Parameters for the computational domain
alpha = 1    # relaxation coefficient for Jacobi iteration (from 0 to 1)
nx = 128     # Grid point in x
ny = 128     # Grid point in x
ub = 1       # Velocity (1m/s)

# the weights matrix                   
w1 = np.zeros([1,2,2,1])             
w2 = np.zeros([1,3,3,1])            
w1[0,:,:,0] = 0.25                  
w2[0][0][1][0] = - Dy*dt/dy**2
w2[0][1][0][0] = - Dx*dt/dx**2 - ub*dt/(dx)                   
w2[0][1][1][0] = 1 + 2*(Dx*dt/dx**2+Dy*dt/dy**2) + 2*ub*dt/(dx) 
w2[0][1][2][0] =  - Dx*dt/dx**2
w2[0][2][1][0] =  - Dy*dt/dy**2 - ub*dt/(dx) 

kernel_initializer_1 = tf.keras.initializers.constant(w1)
kernel_initializer_2 = tf.keras.initializers.constant(w2)
bias_initializer = tf.keras.initializers.constant(np.zeros((1,)))

CNN3D_A_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, nx, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer)
])

CNN3D_A_66 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(66, 66, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_34 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(34, 34, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_18 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(18, 18, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_10 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(10, 10, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_6 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(6, 6, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_A_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',         # A matrix
                                kernel_initializer=kernel_initializer_2,
                                bias_initializer=bias_initializer),
])

CNN3D_res_128 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(nx, nx, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),   
])    
CNN3D_res_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),  
])
CNN3D_res_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer), 
])
CNN3D_res_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])
CNN3D_res_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])
CNN3D_res_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 1)),
         tf.keras.layers.Conv2D(1, kernel_size=2, strides=2, padding='VALID',  # restriction
                                kernel_initializer=kernel_initializer_1,
                                bias_initializer=bias_initializer),
])

CNN3D_prol_1 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(1, 1, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),
])

CNN3D_prol_2 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(2, 2, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),
])

CNN3D_prol_4 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(4, 4, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),
])

CNN3D_prol_8 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(8, 8, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),   
])

CNN3D_prol_16 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(16, 16, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)), 
])

CNN3D_prol_32 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(32, 32, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),   
])

CNN3D_prol_64 = keras.models.Sequential([
         keras.layers.InputLayer(input_shape=(64, 64, 1)),
         tf.keras.layers.UpSampling2D(size=(2, 2)),
])

# you might need incorporate with SFC code here
T = np.zeros([nx,ny])
for i in range(38):
    for j in range(38):
        T[i+44][j+44] = 1 

T = T[2:-2,2:-2]

# domain decomposition
sub_nx, sub_ny, current_domain = halo_exchange.domain_decomposition_2D(T,nx-4,ny-4)

rank = halo_exchange.rank

current_domain = halo_exchange.structured_halo_update_2D(current_domain) # update onetime before the multigrid
current_domain = current_domain.numpy().reshape(64,64)

input_shape = (1,sub_nx+2,sub_ny+2,1)
values = tf.zeros(input_shape)
values = tf.Variable(values)[0,:,:,0].assign(tf.Variable(values)[0,:,:,0]+tf.convert_to_tensor(current_domain.astype('float32')))

start = time.time()
b = values           # only for one time step
multi_itr = 1     # multigrid iteration
j_itr = 1            # jacobi iteration 
for multi_grid in range(multi_itr):    
    w = np.zeros([1,1,1,1])
# --------- Calculate Residual based on initial guess --------  
    # r = CNN3D_A_128(values) - b 
    r = CNN3D_A_64(values) - b 
# ------------------------------------------------------------  
    r = halo_exchange.structured_halo_update_2D(r) # update for the residual

    np.save("r_{}".format(rank), r)
# --------- Interpolate Residual from finer to coaser mesh --------  
    # r_64 = CNN3D_res_128(r) 
    r_32 = CNN3D_res_64(r) 
    r_16 = CNN3D_res_32(r_32) 
    r_8 = CNN3D_res_16(r_16) 
    r_4 = CNN3D_res_8(r_8) 
    r_2 = CNN3D_res_4(r_4) 
    r1 = CNN3D_res_2(r_2)    
# -----------------------------------------------------------------      

# --------- Interpolate Residual from coaser to finer mesh --------  
    for Jacobi in range(j_itr):
        w = w - CNN3D_A_1(w)/w2[0][1][1][0] + r1/w2[0][1][1][0]
    w = w - CNN3D_A_1(w)/w2[0][1][0] + r1/w2[0][1][0]
    
    w_2 = CNN3D_prol_1(w)                   
    w_t1 = halo_exchange.padding_block_halo_2D(w_2,1)
    w_t1 = halo_exchange.structured_halo_update_2D(w_t1)     
    for Jacobi in range(j_itr):
        temp = CNN3D_A_4(w_t1)
        w_2 = w_2 - temp/w2[0][1][1][0] + r_2/w2[0][1][1][0]

    w_4 = CNN3D_prol_2(w_2) 
    w_t2 = halo_exchange.padding_block_halo_2D(w_4,1)
    w_t2 = halo_exchange.structured_halo_update_2D(w_t2) 
    for Jacobi in range(j_itr):
        temp = CNN3D_A_6(w_t2)
        w_4 = w_4 - temp/w2[0][1][1][0] + r_4/w2[0][1][1][0]

    w_8 = CNN3D_prol_4(w_4)
    w_t3 = halo_exchange.padding_block_halo_2D(w_8,1)
    w_t3 = halo_exchange.structured_halo_update_2D(w_t3)  
    for Jacobi in range(j_itr):
        temp = CNN3D_A_10(w_t3)
        w_8 = w_8 - temp/w2[0][1][1][0] + r_8/w2[0][1][1][0]

    w_16 = CNN3D_prol_8(w_8)
    w_t4 = halo_exchange.padding_block_halo_2D(w_16,1)
    w_t4 = halo_exchange.structured_halo_update_2D(w_t4)  
    for Jacobi in range(j_itr):
        temp = CNN3D_A_18(w_t4)
        w_16 = w_16 - temp/w2[0][1][1][0] + r_16/w2[0][1][1][0]

    w_32 = CNN3D_prol_16(w_16) 
    w_t5 = halo_exchange.padding_block_halo_2D(w_32,1)
    w_t5 = halo_exchange.structured_halo_update_2D(w_t5) 
    for Jacobi in range(j_itr):
        temp = CNN3D_A_34(w_t5)
        w_32 = w_32 - temp/w2[0][1][1][0] + r_32/w2[0][1][1][0]

    w_64 = CNN3D_prol_32(w_32)
    w_t6 = halo_exchange.padding_block_halo_2D(w_64,1)
    w_t6 = halo_exchange.structured_halo_update_2D(w_t6)
    for Jacobi in range(j_itr):
        temp = CNN3D_A_66(w_t6)
        w_64 = w_64 - temp/w2[0][1][1][0] + r/w2[0][1][1][0]

    # w_128 = CNN3D_prol_64(w_64)
    # w_128 = w_128 - CNN3D_A_128(w_128)/w2[0][1][1][0] + r/w2[0][1][1][0]
# ----------------------------------------------------------------- 

# --------- Correct initial guess --------  
    # values = values - w_128 
    # values = values - CNN3D_A_128(values)/w2[0][1][1][0] + b/w2[0][1][1][0]

    values = values - w_64
    values = values - CNN3D_A_64(values)/w2[0][1][1][0] + b/w2[0][1][1][0]
    values = halo_exchange.structured_halo_update_2D(values)
# ----------------------------------------  
np.save("/content/parallel_output/parallel_res_{}".format(rank),values)
np.save("/content/parallel_residuals/w_{}".format(rank),w)
np.save("/content/parallel_residuals/w2_{}".format(rank),w_2)
np.save("/content/parallel_residuals/w4_{}".format(rank),w_4)
np.save("/content/parallel_residuals/w8_{}".format(rank),w_8)
np.save("/content/parallel_residuals/w16_{}".format(rank),w_16)
np.save("/content/parallel_residuals/w32_{}".format(rank),w_32)
np.save("/content/parallel_residuals/w64_{}".format(rank),w_64)
end = time.time()
print('Computational time(s):',(end-start))
print('Multigrid iterations:', multi_itr)
print('Jacobi iterations:', j_itr)
