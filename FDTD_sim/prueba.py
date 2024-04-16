import numba
import os
from time import time

from utils.cpu.calculator import calculator_cpu
from utils.cpu.executor import executor as executor_cpu
from utils.cpu.manager import manager_cpu

from utils.cuda.calculator import calculator_cuda
from utils.cuda.executor import executor as executor_cuda
from utils.cuda.manager import manager_cuda

import numpy as np

from numba import cuda, njit, prange

import os, cmath
clear = lambda: os.system('cls')
clear()

tipo_float = np.float64

gpu = cuda.get_current_device()
print("name = %s" % gpu.name)
print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
print("warpSize = %s" % str(gpu.WARP_SIZE))
print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))

def createRandomSortedList(num, start = 1, end = 100):
    arr = np.random.choice(np.arange(start, end + 1), size=num, replace=False)
    arr.sort()
    return arr

def define_cube (nPoints):
    pps = int(np.ceil(np.sqrt(nPoints/6)))+1
    
    x_, y_, z_ = np.meshgrid(np.linspace(0., 1., pps), np.linspace(0., 1., pps), np.linspace(0., 1., pps), indexing='ij')
    
    pos = []
    
    norm = []
    
    for i in range(pps):
        for j in range(pps):
            for k in range(pps):
                
                if (((x_[i,j,k] == 0 or x_[i,j,k] == 1) and y_[i,j,k] != 0 and y_[i,j,k] != 1 and z_[i,j,k] != 0 and z_[i,j,k] != 1) or 
                    ((y_[i,j,k] == 0 or y_[i,j,k] == 1) and x_[i,j,k] != 0 and x_[i,j,k] != 1 and z_[i,j,k] != 0 and z_[i,j,k] != 1) or 
                    ((z_[i,j,k] == 0 or z_[i,j,k] == 1) and y_[i,j,k] != 0 and y_[i,j,k] != 1 and x_[i,j,k] != 0 and x_[i,j,k] != 1) ):
                    
                    if x_[i,j,k] == 0:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([-1., 0., 0.])
                    elif x_[i,j,k] == 1:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([1., 0., 0.])
                    elif y_[i,j,k] == 0:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([0., -1., 0.])
                    elif y_[i,j,k] == 1:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([0., 1., 0.])
                    elif z_[i,j,k] == 0:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([0., 0., -1.])
                    elif z_[i,j,k] == 1:
                        pos.append([x_[i,j,k], y_[i,j,k], z_[i,j,k]])
                            
                        norm.append([0., 0., 1.])
                        
                        
    return (np.array(pos).astype(tipo_float), np.array(norm).astype(tipo_float))

def transducers_array (nTrans, axis, centre_axis, orientation, from_point, to_point):
    
    pps = int(np.ceil(np.sqrt(nTrans)))

    a_, b_ = np.meshgrid(np.linspace(from_point, to_point, pps), np.linspace(from_point, to_point, pps), indexing='ij')
    
    #print('a_', a_, 'b_', b_)

    pos = []
    
    norm = []

    if axis == 0:
        for i in range(pps):
            for j in range(pps):
                pos.append([centre_axis, a_[i,j], b_[i,j]])
                            
                norm.append([orientation, 0., 0.])
    elif axis == 1:
        for i in range(pps):
            for j in range(pps):
                pos.append([a_[i,j], centre_axis, b_[i,j]])
                            
                norm.append([0., orientation, 0.])
    elif axis == 2:
        for i in range(pps):
            for j in range(pps):
                pos.append([a_[i,j], b_[i,j], centre_axis])
                            
                norm.append([0., 0., orientation])
                               
    #print('x', x, 'y', y, 'z', z)
    #print('x_n', x_n, 'y_n', y_n, 'z_n', z_n)
                
    return (np.array(pos).astype(tipo_float), np.array(norm).astype(tipo_float))

#transducers_pos = np.random.rand(N,3).astype(np.float32) #np.float64(np.arange(N*3).reshape(N,3))
#transducers_norm = np.ones((N,3), dtype=np.float32) #np.float64(np.arange(N*3).reshape(N,3))

num_emitter = 4
N=int(32*32)
print(N)

transducers_pos, transducers_norm = transducers_array(nTrans = N, axis=0, centre_axis = 5., orientation = -1., from_point = 0., to_point = 1.)#-1., to_point = 3.)
print(transducers_pos)
transducers_radius = tipo_float(0.5/32)
select_emitters = np.array(createRandomSortedList(num_emitter, start=0, end=transducers_pos.shape[0]-1), dtype=np.int64)#.squeeze()

number_elements = int(2**14) # Up to 2**14 approx float64, up to 2**15 approx float32
print(number_elements)
k = tipo_float(20)
order = 4
#mesh_pos = np.random.rand(number_elements,3).astype(np.float32)
#mesh_norm = np.random.rand(number_elements,3).astype(np.float32)
mesh_pos, mesh_norm = define_cube(number_elements)
mesh_area_div_4pi = np.array([1.0/(4*np.pi*np.ceil(np.sqrt(number_elements/8))) for i in range(mesh_pos.shape[0])]).astype(tipo_float).reshape(mesh_pos.shape[0],1)#np.abs(np.random.rand(number_elements,1)).astype(np.float32)
print(mesh_pos)
mesh_pression = np.random.rand(number_elements, num_emitter).astype(tipo_float) + 1j*np.random.rand(number_elements, num_emitter).astype(tipo_float)

#print('\n\n\n CPU')
#
#ejecutor = executor_cpu(0.5)
#print('Done ejecutor')
#
#ejecutor.init_workspace(transducers_pos, transducers_norm, transducers_radius, select_emitters, number_elements, k, order = order, transducers_modifies_patterns = False, mesh_pos = mesh_pos, mesh_norm = mesh_norm, mesh_area_div_4pi = mesh_area_div_4pi)
#print('Done init_workspace')
#			
#start = time()
#ejecutor.calculate_A_matrix()
#print(time()-start)
##print(time())
#print('\n\n\n',ejecutor.A_matrix)
##temp = ejecutor.A_matrix
#
#start = time()
#ejecutor.calculate_transmission_matrix(mesh_pression)
#print(f'Time: {time()-start}')
#print('\n\n\n',ejecutor.transmission_matrix)
##temp2 = ejecutor.transmission_matrix


print('\n\n\n CUDA')

Pref = tipo_float(0.3)
ejecutor = executor_cuda(Pref, var_type='float64', out_var_type='complex128')
print('Done ejecutor')
number_elements=mesh_pos.shape[0]
print(number_elements)
ejecutor.init_workspace(transducers_pos, transducers_norm, transducers_radius, select_emitters, number_elements, k, order = order, transducers_modifies_patterns = False, mesh_pos = mesh_pos, mesh_norm = mesh_norm, mesh_area_div_4pi = mesh_area_div_4pi)
print('Done init_workspace')

start = time()
ejecutor.calculate_A_matrix()
print(time()-start)
print('\n\n\n',ejecutor.A_matrix)
print('\n\n\n',ejecutor.A_matrix.copy_to_host())
#print(np.sum(np.sum(temp-ejecutor.A_matrix.copy_to_host(), axis=1), axis=0))
#mesh_pression = cuda.to_device(mesh_pression, stream = ejecutor.stream)

start = time()
ejecutor.calculate_mesh_pressions()
print(time()-start)
print('\n\n\n mesh pressions',ejecutor.mesh_pressions)
print('\n\n\n',ejecutor.mesh_pressions.copy_to_host())

start = time()
ejecutor.calculate_transmission_matrix(ejecutor.mesh_pressions)
print(f'Time: {time()-start}')
print('\n\n\n',ejecutor.transmission_matrix)
print('\n\n\n',ejecutor.transmission_matrix.copy_to_host())
matriz = ejecutor.transmission_matrix.copy_to_host()
print(matriz.shape)
suma_r = []
suma_p = []
for i in range(matriz.shape[1]):
    suma_r.append(0)
    suma_p.append(0)
    for j in range(matriz.shape[0]):
        suma_r[i] += matriz[j][i]
        suma_p[i] += matriz[j][i]
    print(cmath.polar(suma_r[i]))
    suma_r[i] = cmath.polar(suma_r[i])[0]
    suma_p[i] = cmath.polar(suma_p[i])[1]
    
#print(np.sum(np.sum(temp2-ejecutor.transmission_matrix.copy_to_host(), axis=1), axis=0))
print(suma_r)

def color_transducers(tPos, eM, values, alt, nTrans, from_point, to_point):
    
    pps = pps = int(np.ceil(np.sqrt(nTrans)))

    a_, b_ = np.meshgrid(np.linspace(from_point, to_point, pps), np.linspace(from_point, to_point, pps), indexing='ij')
    print(a_)
    print(b_)
    color = np.zeros(shape=(pps, pps))
    
    checked = 0
    
    for i in range(tPos.shape[0]):
        idx_x = np.where(a_ == tPos[i, 1])[0][0]
        idx_y = np.where(b_ == tPos[i, 2])[1][0]
        print(idx_x, idx_y)
        if i in eM:
            print(checked)
            #for i_x in idx_x:
            #    for i_y in idx_y:
            color[idx_x, idx_y] = alt
            checked += 1
        else:
            #for i_x in idx_x:
            #    for i_y in idx_y:
            color[idx_x, idx_y] = values[i-checked]
            
    
    return a_, b_, color

import matplotlib.pyplot as plt

pPlot_x, pPlot_y, Z_r = color_transducers(transducers_pos, select_emitters, suma_r, -1, nTrans = N, from_point = 0., to_point = 1.)

#plt.imshow(np.array(Z_r).reshape(32,32))
#plt.show()
print(pPlot_x)

fig, ax0 = plt.subplots()

im = ax0.pcolormesh(pPlot_x, pPlot_y, np.array(Z_r), shading='gouraud')
fig.colorbar(im, ax=ax0)
plt.show()

try:
    while True:
        pass
except KeyboardInterrupt:
    pass

#_, Z_p = color_transducers(transducers_pos, select_emitters, suma_p, 4)
#
#plt.imshow(np.array(Z_p).reshape(32,32))
#plt.show()

pPlot_x, pPlot_y, Z_p = color_transducers(transducers_pos, select_emitters, suma_p, 4, nTrans = N, from_point = 0., to_point = 1.)

#plt.imshow(np.array(Z_r).reshape(32,32))
#plt.show()

fig, ax0 = plt.subplots()

im = ax0.pcolormesh(pPlot_x, pPlot_y, np.array(Z_p), shading='gouraud')
fig.colorbar(im, ax=ax0)
plt.show()
