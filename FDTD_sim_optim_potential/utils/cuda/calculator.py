import numpy as np, cmath #, numba, math
from numpy import complex64 as npc64, complex128 as npc128 
from numba import cuda
from numba.types import uint32 as u32, complex128 as c128, complex64 as c64



MAXTHREADNUMBER = 16

def calculate_bd_b2 (number, base=2, max_value = 1024):
    return min(int(base**(np.ceil(np.log(number)/np.log(base))-1)), max_value)

def optimize_blockdim (processorCount, size_0, size_1=0, size_2=0):
	threshold_1 = 2.5
	threshold_2 = 2.5
	bd_0 = calculate_bd_b2(size_0, 2)
	param = np.ceil(np.log(2*processorCount)/np.log(2))

	if size_1==0 and size_2==0:
		
		#print(size_0)
		
		exp_bd = np.log(bd_0/32)/np.log(2)
		
		#exp_32 = exp_bd/5
		exp_PC = exp_bd / param
		
		if exp_PC <= threshold_1:
			bd = (max(1, calculate_bd_b2(size_0/(2**param), max_value = 32)),)
		elif exp_PC > threshold_1:
			bd = (max(1, calculate_bd_b2(size_0/(2**param))),)
			
		bd = (int(bd[0]),)
			
	elif size_2 == 0:
		
		bd_1 = calculate_bd_b2(size_1)
		
		#print(size_0, size_1)
		
		if bd_1<10:
			
			exp_bd = np.log(bd_0/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			exp_PC = exp_bd / param
		
			if exp_PC <= threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), max_value = 32)), max(1, size_1))
			elif exp_PC > threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param))), max(1, size_1))
			
		else:
						
			exp_bd_0 = np.log(bd_0/32)/np.log(2)
			exp_bd_1 = np.log(bd_1/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			if exp_bd_0 > exp_bd_1:
				exp_0 = exp_bd_0 / param - 1
				exp_1 = exp_bd_1 / param - 2
				div_0 = 2**(np.ceil(param/2))
				div_1 = 2**(param - np.ceil(param/2))
			elif exp_bd_0 < exp_bd_1:
				exp_0 = exp_bd_0 / param - 2
				exp_1 = exp_bd_1 / param - 1
				div_0 = 2**(param - np.ceil(param/2))
				div_1 = 2**(np.ceil(param/2))
			else:
				exp_0 = exp_bd_0 / param - 1
				exp_1 = exp_bd_1 / param - 1
				div_0 = 2**(np.ceil(param/2))
				div_1 = 2**(np.ceil(param/2))
		
			if exp_0 <= threshold_1 and exp_1 <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, max_value = div_0)), (max(1, calculate_bd_b2(size_1/div_1, max_value = div_1))))
			elif exp_0 > threshold_1 and exp_1 <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0)),  (max(1, calculate_bd_b2(size_1/div_1, max_value = div_1))))
			elif exp_0 <= threshold_1 and exp_1 > threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, max_value = div_0)), (max(1, calculate_bd_b2(size_1/div_1))))
			else:
				bd = (max(1, calculate_bd_b2(size_0/div_0)),  (max(1, calculate_bd_b2(size_1/div_1))))
			
		bd = (int(bd[0]), int(bd[1]))

	else:
		
		bd_1 = calculate_bd_b2(size_1)
		bd_2 = calculate_bd_b2(size_2)
		
		#print(size_0, size_1)
		
		if max(bd_1, bd_2)<10:
			
			exp_bd = np.log(bd_0/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			exp_PC = exp_bd / param
		
			if exp_PC <= threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), max_value = 32)), max(1, size_1), max(1, size_2))
			elif exp_PC > threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param))), max(1, size_1), max(1, size_2))
			
		else:
			
			exp_bd_0 = np.log(bd_0/32)/np.log(2)
			exp_bd_1 = np.log(bd_1/32)/np.log(2)
			exp_bd_2 = np.log(bd_2/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			if exp_bd_0 < max(exp_bd_1, exp_bd_2):
				exp_0 = exp_bd_0 / param - 2
				exp_1 = exp_bd_1 / param - 1
				exp_2 = exp_bd_2 / param - 1
				div_0 = 2**(np.ceil(param/4))
				div_1 = 2**(np.ceil(param/3))
				div_2 = 2**(np.ceil(param/3))
			else:
				exp_0 = exp_bd_0 / param - 1
				exp_1 = exp_bd_1 / param - 1
				exp_2 = exp_bd_2 / param - 1
				div_0 = 2**(np.ceil(param/3))
				div_1 = 2**(np.ceil(param/4))
				div_2 = 2**(np.ceil(param/4))
		
			if exp_0 <= threshold_1 and max(exp_1, exp_2) <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, max_value = div_0)),
						(max(1, calculate_bd_b2(size_1/div_1, max_value = div_1))),
						(max(1, calculate_bd_b2(size_2/div_2, max_value = min(div_2, 64)))))
			elif exp_0 > threshold_1 and max(exp_1, exp_2) <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2)), 
						(max(1, calculate_bd_b2(size_1/div_1, max_value = div_1))),
						(max(1, calculate_bd_b2(size_2/div_2, max_value = min(div_2, 64)))))
			elif exp_0 <= threshold_1 and max(exp_1, exp_2) > threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, max_value = div_0)),
						(max(1, calculate_bd_b2(size_1/div_1))),
						(max(1, calculate_bd_b2(size_2/div_2, max_value=64))))
			else:
				bd = (max(1, calculate_bd_b2(size_0/div_0)), 
						(max(1, calculate_bd_b2(size_1/div_1))), 
						(max(1, calculate_bd_b2(size_2/div_2, max_value=64))))
		
		bd = (int(bd[0]), int(bd[1]), int(bd[2]))
	#print(bd)
	return bd


		


'''
Implement functions used to calculate the thigs
'''
	
'''
Function that calculates the next step in velocity
Execute for each axis
Standard discrete spatial derivatives defined as 0.5*([x+1] - [x-1]), with boundaries [1]-[0] and [end]-[end-1]
'''

def step_all_c64_forward_noreturn_cuda(potential_new, potential_old, field, dt, ds, csq, normal, emitters_amplitude, emitters_frequency, emitters_phase, time, size):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	#if i == 0 and j==0 and k==0:
	#if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
		#auxiliar = cuda.shared.array((MAXTHREADNUMBER,MAXTHREADNUMBER,MAXTHREADNUMBER), c64)
		#shared_position[0] = 0
	auxiliar = cuda.local.array(3, c64)

	#################
	## Change pressure to field
	#################
		
	#cuda.syncthreads()
		
	if i< u32(size) and j< u32(size) and k< u32(size):
		
		#auxiliar[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z] = pressure_new[u32(i + j*size + k*size**2)]
		auxiliar[0] = potential_new[u32(i + j*size + k*size**2)]
		
	#cuda.syncthreads()
	
	if i< u32(size) and j< u32(size) and k< u32(size):
				
		if field[u32(i + j*size + k*size**2)] == 1:
				
			potential_new[u32(i + j*size + k*size**2)] = ( ( 2 + dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * potential_new[u32(i + j*size + k*size**2)] - potential_old[u32(i + j*size + k*size**2)] + ( csq * dt**2 / ds **2) * (
					potential_new[u32(min(i + 1, int(size-1)) + j*size + k*size**2)] + potential_new[u32(i + min(j+1, int(size-1))*size + k*size**2)] + potential_new[u32(i + j*size + min(k+1, int(size-1))*size**2)]
					- 6 * potential_new[u32(i + j*size + k*size**2)]
					+ potential_new[u32(max(i - 1, 0) + j*size + k*size**2)] + potential_new[u32(i + max(j-1, 0)*size + k*size**2)] + potential_new[u32(i + j*size + max(k-1, 0)*size**2)]
				) ) / ( dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] + 1 )

		elif emitters_amplitude[u32(i + j*size + k*size**2)] != 0:
			
			potential_new[u32(i + j*size + k*size**2)] = ( ( 2 + dt *  ( 1 + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) ) * potential_new[u32(i + j*size + k*size**2)] - potential_old[u32(i + j*size + k*size**2)] + ( csq * dt**2 / ds ) * (
						cmath.rect( normal[u32(min(i + 1, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 1, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 1, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 1, size-1) + j*size + k*size**2)])
						+ cmath.rect( normal[u32(i + min(j + 1, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 1, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 1, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 1, size-1)*size + k*size**2)])
						+ cmath.rect( normal[u32(i + j*size + min(k + 1, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 1, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 1, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 1, size-1)*size**2)])
						- cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
						- cmath.rect( normal[u32(i + j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
						- cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
					) ) / ( dt * ( 1 + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) + 1 )
			
	#cuda.syncthreads()
	
	if i< u32(size) and j< u32(size) and k< u32(size):
	
		potential_old[u32(i + j*size + k*size**2)] = auxiliar[0]
		
	#cuda.syncthreads()
	
def step_all_c64_backward_noreturn_cuda(potential_new, potential_old, field, dt, ds, csq, normal, emitters_amplitude, emitters_frequency, emitters_phase, time, size):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	#if i == 0 and j==0 and k==0:
	#if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
		#auxiliar = cuda.shared.array((MAXTHREADNUMBER,MAXTHREADNUMBER,MAXTHREADNUMBER), c64)
		#shared_position[0] = 0
	auxiliar = cuda.local.array(3, c64)


	#################
	## Change pressure to field
	#################
		
	#cuda.syncthreads()
		
	if i< u32(size) and j< u32(size) and k< u32(size):
		
		#auxiliar[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z] = pressure_new[u32(i + j*size + k*size**2)]
		auxiliar[0] = potential_new[u32(i + j*size + k*size**2)]
		
	#cuda.syncthreads()
	
	if i< u32(size) and j< u32(size) and k< u32(size):
				
		if field[u32(i + j*size + k*size**2)] == 1:
				
			potential_new[u32(i + j*size + k*size**2)] = ( ( 2 + dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * potential_new[u32(i + j*size + k*size**2)] - potential_old[u32(i + j*size + k*size**2)] + ( csq * dt**2 / ds **2) * (
					potential_new[u32(min(i + 1, int(size-1)) + j*size + k*size**2)] + potential_new[u32(i + min(j+1, int(size-1))*size + k*size**2)] + potential_new[u32(i + j*size + min(k+1, int(size-1))*size**2)]
					- 6 * potential_new[u32(i + j*size + k*size**2)]
					+ potential_new[u32(max(i - 1, 0) + j*size + k*size**2)] + potential_new[u32(i + max(j-1, 0)*size + k*size**2)] + potential_new[u32(i + j*size + max(k-1, 0)*size**2)]
				) ) / ( dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] + 1 )

		elif emitters_amplitude[u32(i + j*size + k*size**2)] != 0:
			
			potential_new[u32(i + j*size + k*size**2)] = ( ( 2 + dt *  ( 1 + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) ) * potential_new[u32(i + j*size + k*size**2)] - potential_old[u32(i + j*size + k*size**2)] + ( csq * dt**2 / ds ) * (
						cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
						+ cmath.rect( normal[u32(i +j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
						+ cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
						- cmath.rect( normal[u32(max(i - 1, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 1, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 1, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 1, 0) + j*size + k*size**2)])
						- cmath.rect( normal[u32(i + max(j-1, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-1, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-1, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-1, 0)*size + k*size**2)])
						- cmath.rect( normal[u32(i + j*size + max(k-1, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-1, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-1, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-1, 0)*size**2)])
					) ) / ( dt * ( 1 + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) + 1 )
			
	#cuda.syncthreads()
	
	if i< u32(size) and j< u32(size) and k< u32(size):
	
		potential_old[u32(i + j*size + k*size**2)] = auxiliar[0]
		
	#cuda.syncthreads()
			
		#if pressure_new[u32(i + j*size + k*size**2)].real >= pressure_old[u32(i + j*size + k*size**2)].real:
		#	auxiliar_real = cuda.atomic.max(pressure_old.real, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].real)
		#
		#else:
		#	auxiliar_real = cuda.atomic.min(pressure_old.real, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].real)
		#	
		#if pressure_new[u32(i + j*size + k*size**2)].imag >= pressure_old[u32(i + j*size + k*size**2)].imag:
		#	auxiliar_imag = cuda.atomic.max(pressure_old.imag, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].imag)
		#
		#else:
		#	auxiliar_imag = cuda.atomic.min(pressure_old.imag, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].imag)
		
		
			
			#if ( field[u32(min(i + 1, size-1) + j*size + k*size**2 + field.shape[0] / 2)] - field[u32(min(i + 1, size-1) + j*size + k*size**2)] != field[u32(max(i - 1, 0) + j*size + k*size**2 + field.shape[0] / 2)] - field[u32(max(i - 1, 0) + j*size + k*size**2)]
			#	or field[u32(i + min(j + 1, size-1)*size + k*size**2 + field.shape[0] / 2)] - field[u32(i + min(j + 1, size-1)*size + k*size**2)] != field[u32(i + max(j - 1, 0)*size + k*size**2 + field.shape[0] / 2)] - field[u32(i + max(j - 1, 0)*size + k*size**2)]
			#	or field[u32(i + j*size + min(k + 1, size-1)*size**2 + field.shape[0] / 2)] - field[u32(i + j*size + min(k + 1, size-1)*size**2)] != field[u32(i + j*size + max(k - 1, 0)*size**2 + field.shape[0] / 2)] - field[u32(i + j*size + max(k - 1, 0)*size**2)] ):
			#																																 
			#	potential_new[u32(i + j*size + k*size**2)] = 2 * potential_new[u32(i + j*size + k*size**2)] - potential_old[u32(i + j*size + k*size**2)] + ( csq * dt**2 / ds ) * (
			#		(
			#			8 * (cmath.rect( normal[u32(min(i + 1, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 1, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 1, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 1, size-1) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + min(j + 1, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 1, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 1, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 1, size-1)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + min(k + 1, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 1, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 1, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 1, size-1)*size**2)])
			#			) + cmath.rect( normal[u32(max(i - 2, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 2, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 2, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 2, 0) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + max(j-2, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-2, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-2, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-2, 0)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + max(k-2, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-2, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-2, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-2, 0)*size**2)])
			#			-  8 *(cmath.rect( normal[u32(max(i - 1, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 1, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 1, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 1, 0) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + max(j-1, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-1, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-1, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-1, 0)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + max(k-1, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-1, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-1, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-1, 0)*size**2)])
			#			) - cmath.rect( normal[u32(min(i + 2, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 2, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 2, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 2, size-1) + j*size + k*size**2)])
			#			- cmath.rect( normal[u32(i + min(j + 2, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 2, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 2, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 2, size-1)*size + k*size**2)])
			#			- cmath.rect( normal[u32(i + j*size + min(k + 2, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 2, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 2, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 2, size-1)*size**2)])
			#		)  /  ( 12 )	
			#		#cmath.rect( normal[u32(min(i + 1, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 1, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 1, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 1, size-1) + j*size + k*size**2)])
			#		#+ cmath.rect( normal[u32(i + min(j + 1, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 1, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 1, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 1, size-1)*size + k*size**2)])
			#		#+ cmath.rect( normal[u32(i + j*size + min(k + 1, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 1, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 1, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 1, size-1)*size**2)])
			#		#- cmath.rect( normal[u32(max(i - 1, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 1, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 1, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 1, 0) + j*size + k*size**2)])
			#		#- cmath.rect( normal[u32(i + max(j-1, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-1, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-1, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-1, 0)*size + k*size**2)])
			#		#- cmath.rect( normal[u32(i + j*size + max(k-1, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-1, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-1, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-1, 0)*size**2)])																						
			#		)
			#
			#else:
				
				
			
			#if ( field[u32(min(i + 1, size-1) + j*size + k*size**2 + field.shape[0] / 2)] - field[u32(min(i + 1, size-1) + j*size + k*size**2)] != field[u32(max(i - 1, 0) + j*size + k*size**2 + field.shape[0] / 2)] - field[u32(max(i - 1, 0) + j*size + k*size**2)]
			#	or field[u32(i + min(j + 1, size-1)*size + k*size**2 + field.shape[0] / 2)] - field[u32(i + min(j + 1, size-1)*size + k*size**2)] != field[u32(i + max(j - 1, 0)*size + k*size**2 + field.shape[0] / 2)] - field[u32(i + max(j - 1, 0)*size + k*size**2)]
			#	or field[u32(i + j*size + min(k + 1, size-1)*size**2 + field.shape[0] / 2)] - field[u32(i + j*size + min(k + 1, size-1)*size**2)] != field[u32(i + j*size + max(k - 1, 0)*size**2 + field.shape[0] / 2)] - field[u32(i + j*size + max(k - 1, 0)*size**2)] ):
																																			 
				
			#else:
			#
			#	potential_new[u32(i + j*size + k*size**2)] = (2 * potential_new[u32(i + j*size + k*size**2)] + potential_old[u32(i + j*size + k*size**2)] * ( dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] - 1 ) + ( csq * dt**2 / ( ds * ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)]) ) ) * (
			#		(
			#			8 * (cmath.rect( normal[u32(min(i + 1, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 1, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 1, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 1, size-1) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + min(j + 1, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 1, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 1, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 1, size-1)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + min(k + 1, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 1, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 1, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 1, size-1)*size**2)])
			#			) + cmath.rect( normal[u32(max(i - 2, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 2, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 2, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 2, 0) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + max(j-2, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-2, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-2, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-2, 0)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + max(k-2, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-2, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-2, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-2, 0)*size**2)])
			#			-  8 *(cmath.rect( normal[u32(max(i - 1, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 1, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 1, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 1, 0) + j*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + max(j-1, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-1, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-1, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-1, 0)*size + k*size**2)])
			#			+ cmath.rect( normal[u32(i + j*size + max(k-1, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-1, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-1, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-1, 0)*size**2)])
			#			) - cmath.rect( normal[u32(min(i + 2, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 2, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 2, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 2, size-1) + j*size + k*size**2)])
			#			- cmath.rect( normal[u32(i + min(j + 2, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 2, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 2, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 2, size-1)*size + k*size**2)])
			#			- cmath.rect( normal[u32(i + j*size + min(k + 2, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 2, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 2, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 2, size-1)*size**2)])
			#		)  /  ( 12 ) ) + 0.5 * (
			#			 (field[u32(min(i + 1, size-1) + j*size + k*size**2 + field.shape[0] / 2)] - field[u32(min(i + 1, size-1) + j*size + k*size**2)] + field[u32(max(i - 1, 0) + j*size + k*size**2)] - field[u32(max(i - 1, 0) + j*size + k*size**2 + field.shape[0] / 2)]) * (
			#				 cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
			#				 - potential_new[u32(min(i + 1, size-1) + j*size + k*size**2)] + potential_new[u32(max(i - 1, 0) + j*size + k*size**2)]
			#			) + (field[u32(i + min(j + 1, size-1)*size + k*size**2 + field.shape[0] / 2)] - field[u32(i + min(j + 1, size-1)*size + k*size**2)] + field[u32(i + max(j-1, 0)*size + k*size**2)] - field[u32(i + max(j-1, 0)*size + k*size**2 + field.shape[0] / 2)]) * (
			#				 cmath.rect( normal[u32(i + j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
			#				 - potential_new[u32(i + min(j + 1, size-1)*size + k*size**2)] + potential_new[u32(i + max(j-1, 0)*size + k*size**2)]
			#			) + (field[u32(i + j*size + min(k + 1, size-1)*size**2 + field.shape[0] / 2)] - field[u32(i + j*size + min(k + 1, size-1)*size**2)] + field[u32(i + j*size + max(k-1, 0)*size**2)] - field[u32(i + j*size + max(k-1, 0)*size**2 + field.shape[0] / 2)]) * (
			#				 cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)], emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
			#				 - potential_new[u32(i + j*size + min(k + 1, size-1)*size**2)] + potential_new[u32(i + j*size + max(k-1, 0)*size**2)]
			#			)						
			#		) / ds ) / ( 0.5 * dt * field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] + 1 )
				
			
				
				#(
				#	8 * (cmath.rect( normal[u32(min(i + 1, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 1, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 1, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 1, size-1) + j*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + min(j + 1, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 1, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 1, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 1, size-1)*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + j*size + min(k + 1, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 1, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 1, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 1, size-1)*size**2)])
				#	) + cmath.rect( normal[u32(max(i - 2, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 2, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 2, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 2, 0) + j*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + max(j-2, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-2, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-2, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-2, 0)*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + j*size + max(k-2, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-2, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-2, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-2, 0)*size**2)])
				#	-  8 *(cmath.rect( normal[u32(max(i - 1, 0) + j*size + k*size**2)] * emitters_amplitude[u32(max(i - 1, 0) + j*size + k*size**2)], emitters_frequency[u32(max(i - 1, 0) + j*size + k*size**2)]*time + emitters_phase[u32(max(i - 1, 0) + j*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + max(j-1, 0)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + max(j-1, 0)*size + k*size**2)], emitters_frequency[u32(i + max(j-1, 0)*size + k*size**2)]*time + emitters_phase[u32(i + max(j-1, 0)*size + k*size**2)])
				#	+ cmath.rect( normal[u32(i + j*size + max(k-1, 0)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + max(k-1, 0)*size**2)], emitters_frequency[u32(i + j*size + max(k-1, 0)*size**2)]*time + emitters_phase[u32(i + j*size + max(k-1, 0)*size**2)])
				#	) - cmath.rect( normal[u32(min(i + 2, size-1) + j*size + k*size**2)] * emitters_amplitude[u32(min(i + 2, size-1) + j*size + k*size**2)], emitters_frequency[u32(min(i + 2, size-1) + j*size + k*size**2)]*time + emitters_phase[u32(min(i + 2, size-1) + j*size + k*size**2)])
				#	- cmath.rect( normal[u32(i + min(j + 2, size-1)*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + min(j + 2, size-1)*size + k*size**2)], emitters_frequency[u32(i + min(j + 2, size-1)*size + k*size**2)]*time + emitters_phase[u32(i + min(j + 2, size-1)*size + k*size**2)])
				#	- cmath.rect( normal[u32(i + j*size + min(k + 2, size-1)*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + min(k + 2, size-1)*size**2)], emitters_frequency[u32(i + j*size + min(k + 2, size-1)*size**2)]*time + emitters_phase[u32(i + j*size + min(k + 2, size-1)*size**2)])
				#)  /  ( 12 * ds )															
			
	
			

	#cuda.syncthreads()
	
def step_all_c128_noreturn_cuda(pressure_new, pressure_old, field, dt, ds, rho, normal, emitters_amplitude, emitters_frequency, emitters_phase, time, size):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	auxiliar = cuda.local.array((MAXTHREADNUMBER,MAXTHREADNUMBER,MAXTHREADNUMBER), c128)
		#shared_position[0] = 0
		
	cuda.syncthreads()
		
	if i< u32(size) and j< u32(size) and k< u32(size):
		
		#auxiliar[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z] = pressure_new[u32(i + j*size + k*size**2)]
		auxiliar[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z] = c128(pressure_new[u32(i + j*size + k*size**2)])
		#cuda.syncthreads()
		
		#if pressure_new[u32(i + j*size + k*size**2)].real >= pressure_old[u32(i + j*size + k*size**2)].real:
		#	auxiliar_real = cuda.atomic.max(pressure_old.real, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].real)
		#
		#else:
		#	auxiliar_real = cuda.atomic.min(pressure_old.real, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].real)
		#	
		#if pressure_new[u32(i + j*size + k*size**2)].imag >= pressure_old[u32(i + j*size + k*size**2)].imag:
		#	auxiliar_imag = cuda.atomic.max(pressure_old.imag, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].imag)
		#
		#else:
		#	auxiliar_imag = cuda.atomic.min(pressure_old.imag, u32(i + j*size + k*size**2), pressure_new[u32(i + j*size + k*size**2)].imag)
		
		if emitters_amplitude[u32(i + j*size + k*size**2)] == 0:
		
			pressure_new[u32(i + j*size + k*size**2)] = (pressure_new[u32(i + j*size + k*size**2)] * ( 1 - field[u32(i + j*size + k*size**2)] ) * ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt + 
														field[u32(i + j*size + k*size**2)]**2 * rho * dt**2 * (
															pressure_new[u32(i + 1 + j*size + k*size**2)] + pressure_new[u32(i + (j+1)*size + k*size**2)] + pressure_new[u32(i + j*size + (k+1)*size**2)] - 6 * pressure_new[u32(i + j*size + k*size**2)] +
															pressure_new[u32(i - 1 + j*size + k*size**2)] + pressure_new[u32(i + (j-1)*size + k*size**2)] + pressure_new[u32(i + j*size + (k-1)*size**2)]
															)/ (ds**2) + field[u32(i + j*size + k*size**2)] * pressure_old[u32(i + j*size + k*size**2)] ) / (
																(1 + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt) * (field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt)
																)

		else:
			
			if field[u32(i + j*size + k*size**2)] == 0:
				
				pressure_new[u32(i + j*size + k*size**2)] = (pressure_new[u32(i + j*size + k*size**2)] - rho * dt * (cmath.rect( normal[u32(i + 1 + j*size + k*size**2)] * emitters_amplitude[u32(i + 1 + j*size + k*size**2)],
																		emitters_frequency[u32(i + 1 + j*size + k*size**2)]*time + emitters_phase[u32(i + 1 + j*size + k*size**2)]) + cmath.rect( normal[u32(i + (j+1)*size + k*size**2)] * emitters_amplitude[u32(i + (j+1)*size + k*size**2)],
																		emitters_frequency[u32(i + (j+1)*size + k*size**2)]*time + emitters_phase[u32(i + (j+1)*size + k*size**2)]) + cmath.rect( normal[u32(i + j*size + (k+1)*size**2)] * emitters_amplitude[u32(i + j*size + (k+1)*size**2)],
																		emitters_frequency[u32(i + j*size + (k+1)*size**2)]*time + emitters_phase[u32(i + j*size + (k+1)*size**2)]) - 3 * cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																		emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
												 
																) / field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) / (
																(1 + ( 1 + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt) 
																)
				
			else:
				
				pressure_new[u32(i + j*size + k*size**2)] = (pressure_new[u32(i + j*size + k*size**2)] * ( 1 - field[u32(i + j*size + k*size**2)] ) * ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt + 
														field[u32(i + j*size + k*size**2)]**2 * rho * dt**2 * (
															pressure_new[u32(i + 1 + j*size + k*size**2)] + pressure_new[u32(i + (j+1)*size + k*size**2)] + pressure_new[u32(i + j*size + (k+1)*size**2)] - 6 * pressure_new[u32(i + j*size + k*size**2)] +
															pressure_new[u32(i - 1 + j*size + k*size**2)] + pressure_new[u32(i + (j-1)*size + k*size**2)] + pressure_new[u32(i + j*size + (k-1)*size**2)]
															)/ (ds**2) + field[u32(i + j*size + k*size**2)] * pressure_old[u32(i + j*size + k*size**2)] -
															rho * dt**2 * (cmath.rect( normal[u32(i + 1 + j*size + k*size**2)] * emitters_amplitude[u32(i + 1 + j*size + k*size**2)],
																		emitters_frequency[u32(i + 1 + j*size + k*size**2)]*time + emitters_phase[u32(i + 1 + j*size + k*size**2)]) + cmath.rect( normal[u32(i + (j+1)*size + k*size**2)] * emitters_amplitude[u32(i + (j+1)*size + k*size**2)],
																		emitters_frequency[u32(i + (j+1)*size + k*size**2)]*time + emitters_phase[u32(i + (j+1)*size + k*size**2)]) + cmath.rect( normal[u32(i + j*size + (k+1)*size**2)] * emitters_amplitude[u32(i + j*size + (k+1)*size**2)],
																		emitters_frequency[u32(i + j*size + (k+1)*size**2)]*time + emitters_phase[u32(i + j*size + (k+1)*size**2)]) - 3 * cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																		emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)]))) / (
																(1 + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt) * (field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt)
																)
		
		pressure_old[u32(i + j*size + k*size**2)] = npc128(auxiliar[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z])
			

	#cuda.syncthreads()


def step_velocity_values_n_emitters_noreturn_cuda (velocity, pressure, field, dt, ds, rho, normal, emitters_amplitude, emitters_frequency, emitters_phase, time, size):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i< u32(size) and j< u32(size) and k< u32(size):
		
		if emitters_amplitude[u32(i + j*size + k*size**2)] == 0:
			
			if i > 0:
			
				velocity[u32(i + j*size + k*size**2)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2)] - field[u32(i + j*size + k*size**2)]**2 * dt *( 
									(pressure[u32(i + j*size + k*size**2)] - pressure[u32(i-1 + j*size + k*size**2)]) / ( ds * rho ) ) 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
			else:
				
				velocity[u32(i + j*size + k*size**2)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2)] 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
				
			if j > 0:
		
				velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] - field[u32(i + j*size + k*size**2)]**2 * dt * (
								(pressure[u32(i + j*size + k*size**2)] - pressure[ u32(i + (j-1)*size + k*size**2)]) / ( ds * rho ) ) 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
				
			else:
				
				velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
		
			if k > 0:

				velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] - field[u32(i + j*size + k*size**2)]**2 * dt * (
								(pressure[u32(i + j*size + k*size**2)] - pressure[u32(i + j*size + (k-1)*size**2)]) / ( ds * rho ) ) 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
				
			else:
				
				velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
		

		else:
			
			if field[u32(i + j*size + k*size**2)] == 0:
				
				velocity[u32(i + j*size + k*size**2)] = cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																		emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
					
				velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] = cmath.rect( normal[u32(i + j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																							emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)]) 
				
				velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] = cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																							emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
				
			else:
			
				if i > 0:
			
					velocity[u32(i + j*size + k*size**2)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2)] - field[u32(i + j*size + k*size**2)]**2 * dt *( 
										(pressure[u32(i + j*size + k*size**2)] - pressure[u32(i-1 + j*size + k*size**2)]) / ( ds * rho ) ) +
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
									) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
				else:
				
					velocity[u32(i + j*size + k*size**2)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2)] +
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
									) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )
				
				if j > 0:
		
					velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] - field[u32(i + j*size + k*size**2)]**2 * dt * (
									(pressure[u32(i + j*size + k*size**2)] - pressure[ u32(i + (j-1)*size + k*size**2)]) / ( ds * rho ) )+
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)]) 
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt)
				
				else:
				
					velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + velocity.shape[0] / 3)] +
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2 + normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)]) 
									) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt)
		
				if k > 0:

					velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] - field[u32(i + j*size + k*size**2)]**2 * dt * (
									(pressure[u32(i + j*size + k*size**2)] - pressure[u32(i + j*size + (k-1)*size**2)]) / ( ds * rho ) ) +
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
								) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt  )
				
				else:
				
					velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)] = ( field[u32(i + j*size + k*size**2)] * velocity[u32(i + j*size + k*size**2 + 2*velocity.shape[0] / 3)]  +
									( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt * cmath.rect( normal[u32(i + j*size + k*size**2 + 2*normal.shape[0] / 3)] * emitters_amplitude[u32(i + j*size + k*size**2)],
																																								emitters_frequency[u32(i + j*size + k*size**2)]*time + emitters_phase[u32(i + j*size + k*size**2)])
									) / ( field[u32(i + j*size + k*size**2)] + ( 1 - field[u32(i + j*size + k*size**2)] + field[u32(i + j*size + k*size**2 + field.shape[0] / 2)] ) * dt )

def step_velocity_values_noreturn_cuda (velocity, v_b, pressure, beta, sigma, dt, ds, rho):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<pressure.shape[0] and j<pressure.shape[1] and k<pressure.shape[2]:
		
		velocity[i,j,k,0] = ( beta[i, j, k] * velocity[i, j, k, 0] - beta[i, j, k]**2 * dt *( 
							(pressure[i, j, k] - pressure[max(0, i-1), j, k]) / ( ds * rho ) ) +
							( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k,0]
						) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
		
		velocity[i,j,k,1] = ( beta[i, j, k] * velocity[i, j, k, 1] - beta[i, j, k]**2 * dt * (
							(pressure[i, j, k] - pressure[i, max(0, j-1), k]) / ( ds * rho ) ) +
							( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k,1]
						) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
		
		velocity[i,j,k,2] = ( beta[i, j, k] * velocity[i, j, k, 2] - beta[i, j, k]**2 * dt * (
							(pressure[i, j, k] - pressure[i, j, max(0, k-1)]) / ( ds * rho ) ) +
							( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k,2]
						) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
			
def step_pressure_values_noreturn_cuda (pressure, velocity, field, dt, ds, rho_csq, size):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i< int(size) and j< int(size) and k< int(size): #Move to the last else?
		
		if i < size - 1:

			if j < size - 1:
		
				if k < size - 1:
					
					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i+1 + int(j*size + k*size**2)] - velocity[i + int(j*size + k*size**2)] + velocity[i + int((j+1)*size + k*size**2) + int(velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(velocity.shape[0] / 3)]
										+ velocity[i + int(j*size + (k+1)*size**2) + int(2*velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(2* velocity.shape[0] / 3)] ) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

				else:
				
					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i+1 + int(j*size + k*size**2)] - velocity[i + int(j*size + k*size**2)] + velocity[i + int((j+1)*size + k*size**2) + int(velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(velocity.shape[0] / 3)]
										) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

			else:
				
				
				if k < size - 1:

					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i+1 + int(j*size + k*size**2)] - velocity[i + int(j*size + k*size**2)]
										+ velocity[i + int(j*size + (k+1)*size**2) + int(2*velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(2* velocity.shape[0] / 3)] ) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )
					
				else:
				
					pressure[i + int(j*size + k*size**2)] =  ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i+1 + int(j*size + k*size**2)] - velocity[i + int(j*size + k*size**2)]
										) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

		

		else:
				
			if j < size - 1:
		
				if k < size - 1:

					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i + int((j+1)*size + k*size**2) + int(velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(velocity.shape[0] / 3)]
										+ velocity[i + int(j*size + (k+1)*size**2) + int(2*velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(2* velocity.shape[0] / 3)] ) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

				else:
				
					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i + int((j+1)*size + k*size**2) + int(velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(velocity.shape[0] / 3)]
										) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

			else:
				
				
				if k < size - 1:

					pressure[i + int(j*size + k*size**2)] = ( pressure[i + int(j*size + k*size**2)] - rho_csq * dt * (
										velocity[i + int(j*size + (k+1)*size**2) + int(2*velocity.shape[0] / 3)] - velocity[i + int(j*size + k*size**2) + int(2* velocity.shape[0] / 3)] ) / ( ds )
										) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )
					
				else:
				
					pressure[i + int(j*size + k*size**2)] =  ( pressure[i + int(j*size + k*size**2)] ) / ( 1 + ( 1 - field[i + int(j*size + k*size**2)] + field[i + int(j*size + k*size**2) + int(field.shape[0] / 2)] ) * dt )

		
def set_velocity_emitters_noreturn_cuda (velocity_boundary, normal, emitters_amplitude, emitters_frequency, emitters_phase, time):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<velocity_boundary.shape[0] and j<velocity_boundary.shape[1] and k<velocity_boundary.shape[2]:
		if emitters_amplitude[i, j, k]!=0:
			velocity_boundary[i, j, k,0] = cmath.rect( normal[i, j, k,0] * emitters_amplitude[i, j, k], emitters_frequency[i, j, k]*time + emitters_phase[i, j, k])
			velocity_boundary[i, j, k,1] = cmath.rect( normal[i, j, k,1] * emitters_amplitude[i, j, k], emitters_frequency[i, j, k]*time + emitters_phase[i, j, k])
			velocity_boundary[i, j, k,2] = cmath.rect( normal[i, j, k,2] * emitters_amplitude[i, j, k], emitters_frequency[i, j, k]*time + emitters_phase[i, j, k])

def set_pressure_emitters_noreturn_cuda (velocity_boundary, normal, emitters_amplitude, emitters_frequency, emitters_phase, time):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<velocity_boundary.shape[0] and j<velocity_boundary.shape[1] and k<velocity_boundary.shape[2]:
		if emitters_amplitude[i, j, k]!=0:
			velocity_boundary[i, j, k] = cmath.rect( normal[i, j, k] * emitters_amplitude[i, j, k], emitters_frequency[i, j, k]*time + emitters_phase[i, j, k])


class calculator_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, nPoints, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.stream = stream
		
		self.config = None
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		#self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
		self.maxThreadsPerBlock = int(cuda.get_current_device().MAX_THREADS_PER_BLOCK)
		
		self.nPoints = nPoints
		
		self.controller = True
		
		self.config_calculator(size, blockdim, stream)

		self.config_calculator_functions()
		
	
	'''
	Implement configurations
	''' 

	def config_calculator_functions (self):
		try:
			self.config = {
				'step_velocity_values':							cuda.jit('void('+self.OutVarType+'[:,:,:,:], '	+self.OutVarType+'[:,:,:,:], '	+self.OutVarType+'[:,:,:], '
																				+self.VarType+'[:,:,:], '		+self.VarType+'[:,:,:], '		+self.VarType+', '
																				+self.VarType+', '				+self.VarType+')', fastmath = True)(step_velocity_values_noreturn_cuda),
				'step_pressure_values':							cuda.jit('void('+self.OutVarType+'[:], '		+self.OutVarType+'[:], '		+self.VarType+'[:], '	
																				+self.VarType+', '				+self.VarType+', '				+self.VarType+', int64)', fastmath = True)(step_pressure_values_noreturn_cuda),
				'set_velocity_emitters':						cuda.jit('void('+self.OutVarType+'[:,:,:,:], '	+self.VarType+'[:,:,:,:], '		+self.VarType+'[:,:,:], '+self.VarType+'[:,:,:], '
																				+self.VarType+'[:,:,:], '		+self.VarType+')', fastmath = True)(set_velocity_emitters_noreturn_cuda),
				'step_velocity_values_n_emitters':				cuda.jit('void('+self.OutVarType+'[:], '		+self.OutVarType+'[:], '		+self.VarType+'[:], '
																				+self.VarType+', '				+self.VarType+', '				+self.VarType +', '
																				+self.VarType+'[:], '			+self.VarType+'[:], '			+self.VarType+'[:], '
																				+self.VarType+'[:], '			+self.VarType+', int64)', fastmath = True)(step_velocity_values_n_emitters_noreturn_cuda),
				'step_all_c64_fw': 								cuda.jit('void('+self.OutVarType+'[:], '		+self.OutVarType+'[:], '		+self.VarType+'[:], '
																				+self.VarType+', '				+self.VarType+', '				+self.VarType +', '																				
																				+self.VarType+'[:], '			+self.VarType+'[:], '			+self.VarType+'[:], '
																				+self.VarType+'[:], '			+self.VarType+', int64)', fastmath = True)(step_all_c64_forward_noreturn_cuda),
				'step_all_c64_bw': 								cuda.jit('void('+self.OutVarType+'[:], '		+self.OutVarType+'[:], '		+self.VarType+'[:], '
																				+self.VarType+', '				+self.VarType+', '				+self.VarType +', '																				
																				+self.VarType+'[:], '			+self.VarType+'[:], '			+self.VarType+'[:], '
																				+self.VarType+'[:], '			+self.VarType+', int64)', fastmath = True)(step_all_c64_backward_noreturn_cuda),				
				'step_all_c128': 								cuda.jit('void('+self.OutVarType+'[:], '		+self.OutVarType+'[:], '		+self.VarType+'[:], '
																				+self.VarType+', '				+self.VarType+', '				+self.VarType +', '
																				+self.VarType+'[:], '			+self.VarType+'[:], '			+self.VarType+'[:], '
																				+self.VarType+'[:], '			+self.VarType+', int64)', fastmath = True)(step_all_c128_noreturn_cuda)				
																	
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.config_calculator_functions: {e}')

	def config_calculator(self, size=None, blockdim = None, stream = None):
		try:
			assert size is not None or blockdim is not None or stream is not None, 'No reconfiguration especified.'
			#assert len(blockdim)==2, 'Incorrect number of parameters.'

			if size is not None:
				self.size = size
			if blockdim is not None:
				self.blockdim = blockdim
			if self.blockdim is not None and self.size is not None:
				
				if isinstance(self.size,int):
					self.griddim = int(np.ceil(self.size / self.blockdim[0]))
				elif len(size) == 2:
					self.griddim = (int(np.ceil(self.size[0] / self.blockdim[0])), int(np.ceil(self.size[1] / self.blockdim[1])))
				elif len(size) == 3:
					self.griddim = (int(np.ceil(self.size[0] / self.blockdim[0])), int(np.ceil(self.size[1] / self.blockdim[1])), int(np.ceil(self.size[2] / self.blockdim[2])))
			#print(self.blockdim)
			if stream is not None:
				self.stream = stream
		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.config_calculator: {e}')
	

	
	'''
	Implement functions that do the things
	'''
	
	def set_velocity_emitters (self, velocity_boundary, emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(velocity_boundary) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_amplitude)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_frequency) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_phase)), 'Arrays must be loaded in GPU device.'

			self.config_calculator(size=(velocity_boundary.shape[0], velocity_boundary.shape[1], velocity_boundary.shape[2]), 
						  blockdim=optimize_blockdim(self.maxThreadsPerBlock, velocity_boundary.shape[0], velocity_boundary.shape[1], velocity_boundary.shape[2]))
			
			#print(velocity_boundary.copy_to_host(), emitters_amplitude.copy_to_host(), emitters_frequency.copy_to_host(), emitters_phase.copy_to_host(), time)

			self.config['set_velocity_emitters'][self.griddim, self.blockdim, self.stream](velocity_boundary, emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.set_velocity_emitters: {e}')

	def step_velocity_values (self, velocity, v_b, pressure, beta, sigma, dt, ds, rho):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(velocity) and cuda.cudadrv.devicearray.is_cuda_ndarray(v_b)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(pressure) and cuda.cudadrv.devicearray.is_cuda_ndarray(beta)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(sigma)), 'Arrays must be loaded in GPU device.'
			#assert int(axis) in [0, 1, 2], f'Axis {axis} not valid.'

			self.config_calculator(size=(velocity.shape[0], velocity.shape[1], velocity.shape[2]), blockdim=optimize_blockdim(self.maxThreadsPerBlock, velocity.shape[0], velocity.shape[1], velocity.shape[2]))
			
			self.config['step_velocity_values'][self.griddim, self.blockdim, self.stream](velocity, v_b, pressure, beta, sigma, dt, ds, rho)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_velocity_values: {e}')

	def step_velocity_values_n_emitters (self, velocity, pressure, field, dt, ds, rho, emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(velocity) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_amplitude)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_frequency) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_phase)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(pressure) and cuda.cudadrv.devicearray.is_cuda_ndarray(field)), 'Arrays must be loaded in GPU device.'
			#assert int(axis) in [0, 1, 2], f'Axis {axis} not valid.'

			self.config_calculator(size=(self.nPoints, self.nPoints, self.nPoints), blockdim=optimize_blockdim(self.maxThreadsPerBlock, self.nPoints, self.nPoints, self.nPoints))
			
			self.config['step_velocity_values_n_emitters'][self.griddim, self.blockdim, self.stream](velocity, pressure, field, dt, ds, rho,
																						emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time, self.nPoints)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_velocity_values_n_emitters: {e}')


	def step_pressure_values (self, pressure, velocity, field, dt, ds, rho_csq):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(pressure) and cuda.cudadrv.devicearray.is_cuda_ndarray(velocity)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(field)), 'Arrays must be loaded in GPU device.'
			
			self.config_calculator(size=(self.nPoints, self.nPoints, self.nPoints), blockdim=optimize_blockdim(self.maxThreadsPerBlock, self.nPoints, self.nPoints, self.nPoints))
			
			self.config['step_pressure_values'][self.griddim, self.blockdim, self.stream](pressure, velocity, field, dt, ds, rho_csq, self.nPoints)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_pressure_values: {e}')
			
	def step_all (self, pressure_new, pressure_old, field, dt, ds, csq, emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(pressure_new) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_amplitude)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_frequency) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitters_phase)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(pressure_old) and cuda.cudadrv.devicearray.is_cuda_ndarray(field)), 'Arrays must be loaded in GPU device.'
			#assert int(axis) in [0, 1, 2], f'Axis {axis} not valid.'

			#print(pressure_old.dtype)

			self.config_calculator(size=(self.nPoints, self.nPoints, self.nPoints), blockdim=optimize_blockdim(self.maxThreadsPerBlock, self.nPoints, self.nPoints, self.nPoints))
			if self.OutVarType == 'complex64':
				if self.controller:
					self.controller = False
					self.config['step_all_c64_fw'][self.griddim, self.blockdim, self.stream](pressure_new, pressure_old, field, dt, ds, csq,
																							emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time, self.nPoints)
				else:
					self.controller = True
					self.config['step_all_c64_bw'][self.griddim, self.blockdim, self.stream](pressure_new, pressure_old, field, dt, ds, csq,
																							emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time, self.nPoints)
			elif self.OutVarType == 'complex128':
				self.config['step_all_c128'][self.griddim, self.blockdim, self.stream](pressure_new, pressure_old, field, dt, ds, 0, csq,
																						emitters_normal, emitters_amplitude, emitters_frequency, emitters_phase, time, self.nPoints)
				

			#print(pressure_new.copy_to_host())

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_all: {e}')

