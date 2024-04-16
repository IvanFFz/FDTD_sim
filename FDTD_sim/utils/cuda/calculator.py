import numpy as np, numba, math, cmath
from numba import cuda

def calculate_bd_b2 (number, base, max_value = 1024):
    return min(int(base**(np.ceil(np.log(number)/np.log(base))-1)), max_value)

def optimize_blockdim (size_0, size_1=0, size_2=0):
	threshold_1 = 1.5
	threshold_2 = 1.5
	bd_0 = calculate_bd_b2(size_0, 2)
	param = 7

	if size_1==0 and size_2==0:
		
		#print(size_0)
		
		exp_bd = np.log(bd_0/32)/np.log(2)
		
		#exp_32 = exp_bd/5
		exp_256 = exp_bd / param
		
		if exp_256 <= threshold_1:
			bd = (max(1, calculate_bd_b2(size_0/(2**param), 2, 32)),)
		elif exp_256 > threshold_1:
			bd = (max(1, calculate_bd_b2(size_0/(2**param), 2)),)
			
	elif size_2 == 0:
		
		bd_1 = calculate_bd_b2(size_1, 2)
		
		#print(size_0, size_1)
		
		if bd_1<10:
			
			exp_bd = np.log(bd_0/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			exp_256 = exp_bd / param
		
			if exp_256 <= threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), 2, 32)), max(1, size_1))
			elif exp_256 > threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), 2)), max(1, size_1))
			
		else:
						
			exp_bd_0 = np.log(bd_0/32)/np.log(2)
			exp_bd_1 = np.log(bd_1/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			if exp_bd_0 > exp_bd_1:
				exp_0 = exp_bd_0 / np.ceil(np.log(2**param)/np.log(2))-1
				exp_1 = exp_bd_1 / np.ceil(np.log(2**param)/np.log(2))-2
				div_0 = 16
				div_1 = 8
			elif exp_bd_0 < exp_bd_1:
				exp_0 = exp_bd_0 / np.ceil(np.log(2**param)/np.log(2))-2
				exp_1 = exp_bd_1 / np.ceil(np.log(2**param)/np.log(2))-1
				div_0 = 8
				div_1 = 16
			else:
				exp_0 = exp_bd_0 / np.ceil(np.log(2**param)/np.log(2))-1
				exp_1 = exp_bd_1 / np.ceil(np.log(2**param)/np.log(2))-1
				div_0 = 16
				div_1 = 16
		
			if exp_0 <= threshold_2 and exp_1 <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2, div_0)), (max(1, calculate_bd_b2(size_1/div_1, 2, div_1))))
			elif exp_0 > threshold_2 and exp_1 <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2)),  (max(1, calculate_bd_b2(size_1/div_1, 2, div_1))))
			elif exp_0 <= threshold_2 and exp_1 > threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2, div_0)), (max(1, calculate_bd_b2(size_1/div_1, 2))))
			else:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2)),  (max(1, calculate_bd_b2(size_1/div_1, 2))))
				
	else:
		
		bd_1 = calculate_bd_b2(size_1, 2)
		bd_2 = calculate_bd_b2(size_2, 2)
		
		#print(size_0, size_1)
		
		if max(bd_1, bd_2)<10:
			
			exp_bd = np.log(bd_0/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			exp_256 = exp_bd / param
		
			if exp_256 <= threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), 2, 32)), max(1, size_1), max(1, size_2))
			elif exp_256 > threshold_1:
				bd = (max(1, calculate_bd_b2(size_0*size_1/(2**param), 2)), max(1, size_1), max(1, size_2))
			
		else:
			
			exp_bd_0 = np.log(bd_0/32)/np.log(2)
			exp_bd_1 = np.log(bd_1/32)/np.log(2)
			exp_bd_2 = np.log(bd_2/32)/np.log(2)
		
			#exp_32 = exp_bd/5
			if exp_bd_0 < max(exp_bd_1, exp_bd_2):
				exp_0 = exp_bd_0 / np.ceil(np.log(2**param)/np.log(2))-2
				exp_1 = exp_bd_1 / np.ceil(np.log(2**param)/np.log(2))-1
				exp_2 = exp_bd_2 / np.ceil(np.log(2**param)/np.log(2))-1
				div_0 = 4
				div_1 = 8
				div_2 = 8
			else:
				exp_0 = exp_bd_0 / np.ceil(np.log(2**param)/np.log(2))-1
				exp_1 = exp_bd_1 / np.ceil(np.log(2**param)/np.log(2))-1
				exp_2 = exp_bd_2 / np.ceil(np.log(2**param)/np.log(2))-1
				div_0 = 8
				div_1 = 4
				div_2 = 4
		
			if exp_0 <= threshold_2 and max(exp_bd_1, exp_bd_2) <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2, div_0)), (max(1, calculate_bd_b2(size_1/div_1, 2, div_1))), (max(1, calculate_bd_b2(size_2/div_2, 2, div_2))))
			elif exp_0 > threshold_2 and max(exp_bd_1, exp_bd_2) <= threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2)),  (max(1, calculate_bd_b2(size_1/div_1, 2, div_1))),  (max(1, calculate_bd_b2(size_2/div_2, 2, div_2))))
			elif exp_0 <= threshold_2 and max(exp_bd_1, exp_bd_2) > threshold_2:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2, div_0)), (max(1, calculate_bd_b2(size_1/div_1, 2))), (max(1, calculate_bd_b2(size_2/div_2, 2))))
			else:
				bd = (max(1, calculate_bd_b2(size_0/div_0, 2)),  (max(1, calculate_bd_b2(size_1/div_1, 2))),  (max(1, calculate_bd_b2(size_2/div_2, 2))))
		
	
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
def step_velocity_values_noreturn_cuda (v_out, v_in, v_b, pressure, beta, sigma, dt, ds, rho, axis):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<pressure.shape[0] and j<pressure.shape[1] and k<pressure.shape[2]:
		if axis == 0:
			v_out[i,j,k] = ( beta[i, j, k] * v_in[i, j, k] - beta[i, j, k]**2 * dt *(
					0.5 * ( pressure[min(pressure.shape[0]-1, i+1), j, k] - pressure[max(0, i-1), j, k]) / ( ds * rho ) ) +
					( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k]
				) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
		elif axis == 1:
			v_out[i,j,k] = ( beta[i, j, k] * v_in[i, j, k] - beta[i, j, k]**2 * dt * (
					0.5 * ( pressure[i, min(pressure.shape[0]-1, j+1), k] - pressure[i, max(0, j-1), k]) / ( 2 * ds * rho ) )+
					( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k]
				) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
		elif axis == 2:
			v_out[i,j,k] = ( beta[i, j, k] * v_in[i, j, k] - beta[i, j, k]**2 * dt * (
					0.5 * ( pressure[i, j, min(pressure.shape[0]-1, k+1)] - pressure[i, j, max(0, k-1)]) / ( 2 * ds * rho ) ) +
					( 1 - beta[i,j,k] + sigma[i, j, k] ) * dt * v_b[i,j,k]
				) / ( beta[i, j, k] + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
			
def step_pressure_values_noreturn_cuda (p_out, p_in, vx, vy, vz, beta, sigma, dt, ds, rho_csq):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<p_in.shape[0] and j<p_in.shape[1] and k<p_in.shape[2]:
		
		p_out[i,j,k] = ( p_in[i, j, k] - rho_csq * dt * (
				vx[min(vx.shape[0]-1, i+1), j, k] - vx[max(0, i-1), j, k] + vy[i, min(vy.shape[0]-1, j+1), k] - vy[i, max(0, j-1), k] + vz[i, j, min(vz.shape[0]-1, k+1)] - vz[i, j, max(0, k-1)] ) / ( 2 * ds )
			) / ( 1 + ( 1 - beta[i, j, k] + sigma[i, j, k] ) * dt )
		
def set_velocity_emitters (velocity_boundary, emitter_location, emitter_properties, time):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<emitter_location.shape[0]:
		velocity_boundary[emitter_location[i, 0], emitter_location[i, 1], emitter_location[i, 2]] = cmath.rect(emitter_properties[i, 0], emitter_properties[i, 1]*time + emitter_properties[i, 2])


class calculator_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

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

		self.config_calculator(size, blockdim, stream)

		self.config_calculator_functions()
		
	
	'''
	Implement configurations
	''' 

	def config_calculator_functions (self):
		try:
			self.config = {
				'step_velocity_values':				cuda.jit('void('+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '
																	+self.VarType+'[:,:,:], '	+self.VarType+'[:,:,:], '	+self.VarType+', '			+self.VarType+', '
																	+self.VarType+', '			+'int64)', fastmath = True)(step_velocity_values_noreturn_cuda),
				'step_pressure_values':				cuda.jit('void('+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '
																	+self.OutVarType+'[:,:,:], '+self.VarType+'[:,:,:], '	+self.VarType+'[:,:,:], '	+self.VarType+', '
																	+self.VarType+', '			+self.VarType+')', fastmath = True)(step_pressure_values_noreturn_cuda),
				'set_velocity_emitters':			cuda.jit('void('+self.OutVarType+'[:,:,:], int64[:,:], '+self.VarType+'[:,:], '+self.VarType+')', fastmath = True)(step_pressure_values_noreturn_cuda)	
																	
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
			
			if stream is not None:
				self.stream = stream
		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.config_calculator: {e}')
	

	
	'''
	Implement functions that do the things
	'''
	
	def set_velocity_emitters (self, velocity_boundary, emitter_location, emitter_properties, time):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(velocity_boundary) and cuda.cudadrv.devicearray.is_cuda_ndarray(emitter_location) 
					and cuda.cudadrv.devicearray.is_cuda_ndarray(emitter_properties) ), 'Arrays must be loaded in GPU device.'

			self.config_calculator(size=(emitter_location.shape[0]), blockdim=optimize_blockdim(emitter_location.shape[0]))
			
			self.config['step_velocity_values'][self.griddim, self.blockdim, self.stream](velocity_boundary, emitter_location, emitter_properties, time)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_velocity_values: {e}')

	def step_velocity_values (self, v_out, v_in, v_b, pressure, beta, sigma, dt, ds, rho, axis):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(v_out) and cuda.cudadrv.devicearray.is_cuda_ndarray(v_in) 
					and cuda.cudadrv.devicearray.is_cuda_ndarray(v_b) and cuda.cudadrv.devicearray.is_cuda_ndarray(pressure)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(beta) and cuda.cudadrv.devicearray.is_cuda_ndarray(sigma)), 'Arrays must be loaded in GPU device.'
			assert int(axis) in [0, 1, 2], f'Axis {axis} not valid.'

			self.config_calculator(size=(v_in.shape[0], v_in.shape[1], v_in.shape[2]), blockdim=optimize_blockdim(v_in.shape[0], v_in.shape[1], v_in.shape[2]))
			
			self.config['step_velocity_values'][self.griddim, self.blockdim, self.stream](v_out, v_in, v_b, pressure, beta, sigma, dt, ds, rho, axis)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_velocity_values: {e}')

	def step_pressure_values (self, p_out, p_in, vx, vy, vz, beta, sigma, dt, ds, rho_csq):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(p_out) and cuda.cudadrv.devicearray.is_cuda_ndarray(p_in) 
					and cuda.cudadrv.devicearray.is_cuda_ndarray(vx) and cuda.cudadrv.devicearray.is_cuda_ndarray(vy)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(vz)
					and cuda.cudadrv.devicearray.is_cuda_ndarray(beta) and cuda.cudadrv.devicearray.is_cuda_ndarray(sigma)), 'Arrays must be loaded in GPU device.'
			
			self.config_calculator(size=(p_in.shape[0], p_in.shape[1], p_in.shape[2]), blockdim=optimize_blockdim(p_in.shape[0], p_in.shape[1], p_in.shape[2]))
			
			self.config['step_pressure_values'][self.griddim, self.blockdim, self.stream](p_out, p_in, vx, vy, vz, beta, sigma, dt, ds, rho_csq)

		except Exception as e:
			print(f'Error in utils.cuda.calculator.calculator_cuda.step_pressure_values: {e}')

