import numpy as np

import numba, math, cmath
from numba import float32, float64, void, cuda
from .calculator import optimize_blockdim

				
'''
Implement functions used to manipulate arrays using cuda
'''

#def search_acoustic_shadows_no_return_cuda (origin_pos, origin_norm, all_meshes_pos, all_meshes_norm, target_pos, target_norm, shadow_relation):
	
	
def concatenate_arrays_delete_rows_noreturn_cuda (old, new, delete_rows, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if tx <(old.shape[0] + new.shape[0]) and ty < old.shape[1]:
		for tz, value in enumerate(delete_rows):
			if (tz == 0 and tx < delete_rows[0]) or (
			tz > 0 and tz <= delete_rows.shape[0] - 1 and tx < value - tz and tx > delete_rows[tz-1] - tz):
				if tx < old.shape[0]-tz:
					#ab[tx]=0
                
					out[tx,ty] = old[tx+tz,ty]
				else:
					#ab[tx]= old.shape[0]
                
					out[tx,ty] = new[tx-old.shape[0]+tz,ty]
			elif (tz == delete_rows.shape[0]-1 and tx > value - tz -1):
				if tx < old.shape[0]-tz:
					#ab[tx]=0
                
					out[tx,ty] = old[tx+tz+1,ty]
				else:
					#ab[tx]= old.shape[0]
                
					out[tx,ty] = new[tx-old.shape[0]+tz+1,ty]
	
def concatenate_arrays_noreturn_cuda (old, new, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	

	if tx<(old.shape[0] + new.shape[0]) and ty < old.shape[1]:
		if tx < old.shape[0]:
			#ab[tx]=0
                
			out[tx,ty] = old[tx,ty]
		else:
			#ab[tx]= old.shape[0]
                
			out[tx,ty] = new[tx-old.shape[0],ty]
			
def concatenate_matrix_noreturn_cuda (old_old, old_new, new_old, new_new, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	
	if tx<(old_old.shape[0] + new_new.shape[0]) and ty < (old_old.shape[1] + new_new.shape[1]):
		if tx < old_old.shape[0] and ty<old_old.shape[1]:
							
			out[tx,ty] = old_old[tx,ty]
		elif tx < old_old.shape[0] and ty>=old_old.shape[1]:
							
			out[tx,ty] = old_new[tx,ty-old_old.shape[1]]
    
		elif tx >= old_old.shape[0] and ty<old_old.shape[1]:
							
			out[tx,ty] = new_old[tx-old_old.shape[0],ty]
                        
		elif tx >= old_old.shape[0] and ty>=old_old.shape[1]:
							
			out[tx,ty] = new_new[tx-old_old.shape[0],ty-old_old.shape[1]]
                        				
def concatenate_matrix_delete_rowsncols_noreturn_cuda (old_old, old_new, new_old, new_new, delete_rowsncols, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	
	if tx<old_old.shape[0] + new_new.shape[0] and ty < old_old.shape[1] + new_new.shape[1]:
		for tz, value in enumerate(delete_rowsncols):
			if (tz == 0 and tx < delete_rowsncols[0]) or (
				tz > 0 and tz <= delete_rowsncols.shape[0] - 1 and tx < value - tz and tx > delete_rowsncols[tz-1] - tz):

				for tzz, valuee in enumerate(delete_rowsncols):
					if (tzz == 0 and ty < delete_rowsncols[0]) or (
					tzz > 0 and tzz <= delete_rowsncols.shape[0] - 1 and ty < valuee - tzz and ty > delete_rowsncols[tzz-1] - tzz):
                    
						if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = old_old[tx+tz,ty+tzz]
						elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = old_new[tx+tz,ty-old_old.shape[1]+tzz]
    
						elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = new_old[tx-old_old.shape[0]+tz,ty+tzz]
                        
						elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = new_new[tx-old_old.shape[0]+tz,ty-old_old.shape[1]+tzz]
                        
					elif (tzz == delete_rowsncols.shape[0]-1 and ty > value - tzz - 1):
						if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = old_old[tx+tz,ty+tzz+1]
						elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = old_new[tx+tz,ty-old_old.shape[1]+tzz+1]
						elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = new_old[tx-old_old.shape[0]+tz,ty+tzz+1]
						elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = new_new[tx-old_old.shape[0]+tz,ty-old_old.shape[1]+tzz+1]
                    
			elif (tz == delete_rowsncols.shape[0]-1 and tx > value - tz -1):
				for tzz, valuee in enumerate(delete_rowsncols):
					if (tzz == 0 and ty < delete_rowsncols[0]) or (
					tzz > 0 and tzz <= delete_rowsncols.shape[0] - 1 and ty < valuee - tzz and ty > delete_rowsncols[tzz-1] - tzz):
                    
						if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = old_old[tx+tz+1,ty+tzz]
						elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = old_new[tx+tz+1,ty-old_old.shape[1]+tzz]
						elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = new_old[tx-old_old.shape[0]+tz+1,ty+tzz]
						elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = new_new[tx-old_old.shape[0]+tz+1,ty-old_old.shape[1]+tzz]
					elif (tzz == delete_rowsncols.shape[0]-1 and ty > valuee - tzz - 1):
						if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = old_old[tx+tz+1,ty+tz+1]
						elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = old_new[tx+tz+1,ty-old_old.shape[1]+tzz+1]
						elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
							out[tx,ty] = new_old[tx-old_old.shape[0]+tz+1,ty+tzz+1]
						elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
							out[tx,ty] = new_new[tx-old_old.shape[0]+tz+1,ty-old_old.shape[1]+tzz+1]

# For sigma definition.
# Sigma-field full of 0 (no absorption), we substitute for a value(effect) to make the acoustic
# field response to absorption in this volume
def add_absorption_object_no_return_cuda(geometry_points, absorption, effect):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<geometry_points.shape[0]:
		
		cuda.atomic.max(absorption,(geometry_points[i, 0], geometry_points[i, 1], geometry_points[i, 2]), effect)

# For beta definition.
# Beta-field full of 1 (air), we substitute for a value between 0 and 1 to make the acoustic
# field response to objects effects		
def add_geometry_object_no_return_cuda(geometry_points, field, effect):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<geometry_points.shape[0]:
		
		cuda.atomic.min(field,(geometry_points[i, 0], geometry_points[i, 1], geometry_points[i, 2]), effect)
		
def add_extended_geometry_nPoints_no_return_cuda (geometry_points, field, distance, effect):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<geometry_points.shape[0]:
		
		for a in range(-distance, distance + 1):
			for b in range(-distance, distance + 1):
				for c in range(-distance, distance + 1):
					if abs(a)+abs(b)+abs(c) >= distance:
						cuda.atomic.min(field,(geometry_points[i, 0] + a, geometry_points[i, 1] + b, geometry_points[i, 2] + c), effect)

#Initialize the limits of the simulation volume with PML method
def PML_limit_volume_no_return_cuda (absorption, maxDist, maxValue, minValue):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<absorption.shape[0] and j<absorption.shape[1] and k<absorption.shape[2]:
		
		if (i<maxDist or i>=absorption.shape[0]-maxDist) and (j<maxDist or j>=absorption.shape[1]-maxDist) and (k<maxDist or k>=absorption.shape[2]-maxDist):
			cuda.atommic.max(absorption,
							(i,j,k),
							(maxValue - minValue)*(1 - min([i, j, k, absorption.shape[0]-i, absorption.shape[1]-j, absorption.shape[2]-k])/maxDist) + minValue)
		
#Then use concatenate arrays
#Emitters properties defined as -> Amplitude, Frequency, Phase <- In that order			
def set_point_as_emitter_no_return_cuda (emitters_amplitude, emitters_frequency, emitters_phase, geometry_points, amplitude, frequency, phase):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<geometry_points.shape[0]:
		
		emitters_amplitude[geometry_points[i, 0], geometry_points[i, 1], geometry_points[i, 2]] = amplitude[i]
		emitters_frequency[geometry_points[i, 0], geometry_points[i, 1], geometry_points[i, 2]] = frequency[i]
		emitters_phase[geometry_points[i, 0], geometry_points[i, 1], geometry_points[i, 2]] = phase[i]
		
class manager_cuda():
	
	def __init__ (self, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):
		
		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'

		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		self.stream = stream
		
		self.config = None

		self.config_manager(size, blockdim, stream)

		self.config_manager_functions()
		
		self.pressure = None
		self.velocity_x = None
		self.velocity_y = None
		self.velocity_z = None
		
		self.velocity_b = None
		self.emitters_amplitude = None
		self.emitters_frequency = None
		self.emitters_phase =None
		
		self.geometry_field = None #beta in the paper
		self.absorptivity = None #sigma in the paper
		
		self.transducers_pos_cuda = None
		self.transducers_norm_cuda = None
		self.transducers_radius = None
		
		self.emitters_pos = None
		
		self.receivers_pos = None
				
		self.mesh_pos = None
		#self.mesh_pression = { 'Old': None, 'New': None }
        
	def config_manager_functions (self):
		try:
			self.config = {
				'concatenate_vartype_arrays_delete_rows':			cuda.jit('void('+self.VarType+'[:,:], '+self.VarType+'[:,:], int64[:], '+self.VarType+'[:,:])', fastmath = True)(concatenate_arrays_delete_rows_noreturn_cuda),
				'concatenate_outvartype_arrays_delete_rows':		cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], int64[:], '+self.OutVarType+'[:,:])', fastmath = True)(concatenate_arrays_delete_rows_noreturn_cuda),
				'concatenate_vartype_arrays':						cuda.jit('void('+self.VarType+'[:,:], '+self.VarType+'[:,:], '+self.VarType+'[:,:])', fastmath = True)(concatenate_arrays_noreturn_cuda),
				'concatenate_outvartype_arrays':					cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:])', fastmath = True)(concatenate_arrays_noreturn_cuda),
				'concatenate_vartype_matrix_delete_rowsncols':		cuda.jit('void('+self.VarType+'[:,:], '+self.VarType+'[:,:], '+self.VarType+'[:,:], '+self.VarType+'[:,:], int64[:], '
																					+self.VarType+'[:,:])', fastmath = True)(concatenate_matrix_delete_rowsncols_noreturn_cuda),
				'concatenate_outvartype_matrix_delete_rowsncols':	cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], int64[:], '
																					+self.OutVarType+'[:,:])', fastmath = True)(concatenate_matrix_delete_rowsncols_noreturn_cuda),
				'concatenate_vartype_matrix':						cuda.jit('void('+self.VarType+'[:,:], '+self.VarType+'[:,:], '+self.VarType+'[:,:], '+self.VarType+'[:,:], '
																					+self.VarType+'[:,:])', fastmath = True)(concatenate_matrix_noreturn_cuda),
				'concatenate_outvartype_matrix':					cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '
																					+self.OutVarType+'[:,:])', fastmath = True)(concatenate_matrix_noreturn_cuda),
				'add_absorption_object': 							cuda.jit('void(int64[:,:], '+self.VarType+'[:,:,:], '+self.VarType+')', fastmath = True)(add_absorption_object_no_return_cuda),
				'add_geometry_object':					 			cuda.jit('void(int64[:,:], '+self.VarType+'[:,:,:], '+self.VarType+')', fastmath = True)(add_geometry_object_no_return_cuda),
				'add_extended_geometry_nPoints':					cuda.jit('void(int64[:,:], '+self.VarType+'[:,:,:], int64, '+self.VarType+')', fastmath = True)(add_extended_geometry_nPoints_no_return_cuda),
				'PML_limit_volume':					 				cuda.jit('void('+self.VarType+'[:,:,:], int64, '+self.VarType+', '+self.VarType+')', fastmath = True)(PML_limit_volume_no_return_cuda),
				'set_point_as_emitter':								cuda.jit('void('+self.VarType+'[:,:,:], '+self.VarType+'[:,:,:], '+self.VarType+'[:,:,:], int64[:,:], '
																					+self.VarType+'[:], '+self.VarType+'[:], '+self.VarType+'[:])', fastmath = True)(set_point_as_emitter_no_return_cuda)
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.config_manager_functions: {e}')

	def config_manager(self, size=None, blockdim = None, stream = None):
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
			print(f'Error in utils.cuda.manager.manager_cuda.config_manager: {e}')
			
	'''
	Implement functions that initializes the containers of the arrays.
	'''
	
	def add_absorption_object(self, geometry_points, absorption, effect):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert (type(geometry_points) == np.ndarray and type(absorption) == np.ndarray), 'Arrays must be defined as numpy array.'
			
			self.config_manager(size=(geometry_points.shape[0]), blockdim=optimize_blockdim(geometry_points.shape[0]))

			self.config['add_absorption_object'][self.griddim, self.blockdim, self.stream](
				geometry_points, absorption, effect)

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.add_absorption_object: {e}')
			
	def add_geometry_object(self, geometry_points, field, effect):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert (type(geometry_points) == np.ndarray and type(field) == np.ndarray), 'Arrays must be defined as numpy array.'
			
			self.config_manager(size=(geometry_points.shape[0]), blockdim=optimize_blockdim(geometry_points.shape[0]))

			self.config['add_geometry_object'][self.griddim, self.blockdim, self.stream](
				geometry_points, field, effect)

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.add_geometry_object: {e}')
			
	def extend_geometry_nPoints(self, geometry_points, field, distance, effect):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert (type(geometry_points) == np.ndarray and type(field) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert int(distance)>0, 'Distance must be positive.'
			
			self.config_manager(size=(geometry_points.shape[0]), blockdim=optimize_blockdim(geometry_points.shape[0]))

			self.config['extend_geometry_nPoints'][self.griddim, self.blockdim, self.stream](
				geometry_points, field, distance, effect)

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.extend_geometry_nPoints: {e}')
			
	def PML_limit_volume(self, absorption, maxDist, maxValue, minValue):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert (type(absorption) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert int(maxDist)>0, 'Distance must be positive.'
			assert minValue >= 0 and maxValue > minValue, f'Max value {maxValue} and Min value {minValue} are not correctly chosen.'
			
			self.config_manager(size=(absorption.shape[0], absorption.shape[1], absorption.shape[2]), blockdim=optimize_blockdim(absorption.shape[0], absorption.shape[1], absorption.shape[2]))

			self.config['PML_limit_volume'][self.griddim, self.blockdim, self.stream](
				absorption, maxDist, maxValue, minValue)

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.PML_limit_volume: {e}')

	def set_point_as_emitter(self, emitters_amplitude, emitters_frequency, emitters_phase, geometry_points, amplitude, frequency, phase):
		
		try:
			
			assert (type(geometry_points) == np.ndarray and type(emitters_amplitude) == np.ndarray
					and type(emitters_frequency) == np.ndarray and type(emitters_phase) == np.ndarray
					and type(amplitude) == np.ndarray and type(frequency) == np.ndarray and type(phase) == np.ndarray), 'Arrays must be defined as numpy array.'
			
			self.config_manager(size=(geometry_points.shape[0]), blockdim=optimize_blockdim(geometry_points.shape[0]))

			self.config['set_point_as_emitter'][self.griddim, self.blockdim, self.stream](
				emitters_amplitude, emitters_frequency, emitters_phase, geometry_points, amplitude, frequency, phase)

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.set_point_as_emitter: {e}')

	def locate_geometry_object(self, geometry_points, max_distance, step_effect):
		try:
			assert (type(original_pos) == np.ndarray and type(original_norm) == np.ndarray
					and type(final_pos) == np.ndarray and type(final_norm) == np.ndarray), 'Emitters must be defined as numpy array.'
			
			self.transducer_pos_cuda = cuda.to_device(transducer_pos, stream = self.stream)
			self.transducer_norm_cuda = cuda.to_device(transducer_norm, stream = self.stream)
			#All transducers have the same radius. Can be change to consider different transducers.
			self.transducer_radius = transducer_radius
			
		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.locate_transducer: {e}')
			
	def locate_absorption_region(self, geometry_points, effect):
		try:
			assert (type(original_pos) == np.ndarray and type(original_norm) == np.ndarray
					and type(final_pos) == np.ndarray and type(final_norm) == np.ndarray), 'Emitters must be defined as numpy array.'
			
			self.transducer_pos_cuda = cuda.to_device(transducer_pos, stream = self.stream)
			self.transducer_norm_cuda = cuda.to_device(transducer_norm, stream = self.stream)
			#All transducers have the same radius. Can be change to consider different transducers.
			self.transducer_radius = transducer_radius
			
		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.locate_transducer: {e}')

	def locate_transducer(self, geometry_points, points_of_emission, max_distance, step_effect):
		try:
			
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(geometry_points) and cuda.cudadrv.devicearray.is_cuda_ndarray(points_of_emission)), 'Arrays must be loaded in GPU device.'
						
			self.locate_geometry_object(geometry_points, max_distance, step_effect)



		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.locate_transducer: {e}')
	
	'''
	Implement functions that manipulate arrays using cuda
	'''

	def concatenate_arrays(self, old, new, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			
			temporal = cuda.device_array((old.shape[0] + new.shape[0], new.shape[1]), dtype = self.new.dtype, stream = self.stream)

			self.config_manager(size=(old.shape[0] + new.shape[0], old.shape[1]), blockdim=optimize_blockdim(old.shape[0] + new.shape[0], old.shape[1]))

			self.config['concatenate_'+vartype+'_arrays'][self.griddim, self.blockdim, self.stream](old, new, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.concatenate_arrays: {e}')

	def concatenate_arrays_delete_rows(self, old, new, rows_to_remove, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			assert rows_to_remove.shape[0] >= 1, 'No rows specified.'
			
			d_rows_to_remove = cuda.to_device(rows_to_remove, stream = self.stream)
			
			temporal = cuda.device_array((old.shape[0] + new.shape[0] - d_rows_to_remove.shape[0], new.shape[1]), dtype = self.new.dtype, stream = self.stream)

			self.config_manager(size=(old.shape[0] + new.shape[0] - d_rows_to_remove.shape[0], d_rows_to_remove.shape[0]+1), blockdim=optimize_blockdim(old.shape[0] + new.shape[0] - d_rows_to_remove.shape[0], d_rows_to_remove.shape[0]+1))

			self.config['concatenate_'+vartype+'_arrays_delete_rows'][self.griddim, self.blockdim, self.stream](old, new, d_rows_to_remove, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.concatenate_arrays_delete_rows: {e}')
	
	def concatenate_matrix(self, old_old, old_new, new_old, new_new, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			
			temporal = cuda.device_array((old_old.shape[0] + new_new.shape[0] ,
								 old_old.shape[1] + new_new.shape[1]),
								dtype = self.old_old.dtype, stream = self.stream)

			self.config_manager(size=(temporal.shape[0], temporal.shape[1]), blockdim=optimize_blockdim(temporal.shape[0], temporal.shape[1]))

			self.config['concatenate_'+vartype+'_matrix'][self.griddim, self.blockdim, self.stream](
				old_old, old_new, new_old, new_new, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.concatenate_matrix: {e}')
		
	def concatenate_matrix_delete_rowsncols(self, old_old, old_new, new_old, new_new, delete_rowsncols, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			assert delete_rowsncols.shape[0] >= 1, 'No rows nor cols specified.'
			
			d_rowsncols_to_remove = cuda.to_device(delete_rowsncols, stream = self.stream)
			
			temporal = cuda.device_array((old_old.shape[0] + new_new.shape[0] - d_rowsncols_to_remove.shape[0],
								 old_old.shape[1] + new_new.shape[1] - d_rowsncols_to_remove.shape[0]),
								dtype = self.old_old.dtype, stream = self.stream)

			self.config_manager(size=(temporal.shape[0], temporal.shape[1]), blockdim=optimize_blockdim(temporal.shape[0], temporal.shape[1]))

			self.config['concatenate_'+vartype+'_matrix_delete_rowsncols'][self.griddim, self.blockdim, self.stream](
				old_old, old_new, new_old, new_new, d_rowsncols_to_remove, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cuda.manager.manager_cuda.concatenate_matrix_delete_rowsncols: {e}')
		
	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.manually_fill_mesh: {e}')