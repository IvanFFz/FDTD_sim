import numpy as np #, open3d as o3d #, pandas as pd

#import os #, numba, math, cmath, itertools

from numpy import int64
from numba import cuda #, int64 #, float32, float64, void
from .calculator import optimize_blockdim

def generate_wall_noreturn_cuda(grid, location, normal, count, grid_min):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i<grid.shape[0] and j<grid.shape[1] and k<grid.shape[2]:
		
		if (location[0] - grid_min - i) * normal[0] + (location[1] - grid_min - j) * normal[1] + (location[2] - grid_min - k) * normal[2] >= 0:
			grid[i, j, k] = 1
			
			cuda.atomic.add(count, 0, 1)
			
def results_to_list_noreturn_cuda(grid, geometry_points, grid_min, count):
		
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < grid.shape[0] and j < grid.shape[1] and k < grid.shape[2] and count[0] < geometry_points.shape[0]:
		if grid[i, j, k] == 1:
			
			idx = cuda.atomic.add(count, 0, 1)

			geometry_points[idx, 0] = i 
			geometry_points[idx, 1] = j
			geometry_points[idx, 2] = k 
			



class simple_objects_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, discretization, grid_limits, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		#assert os.path.exists(path), f'Path {path} not valid.'
		#assert os.path.exists(os.path.join(path, 'model' + file_extension)), f'No model file with extension {file_extension} in {path}.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.stream = stream
		
		self.config = None
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
		
		#self.path = path
		#self.model_file = name + file_extension
		#self.unit_dict = unit_dict
		self.discretization = np.array([discretization]).astype(self.VarType)[0]
		print(self.discretization)
		self.grid_limits = grid_limits.astype(self.VarType) / (self.discretization)
		self.grid_limits = self.grid_limits.astype(int64) #np.array([grid_limits, grid_limits, grid_limits]) #Change to enable different positions
		print(self.grid_limits)
		#self.mesh_volume = None
		#self.mesh_center = mesh_center
		self.results_grid = None
		
		self.auxiliar = {
			
			'grid_volume': None
			
			}
		
		self.location = None
		self.normal = None

		self.config_simple_objects(size, blockdim, stream)

		self.config_simple_objects_functions()
		
		
	'''
	Implement configurations
	''' 

	def config_simple_objects_functions (self):
		try:
			self.config = {
				'generate_wall':					cuda.jit('void(int64[:,:,:], int64[:],  '+self.VarType+'[:], int64[:], int64)', fastmath = True)(generate_wall_noreturn_cuda),
				'results_to_list':					cuda.jit('void(int64[:,:,:], int64[:,:], int64, int64[:])', fastmath = True)(results_to_list_noreturn_cuda)											
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.simple_objects.simple_objects_cuda.config_simple_objects_functions: {e}')

	def config_simple_objects(self, size=None, blockdim = None, stream = None):
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
			print(f'Error in utils.cuda.simple_objects.simple_objects_cuda.config_simple_objects: {e}')
			
	def generate_wall(self, dict_unit):
		try:
			
			aux_location = np.array(dict_unit['location']).astype(self.VarType) / self.discretization
			aux_location = aux_location.astype(int64)
			aux_normal = np.array(dict_unit['orientation']).astype(self.VarType)
			
			assert aux_normal[0] != 0 or aux_normal[1] != 0 or aux_normal[2] != 0, 'Invalid normal specified for wall.'
			assert (aux_location[0] > self.grid_limits[0] and aux_location[0] < self.grid_limits[1] and
					aux_location[1] > self.grid_limits[0] and aux_location[1] < self.grid_limits[1] and
					aux_location[2] > self.grid_limits[0] and aux_location[2] < self.grid_limits[1]), 'The wall is not inside the simulation volume'

			self.auxiliar['grid_volume'] = cuda.to_device(np.zeros((self.grid_limits[1] - self.grid_limits[0] + 1, self.grid_limits[1] - self.grid_limits[0] + 1,
																		self.grid_limits[1] - self.grid_limits[0] + 1), dtype = np.int64))
			
			count = np.array([0]).astype(int64)
			
			self.config_simple_objects(size = (self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]))
			
			self.config['generate_wall'][self.griddim, self.blockdim, self.stream](self.auxiliar['grid_volume'], aux_location, aux_normal, count, self.grid_limits[0])
			print('generated')
			geometry_points = cuda.to_device(np.zeros((count[0], 3), dtype = int64), stream = self.stream)
			count = np.array([0]).astype(int64)			

			self.config['results_to_list'][self.griddim, self.blockdim, self.stream](self.auxiliar['grid_volume'], geometry_points, self.grid_limits[0], count)
			print('translated')
			return geometry_points

		except Exception as e:
			print(f'Error in utils.cuda.simple_objects.simple_objects_cuda.generate_wall: {e}')
			