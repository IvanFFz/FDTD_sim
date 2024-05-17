#from tkinter import W, Scale
#from matplotlib.pyplot import grid
import numpy as np, open3d as o3d #, pandas as pd

import os #, numba, math, cmath, itertools

from numpy import int64
from numba import cuda #, int64 #, float32, float64, void
from .calculator import optimize_blockdim


###########################################################################################
# Add function to increase (maybe decrease) the number of triangles that defines the mesh #
###########################################################################################


def translate_noreturn_cuda (points, translation_vector):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if i<points.shape[0] and j<points.shape[1]:
		cuda.atomic.add(points, (i, j), translation_vector[j])
		
def rotate_noreturn_cuda (points, rotation_matrix, center, rotated_points):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if i<points.shape[0] and j<points.shape[1]:
		rotated_points[i, j] = center[j] + ( points[i, 0] - center[0] ) * rotation_matrix[0, j] + ( points[i, 1] - center[1] ) * rotation_matrix[1, j] + ( points[i, 2] - center[2] ) * rotation_matrix[2, j]
		
def scale_noreturn_cuda (points, scale_factor, center):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if i<points.shape[0] and j<points.shape[1]:
		points[i, j] = points[i, j] * scale_factor
		#cuda.syncthreads()
		cuda.atomic.add(points, (i, j), (1.0 - scale_factor)*center[j])
		
def mesh_to_grid_noreturn_cuda (points, discretization, int_points):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if i < points.shape[0] and j < points.shape[1]:
		int_points[i, j] = round(points[i, j] / discretization)
		
def check_int_ext_noreturn_cuda (surface_points, grid_volume, axis, reversed_path, min_limit):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if axis == 0 and i < grid_volume.shape[1] and j < grid_volume.shape[2]:
		
		for k in range(int(abs(min(1, reversed_path*grid_volume.shape[0] + 2))), int(max(reversed_path*grid_volume.shape[0], -1)), int(reversed_path)):
			for point in range(surface_points.shape[0]):
				
				if surface_points[point, 0] == min_limit[0] + k and surface_points[point, 1] == min_limit[1] + i and surface_points[point, 2] == min_limit[2] + j:
					
					if grid_volume[int(k-reversed_path), i, j] == 0 or grid_volume[int(k-reversed_path), i, j] == 2:
						grid_volume[k, i, j] = 2
						#cuda.atomic.compare_and_swap(grid_volume[k:, i, j], int(0), int(1))
						
					elif grid_volume[k-reversed_path, i, j] == -1 or grid_volume[int(k-reversed_path), i, j] == 1:
						grid_volume[k, i, j] = -1
						#cuda.atomic.compare_and_swap(grid_volume[k:, i, j], int(0), int(-1))
						
					break
				
			if grid_volume[int(k-reversed_path), i, j] != 0 and grid_volume[k, i, j] == 0:
				if grid_volume[int(k-reversed_path), i, j] == 2:
					grid_volume[k, i, j] = 1
				elif grid_volume[int(k-reversed_path), i, j] == -1:
					grid_volume[k, i, j] = 0
				elif grid_volume[int(k-reversed_path), i, j] == 1:
					grid_volume[k, i, j] = 1
				
	elif axis == 1 and i < grid_volume.shape[0] and j < grid_volume.shape[2]:
		
		for k in range(int(abs(min(1, reversed_path*grid_volume.shape[1] + 2))), int(abs(max(reversed_path*grid_volume.shape[1], -1))), int(reversed_path)):
			for point in range(surface_points.shape[0]):

				if surface_points[point, 0] == min_limit[0] + i and surface_points[point, 1] == min_limit[1] + k and surface_points[point, 2] == min_limit[2] + j:
					
					if grid_volume[i, int(k-reversed_path), j] == 0 or grid_volume[i, int(k-reversed_path), j] == 2:
						grid_volume[i, k, j] = 2
						#cuda.atomic.compare_and_swap(grid_volume[i, k:, j], int(0), int(1))
						
					elif grid_volume[i, int(k-reversed_path), j] == -1 or grid_volume[i, int(k-reversed_path), j] == 1:
						grid_volume[i, k, j] = -1
						#cuda.atomic.compare_and_swap(grid_volume[i, k:, j], int(0), int(-1))
						
					break
			
			if grid_volume[i, int(k-reversed_path), j] != 0 and grid_volume[i, k, j] == 0:
				if grid_volume[i, int(k-reversed_path), j] == 2:
					grid_volume[i, k, j] = 1
				elif grid_volume[i, int(k-reversed_path), j] == -1:
					grid_volume[i, k, j] = 0
				elif grid_volume[i, int(k-reversed_path), j] == 1:
					grid_volume[i, k, j] = 1

	elif axis == 2 and i < grid_volume.shape[0] and j < grid_volume.shape[1]:
		
		for k in range(int(abs(min(1, reversed_path*grid_volume.shape[2] + 2))), int(abs(max(reversed_path*grid_volume.shape[2], -1))), int(reversed_path)):
			for point in range(surface_points.shape[0]):
				
				if surface_points[point, 0] == min_limit[0] + i and surface_points[point, 1] == min_limit[1] + j and surface_points[point, 2] == min_limit[2] + k:
										
					if grid_volume[i, j, int(k-reversed_path)] == 0 or grid_volume[i, j, int(k-reversed_path)] == 2:
						grid_volume[i, j, k] = 2
						#cuda.atomic.compare_and_swap(grid_volume[i, j, k:], int(0), int(1))
						
					elif grid_volume[i, j, int(k-reversed_path)] == -1 or grid_volume[i, j, int(k-reversed_path)] == 1:
						grid_volume[i, j, k] = -1
						#cuda.atomic.compare_and_swap(grid_volume[i, j, k:], int(0), int(-1))
						
					break
				
			if grid_volume[i, j, int(k-reversed_path)] != 0 and grid_volume[i, j, k] == 0:
				if grid_volume[i, j, int(k-reversed_path)] == 2:
					grid_volume[i, j, k] = 1
				elif grid_volume[i, j, int(k-reversed_path)] == -1:
					grid_volume[i, j, k] = 0
				elif grid_volume[i, j, int(k-reversed_path)] == 1:
					grid_volume[i, j, k] = 1

def check_surface_noreturn_cuda (surface_points, grid_volume, min_limit):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < grid_volume.shape[0] and j < grid_volume.shape[1] and k < grid_volume.shape[2]:
		for point in range(surface_points.shape[0]):
				
			if surface_points[point, 0] == min_limit[0] + i and surface_points[point, 1] == min_limit[1] + j and surface_points[point, 2] == min_limit[2] + k:
					
				grid_volume[i, j, k] = 1
					#cuda.atomic.compare_and_swap(grid_volume[k:, i, j], int(0), int(-1))
						
				break
										
def add_check_int_ext_results_noreturn_cuda (results_grid, grid_volume):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2]:
		
		if grid_volume[i, j, k] != 0:
			cuda.atomic.add(results_grid, (i, j, k), int(1) )
		
			grid_volume[i, j, k] = 0
		
def compute_results_noreturn_cuda (results_grid, threshold):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2]:
		
		if results_grid[i, j, k] >= threshold:
			results_grid[i, j, k] = 1
		else:
			results_grid[i, j, k] = 0
			
def count_points_result_noreturn_cuda (results_grid, grid_min, grid_max, mesh_min, mesh_max, count):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2]:
		if (	i + mesh_min[0] >= grid_min and j + mesh_min[1] >= grid_min and k + mesh_min[2] >= grid_min 
			and i + mesh_max[0] <= grid_max and j + mesh_max[1] <= grid_max and k + mesh_max[2] <= grid_max 
			and results_grid[i, j, k] == 1):
			
			cuda.atomic.add(count, 0, 1)
			
		else:
			
			results_grid[i, j, k] = 0
			
def results_to_list_noreturn_cuda (results_grid, geometry_points, grid_min, mesh_min, count):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	#points = cuda.shared.array(shape=(8, 4, 4), dtype = int64)
	#if i == 0 and j == 0 and k == 0:
	#	shared_position = cuda.shared.array(shape=(1), dtype = int64)
	#	shared_position[0] = 0
	#	
	#cuda.syncthreads()
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2] and count[0] < geometry_points.shape[0]:
		if results_grid[i, j, k] == 1:
			
			idx = cuda.atomic.add(count, 0, 1)

			geometry_points[idx, 0] = i - grid_min + mesh_min[0]
			geometry_points[idx, 1] = j - grid_min + mesh_min[1]
			geometry_points[idx, 2] = k - grid_min + mesh_min[2]
			
		#else:
			#points[cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z] = 0
			
def surface_to_result_noreturn_cuda (surface_points, geometry_points, grid_min, mesh_min):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	#points = cuda.shared.array(shape=(8, 4, 4), dtype = int64)
	#if i == 0 and j == 0 and k == 0:
	#	shared_position = cuda.shared.array(shape=(1), dtype = int64)
	#	shared_position[0] = 0
	#	
	#cuda.syncthreads()
	
	if i < surface_points.shape[0]:
		
		geometry_points[i, 0] = surface_points[i,0] - grid_min + mesh_min[0]
		geometry_points[i, 1] = surface_points[i,1] - grid_min + mesh_min[1]
		geometry_points[i, 2] = surface_points[i,2] - grid_min + mesh_min[2]
		

class stlReader_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, path, name, unit_dict, discretization, grid_limits, mesh_center = None, file_extension = '.stl', stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		assert os.path.exists(path), f'Path {path} not valid.'
		assert os.path.exists(os.path.join(path, 'model' + file_extension)), f'No model file with extension {file_extension} in {path}.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.stream = stream
		
		self.config = None
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
		
		self.path = path
		self.model_file = name + file_extension
		self.unit_dict = unit_dict
		self.discretization = np.array([discretization / 1e-3]).astype(self.VarType)[0]
		print(self.discretization)
		self.grid_limits = grid_limits.astype(self.VarType) / (1e-3 * self.discretization)
		self.grid_limits = self.grid_limits.astype(int64) #np.array([grid_limits, grid_limits, grid_limits]) #Change to enable different positions
		print(self.grid_limits)
		self.mesh_volume = None
		self.mesh_center = mesh_center
		self.results_grid = None
		
		self.auxiliar = {
			
			'rotate': None,
			'int_mesh_grid': None,
			'mesh_limits': {
				'max': None,
				'min': None
				},
			'grid_volume': None
			
			}
		
		self.normal = None

		self.config_stlReader(size, blockdim, stream)

		self.config_stlReader_functions()
		
		
	'''
	Implement configurations
	''' 

	def config_stlReader_functions (self):
		try:
			self.config = {
				'translate':						cuda.jit('void('+self.VarType+'[:,:], ' + self.VarType+'[:])', fastmath = True)(translate_noreturn_cuda),
				'rotate':							cuda.jit('void('+self.VarType+'[:,:], ' + self.VarType+'[:,:], ' + self.VarType+'[:], ' + self.VarType+'[:,:])', fastmath = True)(rotate_noreturn_cuda),
				'scale':							cuda.jit('void('+self.VarType+'[:,:], ' + self.VarType+', ' + self.VarType+'[:])', fastmath = True)(scale_noreturn_cuda),
				'mesh_to_grid':						cuda.jit('void('+self.VarType+'[:,:], ' +self.VarType+', ' + 'int64[:,:])', fastmath = True)(mesh_to_grid_noreturn_cuda),
				'check_int_ext':					cuda.jit('void(int64[:,:], int64[:,:,:], int64, int64, int64[:])', fastmath = True)(check_int_ext_noreturn_cuda),
				'check_surface':					cuda.jit('void(int64[:,:], int64[:,:,:], int64[:])', fastmath = True)(check_surface_noreturn_cuda),
				'add_check_int_ext_results':		cuda.jit('void(int64[:,:,:], int64[:,:,:])', fastmath = True)(add_check_int_ext_results_noreturn_cuda),
				'compute_results':					cuda.jit('void(int64[:,:,:], int64)', fastmath = True)(compute_results_noreturn_cuda),
				'count_points_result':				cuda.jit('void(int64[:,:,:], int64, int64, int64[:], int64[:], int64[:])', fastmath = True)(count_points_result_noreturn_cuda),
				'results_to_list':					cuda.jit('void(int64[:,:,:], int64[:,:], int64, int64[:], int64[:])', fastmath = True)(results_to_list_noreturn_cuda),
				'surface_to_result':				cuda.jit('void(int64[:,:], int64[:,:], int64, int64[:])', fastmath = True)(surface_to_result_noreturn_cuda)
																	
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.config_stlReader_functions: {e}')

	def config_stlReader(self, size=None, blockdim = None, stream = None):
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
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.config_stlReader: {e}')
			
	def read_file(self):
		try:
			
			self.mesh_volume = o3d.io.read_triangle_mesh(os.path.join(self.path, self.model_file), print_progress = True)
			
			#self.mesh_volume = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.mesh_volume, self.unit_dict['triangle_multiplier'])

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.read_file: {e}')
			
	def get_surface_points(self):
		try:
			
			triangle_count = len(self.mesh_volume.triangles)
			cloud = self.mesh_volume.sample_points_uniformly(number_of_points = int(triangle_count*self.unit_dict['triangle_multiplier']) )
			cloud = cloud.remove_duplicated_points()
			self.get_mesh_center(cloud)
			print(self.mesh_center)
			cloud = np.asarray(cloud.points).astype(self.VarType)
			#with np.printoptions(threshold=np.inf):
			#	print(cloud)
			self.auxiliar['mesh_limits'] = {
				'max' : np.array([round(np.max(cloud) / self.discretization ) + 11]).astype(self.VarType),
				'min' : np.array([round(np.min(cloud) / self.discretization ) - 10]).astype(self.VarType)
				}
			#print(cloud.shape)
			self.mesh_volume = cuda.to_device(np.unique(cloud, axis=0), stream = self.stream)
			
			#print(self.mesh_volume.shape)
			#print('\n')
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			#print('\n')
			self.erase_variable(cloud)
		
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_surface_points: {e}')
			
	def locate_mesh(self):
		try:
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			assert self.mesh_center is not None, 'Mesh center not calculated, make sure you are executing the functions in the correct order.'
			
			self.scale()
			self.stream.synchronize()
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			
			self.rotate()
			self.stream.synchronize()
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			
			self.translate()
			self.stream.synchronize()
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.locate_mesh: {e}')
			
	def mesh_to_grid(self):
		try:
			
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			assert self.mesh_center is not None, 'Mesh center not calculated, make sure you are executing the functions in the correct order.'
			
			self.auxiliar['int_mesh_grid'] = cuda.device_array((self.mesh_volume.shape[0], self.mesh_volume.shape[1]), dtype = int64, stream = self.stream)
			
			self.config_stlReader(size = (self.mesh_volume.shape[0], self.mesh_volume.shape[1]),
								blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))

			self.config['mesh_to_grid'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.discretization, self.auxiliar['int_mesh_grid'])
			
			self.mesh_volume = self.auxiliar['int_mesh_grid']
			
			self.erase_variable(self.auxiliar['int_mesh_grid'])
			print(self.mesh_volume.shape)
			#with np.printoptions(threshold=np.inf):
			#	
			#	print('mesh_points_grid', self.mesh_volume.copy_to_host())
			#	print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])

			self.auxiliar['mesh_limits']['min'] = np.floor(self.auxiliar['mesh_limits']['min']).astype(int64)
			self.auxiliar['mesh_limits']['max'] = np.ceil(self.auxiliar['mesh_limits']['max']).astype(int64)
			
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
	
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.mesh_to_grid: {e}')
			
	def check_int_ext(self, axis, reversed_path = False):
		try:
			
			assert int(axis) in [0, 1, 2], f'{axis} axis is not valid.'
			
			#print(type(self.auxiliar['mesh_limits']['max']), type(self.auxiliar['mesh_limits']['min']))
			#print(self.auxiliar['mesh_limits']['max'], self.auxiliar['mesh_limits']['min'])

			if self.auxiliar['grid_volume'] is None:
				self.auxiliar['grid_volume'] = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'][0] - self.auxiliar['mesh_limits']['min'][0] + 1, self.auxiliar['mesh_limits']['max'][1] - self.auxiliar['mesh_limits']['min'][1] + 1,
																		self.auxiliar['mesh_limits']['max'][2] - self.auxiliar['mesh_limits']['min'][2] + 1), dtype = np.int64))
			
			if self.results_grid is None:	
				self.results_grid = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'][0] - self.auxiliar['mesh_limits']['min'][0] + 1, self.auxiliar['mesh_limits']['max'][1] - self.auxiliar['mesh_limits']['min'][1] + 1,
															 self.auxiliar['mesh_limits']['max'][2] - self.auxiliar['mesh_limits']['min'][2] + 1), dtype = np.int64))
			
			
			if reversed_path:
				reversed_path = int(-1)
				
			else:
				reversed_path = int(1)
				
			if int(axis) == 0:
				
				self.config_stlReader(size = (self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]))
				
			elif int(axis) == 1:
				
				self.config_stlReader(size = (self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[2]))
				
			elif int(axis) == 2:
				
				self.config_stlReader(size = (self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1]))
				
			#print(self.mesh_volume.dtype, self.auxiliar['grid_volume'].dtype, reversed_path.dtype, self.auxiliar['mesh_limits']['min'].dtype)
			self.config['check_int_ext'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.auxiliar['grid_volume'], int(axis), reversed_path, self.auxiliar['mesh_limits']['min'])
			
			#with np.printoptions(threshold=np.inf):
			#	print(self.auxiliar['grid_volume'].copy_to_host())

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.check_int_ext: {e}')
			
	def check_surface(self):
		try:
			
			#assert int(axis) in [0, 1, 2], f'{axis} axis is not valid.'
			
			#print(type(self.auxiliar['mesh_limits']['max']), type(self.auxiliar['mesh_limits']['min']))
			#print(self.auxiliar['mesh_limits']['max'], self.auxiliar['mesh_limits']['min'])

			if self.auxiliar['grid_volume'] is None:
				self.auxiliar['grid_volume'] = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'][0] - self.auxiliar['mesh_limits']['min'][0] + 1, self.auxiliar['mesh_limits']['max'][1] - self.auxiliar['mesh_limits']['min'][1] + 1,
																		self.auxiliar['mesh_limits']['max'][2] - self.auxiliar['mesh_limits']['min'][2] + 1), dtype = np.int64))
			
			if self.results_grid is None:	
				self.results_grid = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'][0] - self.auxiliar['mesh_limits']['min'][0] + 1, self.auxiliar['mesh_limits']['max'][1] - self.auxiliar['mesh_limits']['min'][1] + 1,
															 self.auxiliar['mesh_limits']['max'][2] - self.auxiliar['mesh_limits']['min'][2] + 1), dtype = np.int64))
				
			self.config_stlReader(size = (self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]),
								blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]))
				
			#print(self.mesh_volume.dtype, self.auxiliar['grid_volume'].dtype, reversed_path.dtype, self.auxiliar['mesh_limits']['min'].dtype)
			self.config['check_surface'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.auxiliar['grid_volume'], self.auxiliar['mesh_limits']['min'])
			
			#with np.printoptions(threshold=np.inf):
			#	print(self.auxiliar['grid_volume'].copy_to_host())

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.check_surface: {e}')
			
	def add_check_int_ext_results(self):
		try:
			
			self.config_stlReader(size = (self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.auxiliar['grid_volume'].shape[0], self.auxiliar['grid_volume'].shape[1], self.auxiliar['grid_volume'].shape[2]))
			

			self.config['add_check_int_ext_results'][self.griddim, self.blockdim, self.stream](self.results_grid, self.auxiliar['grid_volume'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.add_check_int_ext_results: {e}')
			
	def compute_results (self, threshold = 5):
		try:
			
			self.config_stlReader(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			self.config['compute_results'][self.griddim, self.blockdim, self.stream](self.results_grid, threshold)
			
			#with np.printoptions(threshold=np.inf):
			#	print('computed', self.results_grid.shape, self.results_grid.copy_to_host())
			count = np.array([0]).astype(int64)
			
			self.config_stlReader(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			self.config['count_points_result'][self.griddim, self.blockdim, self.stream](self.results_grid, self.grid_limits[0], self.grid_limits[1],
																						self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'], count)
			
			#print('counted', count)
			#print(self.auxiliar['mesh_limits']['min'])
			geometry_points = cuda.to_device(np.zeros((count[0], 3), dtype = int64), stream = self.stream)
			
			self.config_stlReader(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			#print(self.blockdim, self.griddim)

			count = np.array([0]).astype(int64)

			self.config['results_to_list'][self.griddim, self.blockdim, self.stream](self.results_grid, geometry_points, self.grid_limits[0], self.auxiliar['mesh_limits']['min'], count)
			#with np.printoptions(threshold=np.inf):
			#	print('points', geometry_points.shape, geometry_points.copy_to_host())
			return geometry_points

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.compute_results: {e}')
		
	#def surface_to_result(self):
	#	try:
	#		
	#		surface_points = cuda.to_device(np.zeros((self.mesh_volume.shape[0], 3), dtype = int64), stream = self.stream)
	#		
	#		self.config_stlReader(size = self.mesh_volume.shape[0],
	#								blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0]))
	#
	#		self.config['surface_to_result'](self.mesh_volume, surface_points, self.grid_limits[0], self.auxiliar['mesh_limits']['min'])
	#		
	#		return surface_points
	#
	#	except Exception as e:
	#		print(f'Error in utils.cuda.stlReader.stlReader_cuda.surface_to_result: {e}')

			
	def extract_object(self, threshold = 4, emission = False):
		try:
			self.read_file()
			
			self.get_surface_points()
			
			self.locate_mesh()
						
			self.mesh_to_grid()
			
			if not emission:
				
				for axis in range(3):
					self.check_int_ext(axis)
					self.stream.synchronize()
					self.add_check_int_ext_results()
				
					self.stream.synchronize()
				
					self.check_int_ext(axis, reversed_path=True)
					self.stream.synchronize()
					self.add_check_int_ext_results()
				
					self.stream.synchronize()
										
			elif emission:

				self.check_surface()
				self.stream.synchronize()
				self.add_check_int_ext_results()
				
				self.stream.synchronize()
				
				
			#with np.printoptions(threshold=np.inf):
			#	print(self.results_grid.copy_to_host())
			
			return (self.compute_results(threshold), self.normal)

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.extract_object: {e}')



	def get_rotation_matrix (self, where_to_point):
		try:
			
			if isinstance(self.unit_dict['normal'], list):
				self.unit_dict['normal'] = np.array(self.unit_dict['normal']).reshape((3, 1))
			if self.unit_dict['normal'].shape != (3,1):
				self.unit_dict['normal'] = self.unit_dict['normal'].squeeze().reshape((3, 1))
				
			if isinstance(self.mesh_center, list):
				mesh_center = np.array(self.mesh_center).reshape((3, 1))
			if self.mesh_center.shape != (3,1):
				mesh_center = self.mesh_center.squeeze().reshape((3, 1))
				
			if isinstance(where_to_point, list):
				where_to_point = np.array(where_to_point).reshape((3, 1))
			if where_to_point.shape != (3,1):
				where_to_point = where_to_point.squeeze().reshape((3, 1))
				
			assert ( mesh_center[0, 0] != where_to_point[0, 0] and mesh_center[1, 0] != where_to_point[1, 0]
					and mesh_center[2, 0] != where_to_point[2, 0] ), 'Something may go wrong with an object pointing to its center.'
			assert np.linalg.norm(self.unit_dict['normal']) != 0, f'Mesh normal has 0 norm.'
				
			final_normal = (where_to_point - mesh_center) / np.linalg.norm(where_to_point - mesh_center)
			#print(mesh_center, where_to_point, final_normal)
			
			if np.linalg.norm(self.unit_dict['normal']) != 1:
				self.unit_dict['normal'] = self.unit_dict['normal'] / np.linalg.norm(self.unit_dict['normal'])
			
			self.rotation_matrix = np.identity(3) + final_normal.dot(self.unit_dict['normal'].T) - self.unit_dict['normal'].dot(final_normal.T) + np.linalg.matrix_power(final_normal.dot(self.unit_dict['normal'].T) - self.unit_dict['normal'].dot(final_normal.T), 2) / ( 1 + self.unit_dict['normal'].squeeze().dot(final_normal.squeeze()))

			self.rotation_matrix = self.rotation_matrix.astype(self.VarType)
			#print(np.matmul(self.rotation_matrix, self.unit_dict['normal']).astype(self.VarType))

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_rotation_matrix: {e}')
				
	def get_mesh_center (self, cloud):
		try:
			
			if self.mesh_center is None:
				self.mesh_center = cloud.get_center().astype(self.VarType)

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_mesh_center: {e}')
			
	def rotate (self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			
			self.get_rotation_matrix(np.array(self.unit_dict['orientation']) / ( 1e-3 * self.discretization))
			#print(self.rotation_matrix)
			self.auxiliar['rotate'] = cuda.to_device(np.zeros((self.mesh_volume.shape[0], self.mesh_volume.shape[1]), dtype = self.mesh_volume.dtype), stream = self.stream)
			
			self.config_stlReader(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))

			self.config['rotate'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.rotation_matrix, self.mesh_center, self.auxiliar['rotate'])
			
			self.mesh_volume = self.auxiliar['rotate']
			
			self.normal = np.matmul(self.rotation_matrix, self.unit_dict['normal']).squeeze().astype(self.VarType)
			
			self.erase_variable(self.auxiliar['rotate'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.rotate: {e}')
			
	def translate(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
						
			self.unit_dict['location'] = np.array(self.unit_dict['location'])

			self.config_stlReader(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))

			self.config['translate'][self.griddim, self.blockdim, self.stream](self.mesh_volume, (self.unit_dict['location'] / 1e-3).astype(self.VarType))
			
			self.auxiliar['mesh_limits']['min'] = self.auxiliar['mesh_limits']['min'] + (self.unit_dict['location']  / (1e-3 * self.discretization)).astype(self.VarType)
			self.auxiliar['mesh_limits']['max'] = self.auxiliar['mesh_limits']['max'] + (self.unit_dict['location']  / (1e-3 * self.discretization)).astype(self.VarType)
			
			#print(self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.translate: {e}')

	def scale(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			
			if self.unit_dict['scale'] != 1.0:
			
				self.config_stlReader(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))
			
				self.config['scale'][self.griddim, self.blockdim, self.stream](self.mesh_volume, np.array([self.unit_dict['scale']]).astype(self.VarType)[0], self.mesh_center)
			
				self.auxiliar['mesh_limits']['min'] = self.auxiliar['mesh_limits']['min'] * np.array([self.unit_dict['scale']]).astype(self.VarType)[0] + (1.0 - np.array([self.unit_dict['scale']]).astype(self.VarType)[0] ) * self.mesh_center
				self.auxiliar['mesh_limits']['max'] = self.auxiliar['mesh_limits']['max'] * np.array([self.unit_dict['scale']]).astype(self.VarType)[0] + (1.0 - np.array([self.unit_dict['scale']]).astype(self.VarType)[0] ) * self.mesh_center

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.scale: {e}')


	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cpu.stlReader.stlReader_cuda.erase_variable: {e}')
