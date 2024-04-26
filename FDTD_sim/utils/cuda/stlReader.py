from tkinter import W, Scale
import numpy as np, pandas as pd, open3d as o3d

import numba, math, cmath, os, itertools

from stl import mesh
from numba import float32, float64, void, cuda
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
		cuda.atomic.add(points, (i, j), (1.0 - scale_factor)*center[j])
		
def mesh_to_grid_noreturn_cuda (points, discretization):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < points.shape[0] and j < points.shape[1] and k < points.shape[2]:
		points[i, j, k] = round(points[i, j, k] / discretization)
		
def check_int_ext_noreturn_cuda (points, grid_volume, axis, reversed_path, min_limit):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
	if axis == 0 and i < grid_volume.shape[1] and j < grid_volume.shape[2]:
		
		for k in range(1, grid_volume.shape[0], reversed_path):
			for point in points:
				
				if int(point[0]) == min_limit + k and int(point[1]) == min_limit + i and int(point[2]) == min_limit + j:
					
					cuda.atomic.add(grid_volume, (k, i, j), 1)
					
				else:
					
					cuda.atomic.add(grid_volume, (k, i, j), grid_volume[k-reversed_path, i, j] - grid_volume[k, i, j])
	
	if axis == 1 and i < grid_volume.shape[0] and j < grid_volume.shape[2]:
		
		for k in range(1, grid_volume.shape[1], reversed_path):
			for point in points:

				if int(point[0]) == min_limit + i and int(point[1]) == min_limit + k and int(point[2]) == min_limit + j:
					
					cuda.atomic.add(grid_volume, (i, k, j), 1)
					
				else:
					
					cuda.atomic.add(grid_volume, (i, k, j), grid_volume[i, k-reversed_path, j] - grid_volume[i, k, j])

	if axis == 2 and i < grid_volume.shape[0] and j < grid_volume.shape[1]:
		
		for k in range(1, grid_volume.shape[2], reversed_path):
			for point in points:
				
				if int(point[0]) == min_limit + i and int(point[1]) == min_limit + j and int(point[2]) == min_limit + k:
					
					cuda.atomic.add(grid_volume, (i, j, k), 1)
					
				elif int(point[0]) == min_limit + i - reversed_path and int(point[1]) == min_limit + j - reversed_path and int(point[2]) == min_limit + k - reversed_path:
					
					cuda.atomic.add(grid_volume, (i, j, k), grid_volume[i, j, k-reversed_path] - grid_volume[i, j, k - 2*reversed_path])
					
				else:
					
					cuda.atomic.add(grid_volume, (i, j, k), grid_volume[i, j, k-reversed_path])
					
def add_check_int_ext_results_noreturn_cuda (results_grid, grid_volume):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2]:
		
		cuda.atomic.add(results_grid, (i, j, k), grid_volume[i, j, k])
		
		cuda.atomic.compare_and_swap(grid_volume[i, j, k], 1, 0)
		
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
		if min(i, j, k) + mesh_min >= grid_min and max(i, j, k) + mesh_max <= grid_max:
			cuda.atomic.add(count, 0, 1)
			cuda.atomic.compare_and_swap(results_grid[i, j, k], 1, count-1)
		else:
			results_grid[i, j, k] = -1
			
def results_to_list_noreturn_cuda (results_grid, geometry_points, mesh_min):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	
	if i < results_grid.shape[0] and j < results_grid.shape[1] and k < results_grid.shape[2]:
		if results_grid[i, j, k] != -1:
			geometry_points[results_grid[i, j, k], 0] = mesh_min + i
			geometry_points[results_grid[i, j, k], 1] = mesh_min + j
			geometry_points[results_grid[i, j, k], 2] = mesh_min + k
		
		

class stlReader_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, path, unit_dict, discretization, grid_limits, mesh_center = None, file_extension = '.stl', stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

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
		self.model_file = 'model' + file_extension
		self.unit_dict = unit_dict
		self.discretization = discretization
		self.grid_limits = grid_limits
		self.mesh_volume = None
		self.mesh_center = mesh_center
		self.results_grid = None
		
		self.auxiliar = {
			
			'rotate': None,
			'boolean_grid': None,
			'mesh_limits': {
				'max': None,
				'min': None
				},
			'grid_volume': None
			
			}

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
				'mesh_to_grid':						cuda.jit('void('+self.VarType+'[:,:], ' + self.VarType+')', fastmath = True)(mesh_to_grid_noreturn_cuda),
				'check_int_ext':					cuda.jit('void('+self.VarType+'[:,:], ' + 'int64[:,:], int64, int64, int64)', fastmath = True)(check_int_ext_noreturn_cuda),
				'add_check_int_ext_results':		cuda.jit('void(int64[:,:], int64[:,:])', fastmath = True)(add_check_int_ext_results_noreturn_cuda),
				'compute_results':					cuda.jit('void(int64[:,:], int64)', fastmath = True)(compute_results_noreturn_cuda),
				'count_points_result':				cuda.jit('void(int64[:,:], int64, int64, int64, int64, int64)', fastmath = True)(count_points_result_noreturn_cuda),
				'results_to_list':					cuda.jit('void(int64[:,:], int64[:,:], int64)', fastmath = True)(results_to_list_noreturn_cuda)
																	
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.config_loader_functions: {e}')

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
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.config_loader: {e}')
			
	def read_file(self):
		try:
			
			self.mesh_volume = o3d.io.read_triangle_mesh(os.path.join(self.path, self.model_file), print_progress = True)
			
			self.mesh_volume = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.mesh_volume, self.voxel_size)

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.read_file: {e}')
			
	def get_surface_points(self):
		try:
			
			self.get_mesh_center()
			
			triangle_count = len(self.mesh_volume.triangles)
			cloud = self.mesh_volume.sample_points_uniformly(number_of_points = int(triangle_count*self.unit_dict['triangle_multiplier']) )
			cloud = cloud.remove_duplicated_points()
			cloud = np.asarray(cloud.points).astype(self.VarType)
			self.auxiliar['mesh_limits'] = {
				'max' : np.ceil(np.max(cloud) / self.discretization) + 10,
				'min' : np.ceil(np.min(cloud) / self.discretization) - 10
				}
			#print(cloud.get_center())
			self.mesh_volume = cuda.to_device(cloud, dtype = cloud.dtype)
			
			self.erase_variable(cloud)
		
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_surface_points: {e}')
			
	def locate_mesh(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			assert self.mesh_center is not None, 'Mesh center not calculated, make sure you are executing the functions in the correct order.'
			
			self.scale()
			self.rotate()
			self.translate()

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.locate_mesh: {e}')
			
	def mesh_to_grid(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			assert self.mesh_center is not None, 'Mesh center not calculated, make sure you are executing the functions in the correct order.'
			
			self.config_manager(size = (self.mesh_volume.shape[0], self.mesh_volume.shape[1], self.mesh_volume.shape[2]),
								blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1], self.mesh_volume.shape[2]))

			self.config['mesh_to_grid'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.discretization)
	
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.mesh_to_grid: {e}')
			
	def check_int_ext(self, axis, reversed_path = False):
		try:
			
			assert int(axis) in [0, 1, 2], f'{axis} axis is not valid.'
			
			if self.auxiliar['grid_volume'] is None:
				self.auxiliar['grid_volume'] = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1, self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1,
																		self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1), dtype = np.int64))
			
			if self.results_grid is None:	
				self.results_grid = cuda.to_device(np.zeros((self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1, self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1,
															 self.auxiliar['mesh_limits']['max'] - self.auxiliar['mesh_limits']['min'] + 1), dtype = np.int64))
			
			if reversed_path:
				reversed_path = -1
				
			else:
				reversed_path = 1
				
			if int(axis) == 0:
				
				self.config_manager(size = (self.mesh_volume.shape[1], self.mesh_volume.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[1], self.mesh_volume.shape[2]))
			elif int(axis) == 1:
				
				self.config_manager(size = (self.mesh_volume.shape[0], self.mesh_volume.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[2]))
				
			elif int(axis) == 2:
				
				self.config_manager(size = (self.mesh_volume.shape[0], self.mesh_volume.shape[1]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))
				

			self.config['check_int_ext'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.auxiliar['grid_volume'], axis, reversed_path, self.auxiliar['mesh_limits']['min'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.check_int_ext: {e}')
			
	def add_check_int_ext_results(self):
		try:
			
			self.config_manager(size = (self.mesh_volume.shape[0], self.mesh_volume.shape[1], self.mesh_volume.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1], self.mesh_volume.shape[2]))
			

			self.config['add_check_int_ext_results'][self.griddim, self.blockdim, self.stream](self.results_grid, self.auxiliar['grid_volume'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.add_check_int_ext_results: {e}')
			
	def compute_results (self, threshold = 4):
		try:
			
			self.config_manager(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			self.config['compute_results'][self.griddim, self.blockdim, self.stream](self.results_grid, threshold)
			
			
			count = 0
			
			self.config_manager(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			self.config['count_points_result'][self.griddim, self.blockdim, self.stream](self.results_grid, self.grid_limits['min'], self.grid_limits['max'],
																						self.auxiliar['mesh_limits']['min'], self.auxiliar['mesh_limits']['max'], count)
			
			
			geometry_points = cuda.device_array((count, 3), dtype = np.int64, stream = self.stream)
			
			self.config_manager(size = (self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]),
									blockdim = optimize_blockdim(self.multiProcessorCount, self.results_grid.shape[0], self.results_grid.shape[1], self.results_grid.shape[2]))

			self.config['results_to_list'][self.griddim, self.blockdim, self.stream](self.results_grid, geometry_points, self.auxiliar['mesh_limits']['min'])
			
			return geometry_points

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.compute_results: {e}')
		
			
	def extract_object(self, threshold = 4):
		try:
			self.read_file()
			
			self.get_surface_points()
			
			self.locate_mesh()
			
			self.mesh_to_grid()
			
			for axis in range(3):
				self.check_int_ext(axis)
				self.add_check_int_ext_results()
				
				self.check_int_ext(axis, reversed_path=True)
				self.add_check_int_ext_results()
			
			return self.compute_results(threshold)

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.extract_object: {e}')



	def get_rotation_matrix (self, mesh_normal, mesh_center, where_to_point):
		try:
			
			if isinstance(mesh_normal, list):
				mesh_normal = np.array(mesh_normal)
			if isinstance(mesh_center, list):
				mesh_center = np.array(mesh_center)
			if mesh_center.shape != (3,):
				mesh_center = mesh_center.squeeze()
			if isinstance(where_to_point, list):
				where_to_point = np.array(where_to_point)
				
			assert mesh_center != where_to_point, 'Something may go wrong with an object pointing to its center.'
			assert np.linalg.norm(mesh_normal) != 0, f'Mesh normal has 0 norm.'
				
			final_normal = (where_to_point - mesh_center) / np.linalg.norm(where_to_point - mesh_center)
			
			if np.linalg.norm(mesh_normal) != 1:
				mesh_normal = mesh_normal / np.linalg.norm(mesh_normal)
			
			self.rotation_matrix = np.identity(3) + final_normal.dot(mesh_normal.T) - mesh_normal.dot(final_normal.T) + np.linalg.matrix_power(final_normal.dot(mesh_normal.T) - mesh_normal.dot(final_normal.T), 2) / ( 1 + mesh_normal.dot(final_normal))

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_rotation_matrix: {e}')
				
	def get_mesh_center (self):
		try:
			
			if self.mesh_center is None:
				self.mesh_center = self.mesh_volume.get_center()

		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.get_mesh_center: {e}')
			
	def rotate (self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			
			self.get_rotation_matrix(self.unit_dict['normal'], self.center, self.unit_dict['orientation'])
			self.auxiliar['rotate'] = cuda.to_device(np.zeros((self.mesh_volume.shape[0], self.mesh_volume.shape[1]), dtype = self.mesh_volume.dtype), stream = self.stream)
			
			self.config_manager(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))

			self.config['rotate'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.rotation_matrix, self.mesh_center, self.auxiliar['rotate'])
			
			self.mesh_volume = self.auxiliar['rotate']
			self.erase_variable(self.auxiliar['rotate'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.rotate: {e}')
			
	def translate(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
						
			self.config_manager(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))

			self.config['translate'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.unit_dict['location'])
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.translate: {e}')

	def scale(self):
		try:
			
			assert cuda.cudadrv.devicearray.is_cuda_ndarray(self.mesh_volume), 'Arrays must be loaded in GPU device.'
			
			if self.unit_dict['scale'] != 1.0:
			
				self.config_manager(size=(self.mesh_volume.shape[0], self.mesh_volume.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.mesh_volume.shape[0], self.mesh_volume.shape[1]))
			
				self.config['scale'][self.griddim, self.blockdim, self.stream](self.mesh_volume, self.unit_dict['scale'], self.mesh_center)
			
		except Exception as e:
			print(f'Error in utils.cuda.stlReader.stlReader_cuda.scale: {e}')


	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cpu.stlReader.stlReader_cuda.erase_variable: {e}')
