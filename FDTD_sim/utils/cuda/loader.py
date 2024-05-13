from tkinter import NORMAL
#from matplotlib.pyplot import grid
#import pandas as pd #, numpy as np 
import json

import os, numpy as np #, math, numba, cmath
from numba import cuda #, float32, float64, void
#from .calculator import optimize_blockdim
from .stlReader import stlReader_cuda as stl_reader
from .simple_objects import simple_objects_cuda as simple_objects





class loader_cuda():
	'''
	Configurate the loader
	'''
	def __init__(self, path, print_config = False, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		assert os.path.exists(path), f'Path {path} not valid.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.stream = stream
		
		self.config = None
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
				
		self.manager = None
		self.grid_limits = None
		self.emitter_center = None

		#self.config_loader(size, blockdim, stream)

		#self.config_loader_functions()
		
		self.load_configuration(path, print_config)
		
	'''
	Implement configurations
	''' 

	#def config_loader_functions (self):
	#	try:
	#		self.config = {
	#			'step_velocity_values':				cuda.jit('void('+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '
	#																+self.VarType+'[:,:,:], '	+self.VarType+'[:,:,:], '	+self.VarType+', '			+self.VarType+', '
	#																+self.VarType+', '			+'int64)', fastmath = True)(step_velocity_values_noreturn_cuda),
	#			'step_pressure_values':				cuda.jit('void('+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '
	#																+self.OutVarType+'[:,:,:], '+self.VarType+'[:,:,:], '	+self.VarType+'[:,:,:], '	+self.VarType+', '
	#																+self.VarType+', '			+self.VarType+')', fastmath = True)(step_pressure_values_noreturn_cuda),
	#			'set_velocity_emitters':			cuda.jit('void('+self.OutVarType+'[:,:,:], int64[:,:], '+self.VarType+'[:,:], '+self.VarType+')', fastmath = True)(step_pressure_values_noreturn_cuda)	
	#																
	#			}
	#		
	#	except Exception as e:
	#		print(f'Error in utils.cuda.loader.loader_cuda.config_loader_functions: {e}')

	#def config_loader(self, size=None, blockdim = None, stream = None):
	#	try:
	#		assert size is not None or blockdim is not None or stream is not None, 'No reconfiguration especified.'
	#		#assert len(blockdim)==2, 'Incorrect number of parameters.'
	#
	#		if size is not None:
	#			self.size = size
	#		if blockdim is not None:
	#			self.blockdim = blockdim
	#		if self.blockdim is not None and self.size is not None:
	#			
	#			if isinstance(self.size,int):
	#				self.griddim = int(np.ceil(self.size / self.blockdim[0]))
	#			elif len(size) == 2:
	#				self.griddim = (int(np.ceil(self.size[0] / self.blockdim[0])), int(np.ceil(self.size[1] / self.blockdim[1])))
	#			elif len(size) == 3:
	#				self.griddim = (int(np.ceil(self.size[0] / self.blockdim[0])), int(np.ceil(self.size[1] / self.blockdim[1])), int(np.ceil(self.size[2] / self.blockdim[2])))
	#		
	#		if stream is not None:
	#			self.stream = stream
	#	except Exception as e:
	#		print(f'Error in utils.cuda.loader.loader_cuda.config_loader: {e}')
		
	def set_extra_parameters(self, manager, grid_limits):
		try:
			
			self.manager = manager
			
			self.grid_limits = grid_limits
			
		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_configuration: {e}')

	def load_configuration(self, file_path, print_config=False):
		try:
			print(file_path)
			with open(file_path) as json_file:
				#print('In')
				self.configuration = json.load(json_file)
				
			print('Done')
			
			self.VarType = self.configuration['precission'][0]
			self.OutVarType = self.configuration['precission'][1]
			
			if print_config:
				print('\nConfiguration loaded:')
				self.print_configuration()

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_configuration: {e}')
			
	def print_configuration(self):
		try:
			print(self.configuration)

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_configuration: {e}')
			
	def load_transducers (self):
		try:
			
			if self.configuration['transducers'] != 'None':
				for transducer in self.configuration['transducers']:
					for n_unit in transducer['units']:
						self.load_emitter(transducer['model'], transducer['zone_emission'], transducer['file_extension'], transducer['amplitude'], transducer['frequency'], n_unit)

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_transducers: {e}')
	
	def load_emitter(self, path, zone_emission, file_extension, amplitude, frequency, dict_unit):
		try:
			
			reader = stl_reader(path, 'model', dict_unit, self.configuration['grid']['sim_parameters']['ds'], self.grid_limits, file_extension= file_extension,
								stream=self.stream, var_type=self.VarType, out_var_type=self.OutVarType)
			geometry_points, _ = reader.extract_object()
			mesh_center = reader.mesh_center
			
			print('Transducer loaded')

			reader = stl_reader(zone_emission, 'emission_zone', dict_unit, self.configuration['grid']['sim_parameters']['ds'], self.grid_limits, mesh_center=mesh_center, file_extension = file_extension,
								stream=self.stream, var_type=self.VarType, out_var_type=self.OutVarType)
			emission_points, normal_emission = reader.extract_object()
			
			print('Emission loaded')
			#print(normal_emission)
			self.erase_variable(reader)
			#import numpy as np
			#with np.printoptions(threshold=np.inf):
			#	print(geometry_points.copy_to_host())
			#	print(emission_points.copy_to_host())
			#geometry_points_check = np.sort(geometry_points.copy_to_host(), order=['f1'], axis = 0).view(np.int)
			#emission_points_check = np.sort(emission_points.copy_to_host(), order=['f1'], axis = 0).view(np.int)
			
			#geometry_points_check = geometry_points.copy_to_host()
			#geometry_points_check = geometry_points_check[geometry_points_check[:,2].argsort()]
			#geometry_points_check = geometry_points_check[geometry_points_check[:,1].argsort(kind='mergesort')]
			#geometry_points_check = geometry_points_check[geometry_points_check[:,0].argsort(kind='mergesort')]
			#emission_points_check = emission_points.copy_to_host()
			#geometry_points_check = emission_points_check[emission_points_check[:,2].argsort()]
			#geometry_points_check = emission_points_check[emission_points_check[:,1].argsort(kind='mergesort')]
			#geometry_points_check = emission_points_check[emission_points_check[:,0].argsort(kind='mergesort')]
			#
			#check_point = np.sum(geometry_points_check - emission_points_check)
			#
			#print('\n \n'+ str(geometry_points_check.shape) +'\n' + str(check_point) + '\n'+ str(emission_points_check.shape) +' \n \n')

			self.manager.locate_transducer(geometry_points, emission_points, np.array([amplitude]).astype(self.VarType)[0], 
											np.array([frequency]).astype(self.VarType)[0], np.array([dict_unit['initial_phase']]).astype(self.VarType)[0], 
											max_distance = 0, normal_emission = np.array(normal_emission).astype(self.VarType))
			
			print('Data loaded')

			self.erase_variable(geometry_points, emission_points, mesh_center)			

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_emitter: {e}')
			
	def load_objects (self):
		try:
			
			if self.configuration['objects'] != 'None':
				for obj in self.configuration['objects']:
					for n_unit in obj['units']:
						self.load_solid(obj['model'], obj['file_extension'], n_unit)

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_objects: {e}')
	
	def load_solid(self, path, file_extension, dict_unit):
		try:
			reader = stl_reader(path, 'model', dict_unit, self.configuration['grid']['sim_parameters']['ds'], self.grid_limits, file_extension= file_extension,
								stream=self.stream, var_type=self.VarType, out_var_type=self.OutVarType)
			
			geometry_points, _ = reader.extract_object()
			
			self.manager.locate_geometry_object(geometry_points, self.configuration['boundary']['max_object_distance'])
			
			self.erase_variable(reader, geometry_points)

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_solid: {e}')
	
	def load_boundaries (self):
		try:
			
			if self.configuration['boundaries'] != 'None':
				for boundary in self.configuration['boundaries']:
					if boundary['model'] == 'plain_wall':
						for n_unit in boundary['units']:
							self.load_wall(n_unit)

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_objects: {e}')

	def load_wall(self, n_unit):
		try:
			
			wall = simple_objects(self.configuration['grid']['sim_parameters']['ds'], self.grid_limits,
								stream=self.stream, var_type=self.VarType, out_var_type=self.OutVarType)
							
			geometry_points = wall.generate_wall(n_unit)
			
			print(geometry_points.shape, geometry_points.copy_to_host())

			self.manager.locate_geometry_object(geometry_points, n_unit['max_object_distance'])
			
			self.erase_variable(wall, geometry_points)
							
		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_wall: {e}')
	
	
	def load_plotter_configuration (self):
		try:
			
			mode = self.configuration['plot']['mode']
		
			if mode == "plane":

				return self.configuration['plot']
			
			else:
				print(f'Mode {mode} not implemented. Simulation will not be plotted.')
				return None				

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_solid: {e}')

	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.erase_variable: {e}')
