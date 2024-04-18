import numpy as np, pandas as pd

import numba, math, cmath, os
from numba import float32, float64, void, cuda
from .calculator import optimize_blockdim





class loader_cuda():
	'''
	Configurate the calculator
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

		self.config_loader(size, blockdim, stream)

		self.config_loader_functions()
		
		self.load_configuration(path, print_config)
		
	'''
	Implement configurations
	''' 

	def config_loader_functions (self):
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
			print(f'Error in utils.cuda.loader.loader_cuda.config_loader_functions: {e}')

	def config_loader(self, size=None, blockdim = None, stream = None):
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
			print(f'Error in utils.cuda.loader.loader_cuda.config_loader: {e}')
			

	def load_configuration(self, file_path, print_config=False):
		try:
			self.configuration = pd.read_json(file_path, lines=True)
			
			if print_config:
				print('\nConfiguration loaded:')
				self.print_configuration()

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_configuration: {e}')
			
	def print_configuration(self):
		try:
			print(self.configuration.to_string())

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_configuration: {e}')
			
	def load_transducers (self):
		try:
			
			for transducer in self.configuration['transducers']:
				for n_unit in transducer['units']:
					self.load_emitter(transducer['model'], transducer['zone_emission'], n_unit['location'], n_unit['angle'])

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_transducers: {e}')
	
	def load_emitter(self, path, zone_emission, location, angle):
		try:
			print('Work in progress')

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_emitter: {e}')
			
	def load_objects (self):
		try:
			
			for obj in self.configuration['objects']:
				for n_unit in obj['units']:
					self.load_solid(obj['model'], obj['zone_emission'], n_unit['location'], n_unit['angle'])

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_objects: {e}')
	
	def load_solid(self, path, zone_emission, location, angle):
		try:
			print('Work in progress')

		except Exception as e:
			print(f'Error in utils.cuda.loader.loader_cuda.load_solid: {e}')
	