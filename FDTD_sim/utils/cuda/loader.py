import numpy as np

import numba, math, cmath
from numba import float32, float64, void, cuda
from .calculator import optimize_blockdim



class loader_cuda():
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
	