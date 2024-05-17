from mimetypes import init
import numpy as np
from numpy import float32 as f32, float64 as f64, complex64 as c64, complex128 as c128

import numba, math, cmath
from numba import float32, float64, void, cuda
from utils.cuda.calculator import optimize_blockdim

'''
Implementation of a step of Jacobi's iterative method
norm_error must be 0
'''
def solve_linear_system_noreturn_cuda (coefficients, b_part, guess, unknown):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<guess.shape[0]:
		#unknown[i] = 0.0 + 0.0j
		for j in range(coefficients.shape[1]):
			if j!=i:
				#cuda.atomic.add(unknown.real, i, -1*(coefficients[i,j]*guess[j]).real)
				#cuda.atomic.add(unknown.imag, i, -1*(coefficients[i,j]*guess[j]).imag)
				unknown[i] -= (coefficients[i,j]*guess[j])/coefficients[i,i]


		#cuda.atomic.add(unknown.real, i, b_part[i].real)
		#cuda.atomic.add(unknown.imag, i, b_part[i].imag)
		unknown[i] += b_part[i]/coefficients[i,i]

		#unknown[i] = unknown[i]/coefficients[i,i]
		
def calculate_norm_error_complexNumbers_and_copy_noreturn_cuda(old, new, norm_error):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	
	if i<old.shape[0]:
		cuda.atomic.add(norm_error, 0, ((old[i]-new[i]).real)**2 + ((old[i]-new[i]).imag)**2)

		old[i] = new[i]

class solver_cuda():
	'''
	Configurate the calculator
	'''
	def __init__(self, max_iter=100, stream=None, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		if var_type == 'float32':
			self.var_type = f32
		elif var_type == 'float64':
			self.var_type = f64
		else:
			raise Exception (f'Bad type selected {var_type}')
        
		self.VarType = var_type
        
		if out_var_type == 'complex64':
			self.out_var_type = c64
		elif out_var_type == 'complex128':
			self.out_var_type = c128
		else:
			raise Exception (f'Bad type selected {out_var_type}')
        
		self.OutVarType = out_var_type
		
		self.stream = stream
		
		self.config = None
		self.size = size
		self.blockdim = blockdim
		self.griddim = None
		
		self.max_iter = max_iter

		self.config_solver(size, blockdim, stream)

		self.config_solver_functions()
		
	'''
	Implement configurations
	'''

	def config_solver_functions (self):
		try:
			self.config = {
				'solve_linear_system':						cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:], '+self.OutVarType+'[:], '+self.OutVarType+'[:])', fastmath = True)(solve_linear_system_noreturn_cuda),
				'calculate_norm_error_complexNumbers_and_copy':		cuda.jit('void('+self.OutVarType+'[:], '+self.OutVarType+'[:], '+self.VarType+'[:])', fastmath = True)(calculate_norm_error_complexNumbers_and_copy_noreturn_cuda)
				}
			
		except Exception as e:
			print(f'Error in utils.cuda.solver.solver_cuda.config_solver_functions: {e}')

	def config_solver(self, size=None, blockdim = None, stream = None):
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
			
			if stream is not None:
				self.stream = stream
		except Exception as e:
			print(f'Error in utils.cuda.solver.solver_cuda.config_solver: {e}')
	
	'''
	Implement functions that do the things
	'''

	def calculate_solution_linear_system (self, coefficients, b_part, initial_guess, max_norm_error=1e-3):
		try:
			assert (cuda.cudadrv.devicearray.is_cuda_ndarray(coefficients) and cuda.cudadrv.devicearray.is_cuda_ndarray(b_part) 
				and cuda.cudadrv.devicearray.is_cuda_ndarray(initial_guess)), 'Arrays must be loaded in GPU device.'

			max_norm_error = self.var_type(max_norm_error)

			norm_error = np.array([2 * max_norm_error]).astype(self.var_type)
			
			iterations = 0
			#print(b_part.copy_to_host())
			while norm_error[0] > max_norm_error and iterations < self.max_iter:
				#print ('ig\n',initial_guess.copy_to_host())
				
				unknown = cuda.to_device(np.zeros(initial_guess.shape[0]).astype(self.var_type) + 1j*np.zeros(initial_guess.shape[0]).astype(self.var_type), stream = self.stream)
				
				self.config_solver(size=(coefficients.shape[0]), blockdim=optimize_blockdim(coefficients.shape[0]))
				self.config['solve_linear_system'][self.griddim, self.blockdim, self.stream](coefficients, b_part, initial_guess, unknown)
				
				self.stream.synchronize()

				self.config_solver(size=(initial_guess.shape[0]), blockdim=optimize_blockdim(initial_guess.shape[0]))
				norm_error[0] = 0
				self.config['calculate_norm_error_complexNumbers_and_copy'][self.griddim, self.blockdim, self.stream](initial_guess, unknown, norm_error)
				
				self.stream.synchronize()

				#initial_guess = unknown
				iterations +=1
				print(iterations, norm_error[0])
				#print ('u\n',unknown.copy_to_host())

		except Exception as e:
			print(f'Error in utils.cuda.solver.solver_cuda.calculate_solution_linear_system: {e}')
