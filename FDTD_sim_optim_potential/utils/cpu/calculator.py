import numpy as np, numba, math, cmath
from numba import njit, prange

'''
Implement functions used to calculate the thigs
'''

def sq_distance_noreturn_cpu (A, B, out):
	'''
	Function that calculates distance between A and B
	'''
	for i in prange(A.shape[0]):
		for j in prange(B.shape[0]):
			out[i,j] = (A[i, 0] - B[j, 0])**2 + (A[i, 1] - B[j, 1])**2 + (A[i, 2] - B[j, 2])**2
		
def cos_angle_sn_noreturn_cpu (A, B, C, out):
	'''
	Function that calculates the cosine of the angle between A-B and C
	'''
	for i in prange(A.shape[0]):
		for j in prange(B.shape[0]):
			if A[i,0]!=B[j,0] or A[i,1]!=B[j,1] or A[i,2]!=B[j,2]:
				out[i,j] = ((A[i, 0]-B[j, 0])*C[j, 0]+(A[i, 1]-B[j, 1])*C[j, 1]+(A[i, 2]-B[j, 2])*C[j, 2])/math.sqrt(
					((A[i, 0]-B[j, 0])**2+(A[i, 1]-B[j, 1])**2+(A[i, 2]-B[j, 2])**2)*(C[j, 0]**2+C[j, 1]**2+C[j, 2]**2))
			else:
				out[i,j] = 0


def sin_angle_np_noreturn_cpu (A, B, C, out, order):
	'''
	Function that calculates the sine of the angle between A - B and C
	'''
	if order == 0:
		for i in prange(A.shape[0]):
			for j in prange(B.shape[0]):
				if A[i,0]!=B[j,0] or A[i,1]!=B[j,1] or A[i,2]!=B[j,2]:
					out[i,j] = 1.0
				else:
					out[i,j] = 0.
	else:
		for i in prange(A.shape[0]):
			for j in prange(B.shape[0]):
				if A[i,0]!=B[j,0] or A[i,1]!=B[j,1] or A[i,2]!=B[j,2]:
					out[i,j] = math.sqrt( 1 - ((A[i, 0]-B[j, 0])*C[j, 0]+(A[i, 1]-B[j, 1])*C[j, 1]+(A[i, 2]-B[j, 2])*C[j, 2])**2 / (
										((A[i, 0]-B[j, 0])**2+(A[i, 1]-B[j, 1])**2+(A[i, 2]-B[j, 2])**2)*(C[j, 0]**2+C[j, 1]**2+C[j, 2]**2)) )
				else:
					out[i,j] = 0.

def bessel_o1_divx_noreturn_cpu (kr, out, order):
	'''
	Function that calculates the approximation of (the bessel function of first order of a value multiplyed by the elements of a matrix) divided by
	the value multiplyed by the elements of the matrix. Multiplyed by two to make coincide with expression 12 of "High-speed acoustic holography with arbitrary
	scattering objects"

	It can be initialized to 1 to avoid the time wasted in filling it.
		
	Optimized to be run with sin_angle_np_noreturn_cpu
	'''
	if order == 0:
		pass
	elif order  == 2:
		for i in prange(out.shape[0]):
			for j in prange(out.shape[1]):
				out[i,j] = 1.0 - (kr * out[i,j])**2 / 8
	elif order == 4:
		for i in prange(out.shape[0]):
			for j in prange(out.shape[1]):
				out[i,j] = 1.0 - (kr * out[i,j])**2 / 8 + (kr * out[i,j])**4 / 192 
	elif order == 6:
		for i in prange(out.shape[0]):
			for j in prange(out.shape[1]):
				out[i,j] = 1.0 - (kr * out[i,j])**2 / 8 + (kr * out[i,j])**4 / 192 -  (kr * out[i,j])**6 / 9216
	elif order == 8:
		for i in prange(out.shape[0]):
			for j in prange(out.shape[1]):
				out[i,j] = 1.0 - (kr * out[i,j])**2 / 8 + (kr * out[i,j])**4 / 192 -  (kr * out[i,j])**6 / 9216 + (kr * out[i,j])**8 / 737280
	elif order == 10:
		for i in prange(out.shape[0]):
			for j in prange(out.shape[1]):
				out[i,j] = 1.0 - (kr * out[i,j])**2 / 8 + (kr * out[i,j])**4 / 192 -  (kr * out[i,j])**6 / 9216 + (kr * out[i,j])**8 / 737280 -  (kr * out[i,j])**10 / 88473600
		
def sm_DD_Green_function_noreturn_cpu (area_div_4pi, sq_dist, cos_angle, phase, k, out):
	'''
	Function that calculates area sm multiplyed by the directional derivative of green's function
	'''
	for a in prange(sq_dist.shape[0]):
		for b in prange(sq_dist.shape[1]):
			if sq_dist[a, b] != 0.0:
				out[a, b] = area_div_4pi[b, 0] * cos_angle[a, b] * math.sqrt(1 + k**2 * sq_dist[a, b]) * phase[a, b] /sq_dist[a, b]
			else:
				out[a,b] = 0.0 + 0.0j

def direct_contribution_pm_noreturn_cpu (bessel, Pref, sq_dist, phase, out):
	'''
	Function that calculates the direct incident contribution of a transducer using a piston model
	'''
	for a in prange(sq_dist.shape[0]):
		for b in prange(sq_dist.shape[1]):
			if sq_dist[a, b] != 0.0:
				out[a, b] = bessel[a, b] * Pref * phase[a, b] / math.sqrt(sq_dist[a, b])
			else:
				out[a,b] = 0.0 + 0.0j
		
def complex_phase_noreturn_cpu (sq_dist, k, out):
	'''
	Function that calculates the scalar product
	'''
	for a in prange(sq_dist.shape[0]):
		for b in prange(sq_dist.shape[1]):
			out[a, b] = cmath.exp(1j*k*math.sqrt(sq_dist[a, b]) - math.atan(k*math.sqrt(sq_dist[a, b])))
      

class calculator_cpu():
	'''
	Configurate the calculator
	'''
	def __init__(self, var_type='float64', out_var_type='complex128'):

		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.config = None

		self.config_calculator_functions()

	def config_calculator_functions (self):
		try:
			self.config = {
				'sq_distance':				njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(sq_distance_noreturn_cpu),
				'cos_angle_sn':				njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(cos_angle_sn_noreturn_cpu),
				'sin_angle_np':				njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], int32)', parallel=True, fastmath = True)(sin_angle_np_noreturn_cpu),
				'bessel_divx':				njit('void('+self.VarType+', '+self.VarType+'[:,::1], int32)', parallel=True, fastmath = True)(bessel_o1_divx_noreturn_cpu),
				'sm_DD_Green_function':		njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.OutVarType+'[:,::1], '
														+self.VarType+', '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(sm_DD_Green_function_noreturn_cpu),
				'direct_contribution_pm':	njit('void('+self.VarType+'[:,::1], '+self.VarType+', '+self.VarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(direct_contribution_pm_noreturn_cpu),
				'complex_phase':			njit('void('+self.VarType+'[:,::1], '+self.VarType+', '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(complex_phase_noreturn_cpu)
				}

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.config_calculator_functions: {e}')

	  
	'''
	Implement functions that do the things
	'''

	def calculate_sq_distances (self, A, B, out):
		try:
			assert (type(A) == np.ndarray and type(B) == np.ndarray 
				and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'

			self.config['sq_distance'](A, B, out)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_sq_distances: {e}')

	def calculate_cos_angle_sn (self, A, B, C, out):
		try:
			assert (type(A) == np.ndarray and type(B) == np.ndarray 
				and type(C) == np.ndarray and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert B.shape[0] == C.shape[0], 'Each element must have a normal vector associated to it.'

			self.config['cos_angle_sn'](A, B, C, out)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_cos_angle_sn: {e}')

	def calculate_sin_angle_np (self, A, B, C, out, order = 0):
		try:
			assert (type(A) == np.ndarray and type(B) == np.ndarray 
				and type(C) == np.ndarray and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert int(order) in [0, 1, 2], 'Code not prepared for this order of approximation.'
			assert B.shape[0] == C.shape[0], 'Each element must have a normal vector associated to it.'

			self.config['sin_angle_np'](A, B, C, out, order)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_sin_angle_np: {e}')

	def calculate_bessel_divx (self, kr, A, B, C, out, order = 0):
		try:
			assert (type(A) == np.ndarray and type(B) == np.ndarray 
				and type(C) == np.ndarray and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert isinstance(kr, float), 'Value must be float.'
			assert B.shape[0] == C.shape[0], 'Each element must have a normal vector associated to it.'
			assert int(order) in [0, 2, 4, 6, 8, 10], 'Code not prepared for this order of approximation.'
			
			#self.config['sin_angle_np'](A, B, C, out, int(order))
			self.calculate_sin_angle_np(A, B, C, out, int(order))

			self.config['bessel_divx'](kr, out, int(order))

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_bessel_divx: {e}')

	def calculate_sm_DD_Green_function (self, area_div_4pi, sq_dist, cos_angle, phase, k, out):
		try:
			assert (type(area_div_4pi) == np.ndarray and type(sq_dist) == np.ndarray and type(cos_angle) == np.ndarray
				and type(phase) == np.ndarray and type(out) == np.ndarray), f'Arrays must be defined as numpy array.\n area_div_4pi: {area_div_4pi}\n sq_dist: {sq_dist}\n cos_angle: {cos_angle}\n phase: {phase}'
			assert isinstance(k, float), 'Constant k must be float.'
			assert (area_div_4pi.shape[0] == sq_dist.shape[1] and sq_dist.shape[0] == cos_angle.shape[0]
				and cos_angle.shape[0] == phase.shape[0]), f'Dimensions 0 does not match:\n area_div_4pi:{area_div_4pi.shape}\n sq_dist:{sq_dist.shape}\n cos_angle: {cos_angle.shape}\n phase: {phase.shape}.'
			assert (sq_dist.shape[1] == cos_angle.shape[1] and cos_angle.shape[1] == phase.shape[1]), 'Dimensions 1 does not match.'
			
			self.config['sm_DD_Green_function'](area_div_4pi, sq_dist, cos_angle, phase, k, out)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_sm_DD_Green_function: {e}')

	def calculate_direct_contribution_pm (self, bessel, Pref, sq_dist, phase, out):
		try:
			assert (type(bessel) == np.ndarray and type(sq_dist) == np.ndarray
				and type(phase) == np.ndarray and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert isinstance(Pref, float), 'Value must be float.'
			assert (bessel.shape[0] == sq_dist.shape[0] and sq_dist.shape[0] == phase.shape[0]), 'Dimensions 0 does not match.'
			assert (bessel.shape[1] == sq_dist.shape[1] and sq_dist.shape[1] == phase.shape[1]), 'Dimensions 1 does not match.'
			
			self.config['direct_contribution_pm'](bessel, Pref, sq_dist, phase, out)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_direct_contribution_pm: {e}')

	def calculate_complex_phase (self, sq_dist, k, out):
		try:
			assert (type(sq_dist) == np.ndarray and type(out) == np.ndarray), 'Arrays must be defined as numpy array.'
			assert isinstance(k, float), 'Constant k must be float.'

			self.config['complex_phase'](sq_dist, k, out)

		except Exception as e:
			print(f'Error in utils.cpu.calculator.calculator_cpu.calculate_complex_phase: {e}')