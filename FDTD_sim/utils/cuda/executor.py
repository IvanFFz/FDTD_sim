import numpy as np
from numpy import float32 as f32, float64 as f64, complex64 as c64, complex128 as c128

import numba, math, cmath
from numba import cuda
from .calculator import calculator_cuda, calculate_bd_b2 as to_b2, optimize_blockdim
from .manager import manager_cuda
from .solver import solver_cuda



def define_A_matrix_noreturn_cuda (matrix):
    '''
	Function that calculates the SQUARE of the distance
	'''
		
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
    if i<matrix.shape[0] and j<matrix.shape[1]:
        if i==j:
            matrix[i,j] = 0.5 + 0.0j
        else:
            matrix[i,j] = -1.0* matrix[i,j]


def calculate_transmission_matrix_noreturn_cuda (direct_contribution, sm_DD_Green_function, mesh_pressions, out):
    '''
	Function that calculates the SQUARE of the distance
	'''
		
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	
    if i<direct_contribution.shape[0] and j<direct_contribution.shape[1]:
        out[i,j] = direct_contribution[i,j]
        
        for k in range(mesh_pressions.shape[0]):
            #out[i,j] += sm_DD_Green_function[i,k] * mesh_pressions[k,j]
            cuda.atomic.add(out.real, (i,j), (sm_DD_Green_function[i,k] * mesh_pressions[k,j]).real)
            cuda.atomic.add(out.imag, (i,j), (sm_DD_Green_function[i,k] * mesh_pressions[k,j]).imag)
            	
def extract_data_for_mesh_pressions_noreturn_cuda (direct_contribution, adapted_data, elements):
    '''
	Function that extracts the necessary data to calculate mesh pressions
	'''
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if i<adapted_data.shape[0]:
        adapted_data[i] = direct_contribution[i,elements]
        
def save_mesh_pressions_noreturn_cuda (matrix, array, elements):
    '''
	Function that saves the data of the mesh pressions
	'''
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if i<array.shape[0]:
        matrix[i,elements] = array[i]
'''

Not prepared to use the "transducers_modifies_patterns" option. That means that the presence of the 
transducer in the volume does not affect the resulting patter. WIP.

'''

class executor ():
    
    def __init__ (self, ds, dt, nPoint_per_side, layer_thickness, maxDistEffect, airAbsorptivity = 0, size=None, var_type='float64', out_var_type = 'complex128', blockdim=(16,16)):
        
        assert cuda.is_available(), 'Cuda is not available.'
        #assert stream is not None, 'Cuda not configured. Stream required.'
        
        ## In the future we can modify it to use different transducers with different Pref's.
        self.Pref = Pref
        
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
        

        self.calculator = None
        
        self.manager = None
        
        self.solver = None
        
        self.blockdim = blockdim
        self.size = size
        self.griddim = None
        
        self.config_executor_functions()
        self.config_simulation(dt, ds, nPoint_per_side)
        self.config_geometries(layer_thickness, maxDistEffect, airAbsorptivity)
        
        self.auxiliar = {
                    
                    'sq_distance':		        	        None,
                    'cos_angle_sn':		        	        None,			
                    'bessel_divx':		        	        None,
                    #'sm_DD_Green_function':		    None,
                    #'direct_contribution_pm':		None,
                    'complex_phase':			            None,
                    'Total_Mesh':                           None,
                    'direct_contribution_emitter_mesh':     None,
                    'mesh_pression_emitter':                None
            }
        
        self.A_matrix = None
        self.mesh_pressions = None
        self.transmission_matrix = None
      
    def config_executor_functions (self):
        try:
            self.config = {
				'define_A_matrix':                      cuda.jit('void('+self.OutVarType+'[:,:])', fastmath = True)(define_A_matrix_noreturn_cuda),
                'calculate_transmission_matrix':        cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:])', fastmath = True)(calculate_transmission_matrix_noreturn_cuda),
                'extract_data_for_mesh_pressions':      cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:], int64)', fastmath = True)(extract_data_for_mesh_pressions_noreturn_cuda),
                'save_mesh_pressions':                 cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:], int64)', fastmath = True)(save_mesh_pressions_noreturn_cuda)
            }
			
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.config_functions: {e}')

    def config_executor(self, size=None, blockdim = None, stream = None):
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
            print(f'Error in utils.cuda.executor.executor.config_executor: {e}')
		     
    def config_geometries(self, layer_thickness, maxDistEffect, airAbsorptivity = 0):
        try:
             
            #To define absorptivity
            self.layer_thickness = layer_thickness
            self.maxAbsorptivity = 0.5 / self.dt
            self.airAbsorptivity = airAbsorptivity
             
            #To define the geometry field
            self.maxDistEffect = maxDistEffect
			
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.config_geometries: {e}')
            
    def config_simulation(self, dt, ds, nPoints):
        try:
            
            self.dt = dt
            self.ds = ds
            self.nPoints = nPoints
                        
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.config_simulation: {e}')

            

    def init_workspace (self, transducers_pos, transducers_norm, transducers_radius, select_emitters, number_elements, maxDist=6, maxValue=-2, transducers_modifies_patterns = False, mesh_pos = None, mesh_norm = None, mesh_area_div_4pi = None):
        try:
            
            self.stream = cuda.stream()
              
            self.config_executor(blockdim=self.blockdim, stream=self.stream)
        
            self.calculator = calculator_cuda(stream = self.stream, var_type = self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)
        
            self.manager = manager_cuda(stream = self.stream, var_type = self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)
            
            self.solver = solver_cuda(max_iter=100, stream=self.stream, var_type=self.VarType, out_var_type = self.OutVarType, blockdim=self.blockdim)

            self.init_grid()
		
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.init_workspace: {e}')
 
    def init_grid(self, maxDist=6, maxValue=-2):
        try:
            
            self.manager.pressure = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            
            self.manager.velocity_x = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.velocity_y = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.velocity_z = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
		
            self.manager.velocity_b = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            
            self.manager.geometry_field = cuda.to_device(np.ones((self.nPoints, self.nPoints, self.nPoints), dtype = self.var_type), stream = self.stream) #beta in the paper, init to air (1)
            self.manager.absorptivity = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.var_type), stream = self.stream) #sigma in the paper
            
            if maxValue==-2:
                maxValue = 0.5/self.dt
              
            self.manager.PML_limit_volume_no_return_cuda(self.manager.absorptivity, maxDist, maxValue)

            self.grid = cuda.device_array((positions_B.shape[0],positions_A.shape[0]), dtype = self.var_type, stream = self.stream)
		
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.init_workspace: {e}')

    def preprocess_transducers (self, transducers_modifies_patterns = False):
        try:
            
            self.manager.rel_transducer_transducer['direct_contribution_pm'] = cuda.device_array((self.manager.transducers_pos_cuda.shape[0],self.manager.transducers_pos_cuda.shape[0]), dtype = self.out_var_type, stream = self.stream)
            
            self.calculate_direct_contribution_pm(self.manager.transducers_pos_cuda, self.manager.transducers_pos_cuda, self.manager.transducers_norm_cuda,
                                                  self.k, self.manager.rel_transducer_transducer['direct_contribution_pm'])
            #Assume the area of the transducer to be pi*r**2, so area/4*pi = r**2/4
            if transducers_modifies_patterns:
                self.manager.rel_transducer_transducer['sm_DD_Green_function'] = cuda.device_array((self.manager.transducers_pos_cuda.shape[0],self.manager.transducers_pos_cuda.shape[0]), dtype = self.out_var_type, stream = self.stream)
			
                self.calculate_sm_DD_Green_function(self.manager.transducers_pos_cuda, self.manager.transducers_norm_cuda, self.manager.transducers_pos_cuda,
                                                    self.manager.transducers_radius**2/4, self.k, self.manager.rel_transducer_transducer['sm_DD_Green_function'],
                                                    sq_distance_calculated=True)            
            self.stream.synchronize()
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.preprocess_transducers: {e}')

    def calculate_sm_DD_Green_function (self, positions_A, normals_A, positions_B, area_div_4pi_A, k, out, sq_distance_calculated = False):
        
        '''
        calculates the Green function of the mesh element A (positions_A and normals_A) in the point B (positions_B)
        '''
                
        try:
            
            self.auxiliar['cos_angle_sn'] = cuda.device_array((positions_B.shape[0],positions_A.shape[0]), dtype = self.var_type, stream = self.stream)
            
            self.auxiliar['complex_phase'] = cuda.device_array((positions_B.shape[0],positions_A.shape[0]), dtype = self.out_var_type, stream = self.stream)
           
            if not sq_distance_calculated:
                self.auxiliar['sq_distance'] = cuda.device_array((positions_B.shape[0],positions_A.shape[0]), dtype = self.var_type, stream = self.stream)
                
                self.calculator.calculate_sq_distances(positions_B, positions_A, self.auxiliar['sq_distance'])
                #print(self.auxiliar['sq_distance'].copy_to_host())
                #try:
                #    while True:
                #        pass
                #except KeyboardInterrupt:
                #    pass

            self.calculator.calculate_cos_angle_sn(positions_B, positions_A, normals_A, self.auxiliar['cos_angle_sn'])
            
            self.stream.synchronize()
            #print(self.auxiliar['cos_angle_sn'].copy_to_host())
            #try:
            #    while True:
            #        pass
            #except KeyboardInterrupt:
            #    pass
            
            self.calculator.calculate_complex_phase(self.auxiliar['sq_distance'], k, self.auxiliar['complex_phase'])
            
            self.stream.synchronize()
            #print(self.auxiliar['complex_phase'].copy_to_host())
            #try:
            #    while True:
            #        pass
            #except KeyboardInterrupt:
            #    pass

            self.calculator.calculate_sm_DD_Green_function(area_div_4pi_A, self.auxiliar['sq_distance'], self.auxiliar['cos_angle_sn'], self.auxiliar['complex_phase'], k, out)
            
            self.stream.synchronize()
            
            self.manager.erase_variable(self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], self.auxiliar['cos_angle_sn'])
            
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_sm_DD_Green_function: {e}')
            
    def calculate_direct_contribution_pm (self, positions_A, positions_B , normals_B, k, out, sq_distance_calculated = False):
        
        '''
        calculates the direct contribution of the transducer located in positions_B with normal normals_B in the
        point positions_A
        '''
        
        try:
            
            self.auxiliar['bessel_divx'] = cuda.device_array((positions_A.shape[0],positions_B.shape[0]), dtype = self.var_type, stream = self.stream)
            self.auxiliar['complex_phase'] = cuda.device_array((positions_A.shape[0],positions_B.shape[0]), dtype = self.out_var_type, stream = self.stream)
            if not sq_distance_calculated:
                self.auxiliar['sq_distance'] = cuda.device_array((positions_A.shape[0],positions_B.shape[0]), dtype = self.var_type, stream = self.stream)
                self.calculator.calculate_sq_distances(positions_A, positions_B, self.auxiliar['sq_distance'])
                
            self.calculator.calculate_bessel_divx(k*self.manager.transducers_radius, positions_A, positions_B, normals_B, self.auxiliar['bessel_divx'], order = self.order)
            self.stream.synchronize()
            
            self.calculator.calculate_complex_phase(self.auxiliar['sq_distance'], k, self.auxiliar['complex_phase'])
            
            self.stream.synchronize()
            
            self.calculator.calculate_direct_contribution_pm(self.auxiliar['bessel_divx'], self.Pref, self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], out)
            self.stream.synchronize()
            
            self.manager.erase_variable(self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], self.auxiliar['bessel_divx'])

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_direct_contribution_pm: {e}')

    def define_A_matrix (self, matrix):
        
        try:
            
            self.config_executor(size=(matrix.shape[0], matrix.shape[1]), blockdim=optimize_blockdim(matrix.shape[0], matrix.shape[1]))
            
            self.config['define_A_matrix'][self.griddim, self.blockdim, self.stream](matrix)


        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.define_A_matrix: {e}')

    def calculate_A_matrix (self):
        try:
            self.calculate_sm_DD_Green_function(self.manager.mesh_pos['New'], self.manager.mesh_norm['New'], self.manager.mesh_pos['New'],
                                                self.manager.mesh_area_div_4pi['New'], self.k, self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['New']) 
            #print (self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['New'].copy_to_host())
            #
            #try:
            #    while True:
            #        pass
            #except KeyboardInterrupt:
            #    pass
            if self.manager.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'] is not None:
                
                self.calculate_sm_DD_Green_function(self.manager.mesh_pos['New'], self.manager.mesh_norm['New'], self.manager.mesh_pos['Old'],
                                                self.manager.mesh_area_div_4pi['New'], self.k, self.manager.rel_mesh_mesh['sm_DD_Green_function']['Old']['New']) 
                            
                self.calculate_sm_DD_Green_function(self.manager.mesh_pos['Old'], self.manager.mesh_norm['Old'], self.manager.mesh_pos['New'],
                                                self.manager.mesh_area_div_4pi['Old'], self.k, self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['Old']) 
            
                
                total_elements = self.manager.concatenate_matrix(self.manager.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'], self.manager.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'],
                                                                 self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'], self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['New'],
                                                                 self.OutVarType)
                
            else:
                
                total_elements = self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['New']
            
            if self.manager.rel_transducer_transducer['sm_DD_Green_function'] is not None:
                
                self.calculate_sm_DD_Green_function(self.manager.mesh_pos, self.manager.mesh_norm, self.manager.receivers_pos,
                                                self.manager.mesh_area_div_4pi, self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])
                self.calculate_sm_DD_Green_function(self.manager.receivers_pos, self.manager.receivers_norm, self.manager.mesh_pos,
                                                self.manager.transducers_radius**2/4, self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_receivers'])
            
                total_elements = self.manager.concatenate_matrix(self.manager.rel_receivers_receivers['sm_DD_Green_function'], self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'],
                                                                 self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_receivers'], total_elements)
            
            self.define_A_matrix(total_elements)
            
            self.A_matrix = total_elements                

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_A_matrix: {e}')

    def calculate_direct_contribution_emitters_mesh (self):
        try:
            if self.manager.mesh_pos['Old'] is not None:
                
                self.auxiliar['Total_Mesh'] = self.manager.concatenate_arrays(self.manager.mesh_pos['Old'], self.manager.mesh_pos['New'], self.VarType)
                self.stream.synchronize()
                self.calculate_direct_contribution_pm(self.auxiliar['Total_Mesh'], self.manager.emitters_pos, self.manager.emitters_norm,
                                                      self.k, self.manager.rel_emitters_mesh['direct_contribution_pm'])
            
            else:
                
                self.calculate_direct_contribution_pm(self.manager.mesh_pos['New'], self.manager.emitters_pos, self.manager.emitters_norm,
                                                      self.k, self.manager.rel_emitters_mesh['direct_contribution_pm'])
            #print('yup',self.manager.rel_emitters_mesh['direct_contribution_pm'].copy_to_host())
            if self.manager.rel_transducer_transducer['sm_DD_Green_function'] is not None:
                return self.manager.concatenate_arrays(self.manager.rel_emitters_mesh['direct_contribution_pm'], self.manager.rel_emitters_receivers['direct_contribution_pm'], self.OutVarType)
            else:
                return self.manager.rel_emitters_mesh['direct_contribution_pm']

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_direct_contribution_emitters_mesh: {e}')

    def calculate_GF_mesh_to_receiver(self, A_calculated_with_interaction_m_r = False):
        try:
            
            if not A_calculated_with_interaction_m_r:
                if self.manager.mesh_pos['Old'] is not None:    
                    self.calculate_sm_DD_Green_function(self.manager.concatenate_arrays(self.manager.mesh_pos['Old'], self.manager.mesh_pos['New'], self.VarType),
                                                         self.manager.concatenate_arrays(self.manager.mesh_norm['Old'], self.manager.mesh_norm['New'], self.VarType),
                                                        self.manager.receivers_pos,
                                                self.manager.concatenate_arrays(self.manager.mesh_area_div_4pi['Old'], self.manager.mesh_area_div_4pi['New'], self.VarType), self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])
                
                else:
                    self.calculate_sm_DD_Green_function(self.manager.mesh_pos['New'], self.manager.mesh_norm['New'], self.manager.receivers_pos,
                                                self.manager.mesh_area_div_4pi['New'], self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])
                
            if self.manager.rel_transducer_transducer['sm_DD_Green_function'] is not None:
                return self.manager.concatenate_arrays(self.manager.rel_receivers_receivers['sm_DD_Green_function'], self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'], self.OutVarType)
            else:
                return  self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh']

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_GF_mesh_to_receiver: {e}')

    def calculate_transmission_matrix(self, mesh_pression, A_calculated_with_interaction_m_r = False):
        '''
        Not prepared to add the pression generated in the receivers. Work to be done.
        '''
        try:
            
            GF_matrix = self.calculate_GF_mesh_to_receiver(A_calculated_with_interaction_m_r)
            #print(GF_matrix.shape)
            #assert mesh_pression.shape[1] == direct_contribution.shape[1] and mesh_pression.shape[0] == GF_matrix.shape[1], f'Shape of pression matrix is not correct: {mesh_pression.shape, direct_contribution.shape}.'

            self.config_executor(size=(self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[0], self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[1]), blockdim=optimize_blockdim(self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[0], self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[1]))
            
            self.transmission_matrix = cuda.device_array((self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[0] , self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[1]), dtype = self.manager.rel_emitters_receivers['direct_contribution_pm'].dtype, stream = self.stream)
            #print(mesh_pression.shape)
            self.config['calculate_transmission_matrix'][self.griddim, self.blockdim, self.stream](self.manager.rel_emitters_receivers['direct_contribution_pm'], GF_matrix, 
                                                                                                   mesh_pression,
                                                                                                 self.transmission_matrix)
            #print(self.transmission_matrix.shape) 
            
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_transmission_matrix: {e}')
            
    def prepare_arrays_for_mesh_pressions(self, elements):
        try:
            
            self.config_executor(size=(self.manager.rel_emitters_mesh['direct_contribution_pm'].shape[0]), blockdim=optimize_blockdim(self.manager.rel_emitters_mesh['direct_contribution_pm'].shape[0]))
            print('config')
            self.config['extract_data_for_mesh_pressions'][self.griddim, self.blockdim, self.stream](self.manager.rel_emitters_mesh['direct_contribution_pm'], self.auxiliar['direct_contribution_emitter_mesh'], elements)
            print('config done') 
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.prepare_arrays_for_mesh_pressions: {e}')

    def save_data_for_mesh_pressions(self, elements):
        try:
            
            self.config_executor(size=(self.manager.rel_emitters_mesh['direct_contribution_pm'].shape[0]), blockdim=optimize_blockdim(self.manager.rel_emitters_mesh['direct_contribution_pm'].shape[0]))
            print('config')
            self.config['save_mesh_pressions'][self.griddim, self.blockdim, self.stream](self.mesh_pressions, self.auxiliar['mesh_pression_emitter'], elements)
            print('config done')    
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.save_data_for_mesh_pressions: {e}')


    def calculate_mesh_pressions (self):
        try:
            
            if self.manager.mesh_pos['Old'] is not None:
                self.mesh_pressions = cuda.to_device(np.zeros((self.manager.mesh_pos['Old'].shape[0] + self.manager.mesh_pos['New'].shape[0], self.manager.emitters_pos.shape[0])).astype(self.var_type) + 1j*np.zeros((self.manager.mesh_pos['Old'].shape[0] + self.manager.mesh_pos['New'].shape[0], self.manager.emitters_pos.shape[0])).astype(self.var_type), stream = self.stream)
            else:
                self.mesh_pressions = cuda.to_device(np.zeros((self.manager.mesh_pos['New'].shape[0], self.manager.emitters_pos.shape[0])).astype(self.var_type) + 1j*np.zeros((self.manager.mesh_pos['New'].shape[0], self.manager.emitters_pos.shape[0])).astype(self.var_type), stream = self.stream)
			
            self.calculate_direct_contribution_emitters_mesh()
            
            for i in range(self.manager.emitters_pos.shape[0]):
                print('in')
                if self.manager.mesh_pos['Old'] is not None:
                    self.auxiliar['mesh_pression_emitter'] = cuda.to_device(np.zeros(self.manager.mesh_pos['Old'].shape[0] + self.manager.mesh_pos['New'].shape[0]).astype(self.var_type) + 1j*np.zeros(self.manager.mesh_pos['Old'].shape[0] + self.manager.mesh_pos['New'].shape[0]).astype(self.var_type), stream = self.stream)
                else:
                    self.auxiliar['mesh_pression_emitter'] = cuda.to_device(np.zeros(self.manager.mesh_pos['New'].shape[0]).astype(self.var_type) + 1j*np.zeros(self.manager.mesh_pos['New'].shape[0]).astype(self.var_type), stream = self.stream)
                
                self.auxiliar['direct_contribution_emitter_mesh'] = cuda.device_array(self.manager.rel_emitters_mesh['direct_contribution_pm'].shape[0], dtype = self.manager.rel_emitters_mesh['direct_contribution_pm'].dtype, stream = self.stream)
                print('init')
                self.prepare_arrays_for_mesh_pressions(i)
                print('prepared')
                self.stream.synchronize()
                #print('problem', self.A_matrix.copy_to_host())
                self.solver.calculate_solution_linear_system(self.A_matrix, self.auxiliar['direct_contribution_emitter_mesh'], self.auxiliar['mesh_pression_emitter'])
                print('solved')
                self.save_data_for_mesh_pressions(i)
                print('saved', i)
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.calculate_mesh_pressions: {e}')
            









