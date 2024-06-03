import numpy as np, numba, math, cmath
from numpy import float32 as f32, float64 as f64, complex64 as c64, complex128 as c128

from numba import njit, prange
from .calculator import calculator_cpu
from .manager import manager_cpu

def define_A_matrix_noreturn_cpu (matrix):
    '''
	Function that calculates the SQUARE of the distance
	'''
		
    for i in prange(matrix.shape[0]):
        for j in prange(matrix.shape[1]):
            if i==j:
                matrix[i,j] = 0.5 + 0.0j
            else:
                matrix[i,j] = -1.0* matrix[i,j]


def calculate_transmission_matrix_noreturn_cpu (direct_contribution, sm_DD_Green_function, mesh_pressions, out):
    '''
	Function that calculates the SQUARE of the distance
	'''
		
    for i in prange(direct_contribution.shape[0]):
        for j in prange(direct_contribution.shape[1]):
            out[i,j] = direct_contribution[i,j]
        
            for k in range(mesh_pressions.shape[0]):
                out[i,j] += sm_DD_Green_function[i,k] * mesh_pressions[k,j]
	

'''

Not prepared to use the "transducers_modifies_patterns" option. That means that the presence of the 
transducer in the volume does not affect the resulting patter. WIP.

'''

class executor ():
    
    def __init__ (self, Pref, var_type='float64', out_var_type = 'complex128'):
        
        
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
        
        self.config = None
       
        self.config_executor_functions()
        
        self.auxiliar = {
                    
                    'sq_distance':		        	None,
                    'cos_angle_sn':		        	None,			
                    'bessel_divx':		        	None,
                    #'sm_DD_Green_function':		    None,
                    #'direct_contribution_pm':		None,
                    'complex_phase':			    None,
                    'Total_Mesh':                   None
            }
        
        self.A_matrix = None
        self.transmission_matrix = None
      
    def config_executor_functions (self):
        try:
            self.config = {
				'define_A_matrix':                      njit('void('+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(define_A_matrix_noreturn_cpu),
                'calculate_transmission_matrix':        njit('void('+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(calculate_transmission_matrix_noreturn_cpu)
            }
			
        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.config_executor_functions: {e}')


    def init_workspace (self, transducers_pos, transducers_norm, transducers_radius, select_emitters, number_elements, k, order = 0, transducers_modifies_patterns = False, mesh_pos = None, mesh_norm = None, mesh_area_div_4pi = None):
        try:
            
            self.k = k
            self.order = order
            
            self.calculator = calculator_cpu(var_type = self.VarType, out_var_type = self.OutVarType)
        
            self.manager = manager_cpu(var_type = self.VarType, out_var_type = self.OutVarType)
            
            self.manager.locate_transducers(transducers_pos, transducers_norm, transducers_radius)
            
            self.preprocess_transducers(transducers_modifies_patterns)
            
            self.manager.assign_emitters(select_emitters)
            
            self.manager.init_mesh(number_elements)
            
            if mesh_pos is not None and mesh_norm is not None and mesh_area_div_4pi is not None:
                self.manager.manually_fill_mesh(mesh_pos, mesh_norm, mesh_area_div_4pi)
		
        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.init_workspace: {e}')
            
    def preprocess_transducers (self, transducers_modifies_patterns = False):
        try:
            
            self.manager.rel_transducer_transducer['direct_contribution_pm'] = np.empty([self.manager.transducers_pos_cuda.shape[0],self.manager.transducers_pos_cuda.shape[0]], dtype = self.out_var_type)
            
            self.calculate_direct_contribution_pm(self.manager.transducers_pos_cuda, self.manager.transducers_pos_cuda, self.manager.transducers_norm_cuda,
                                                  self.k, self.manager.rel_transducer_transducer['direct_contribution_pm'])
            #Assume the area of the transducer to be pi*r**2, so area/4*pi = r**2/4
            if transducers_modifies_patterns:
                self.manager.rel_transducer_transducer['sm_DD_Green_function'] = np.empty([self.manager.transducers_pos_cuda.shape[0],self.manager.transducers_pos_cuda.shape[0]], dtype = self.out_var_type)
			
                self.calculate_sm_DD_Green_function(self.manager.transducers_pos_cuda, self.manager.transducers_norm_cuda, self.manager.transducers_pos_cuda,
                                                    self.manager.transducers_radius**2/4, self.k, self.manager.rel_transducer_transducer['sm_DD_Green_function'],
                                                    sq_distance_calculated=True)            
		
        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.preprocess_transducers: {e}')

    def calculate_sm_DD_Green_function (self, positions_A, normals_A, positions_B, area_div_4pi_A, k, out, sq_distance_calculated = False):
        
        '''
        calculates the Green function of the mesh element A (positions_A and normals_A) in the point B (positions_B)
        '''
                
        try:
            
            self.auxiliar['cos_angle_sn'] = np.empty([positions_B.shape[0],positions_A.shape[0]], dtype = self.var_type)
            
            self.auxiliar['complex_phase'] = np.empty([positions_B.shape[0],positions_A.shape[0]], dtype = self.out_var_type)
           
            if not sq_distance_calculated:
                self.auxiliar['sq_distance'] = np.empty([positions_B.shape[0],positions_A.shape[0]], dtype = self.var_type)
                
                self.calculator.calculate_sq_distances(positions_B, positions_A, self.auxiliar['sq_distance'])
                
            self.calculator.calculate_cos_angle_sn(positions_B, positions_A, normals_A, self.auxiliar['cos_angle_sn'])
            
            self.calculator.calculate_complex_phase(self.auxiliar['sq_distance'], k, self.auxiliar['complex_phase'])
            
            self.calculator.calculate_sm_DD_Green_function(area_div_4pi_A, self.auxiliar['sq_distance'], self.auxiliar['cos_angle_sn'], self.auxiliar['complex_phase'], k, out)
            
            self.manager.erase_variable(self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], self.auxiliar['cos_angle_sn'])

        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.calculate_sm_DD_Green_function: {e}')
            
    def calculate_direct_contribution_pm (self, positions_A, positions_B , normals_B, k, out, sq_distance_calculated = False):
        
        '''
        calculates the direct contribution of the transducer located in positions_B with normal normals_B in the
        point positions_A
        '''
        
        try:
            
            self.auxiliar['bessel_divx'] = np.empty([positions_A.shape[0],positions_B.shape[0]], dtype = self.var_type)
            self.auxiliar['complex_phase'] = np.empty([positions_A.shape[0],positions_B.shape[0]], dtype = self.out_var_type)
            if not sq_distance_calculated:
                self.auxiliar['sq_distance'] = np.empty([positions_A.shape[0],positions_B.shape[0]], dtype = self.var_type)
                self.calculator.calculate_sq_distances(positions_A, positions_B, self.auxiliar['sq_distance'])
                
            self.calculator.calculate_bessel_divx(k*self.manager.transducers_radius, positions_A, positions_B, normals_B, self.auxiliar['bessel_divx'], order = self.order)
            
            self.calculator.calculate_complex_phase(self.auxiliar['sq_distance'], k, self.auxiliar['complex_phase'])
            
            self.calculator.calculate_direct_contribution_pm(self.auxiliar['bessel_divx'], self.Pref, self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], out)
            
            self.manager.erase_variable(self.auxiliar['sq_distance'], self.auxiliar['complex_phase'], self.auxiliar['bessel_divx'])

        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.calculate_direct_contribution_pm: {e}')

    def define_A_matrix (self, matrix):
        
        try:
            self.config['define_A_matrix'](matrix)

        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.define_A_matrix: {e}')

    def calculate_A_matrix (self):
        try:
            self.calculate_sm_DD_Green_function(self.manager.mesh_pos['New'], self.manager.mesh_norm['New'], self.manager.mesh_pos['New'],
                                                self.manager.mesh_area_div_4pi['New'], self.k, self.manager.rel_mesh_mesh['sm_DD_Green_function']['New']['New']) 
            
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
            print(f'Error in utils.cpu.executor.executor.calculate_A_matrix: {e}')

    def calculate_direct_contribution_emitters_mesh (self):
        try:
            if self.manager.mesh_pos['Old'] is not None:
                self.auxiliar['Total_Mesh'] = self.manager.concatenate_arrays(self.manager.mesh_pos['Old'], self.manager.mesh_pos['New'], self.VarType)
               
                self.calculate_direct_contribution_pm(self.auxiliar['Total_Mesh'], self.manager.emitters_pos, self.manager.emitters_norm,
                                                      self.k, self.manager.rel_emitters_mesh['direct_contribution_pm'])
               
            else:
                self.calculate_direct_contribution_pm(self.manager.mesh_pos['New'], self.manager.emitters_pos, self.manager.emitters_norm,
                                                      self.k, self.manager.rel_emitters_mesh['direct_contribution_pm'])
            
            if self.manager.rel_transducer_transducer['sm_DD_Green_function'] is not None:
                return self.manager.concatenate_arrays(self.manager.rel_emitters_mesh['direct_contribution_pm'], self.manager.rel_emitters_receivers['direct_contribution_pm'], self.OutVarType)
            else:
                return self.manager.rel_emitters_mesh['direct_contribution_pm']

        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.calculate_direct_contribution_emitters_mesh: {e}')

    def calculate_GF_mesh_to_receiver(self, A_calculated_with_interaction_m_r = False):
        try:
            
            if not A_calculated_with_interaction_m_r:
                if self.manager.mesh_pos['Old'] is not None:    
                    self.calculate_sm_DD_Green_function(self.manager.concatenate_arrays(self.manager.mesh_pos['Old'], self.manager.mesh_pos['New'], self.VarType)
                                                        ,self.manager.concatenate_arrays(self.manager.mesh_norm['Old'], self.manager.mesh_norm['New'], self.VarType)
                                                        , self.manager.receivers_pos,
                                                self.manager.concatenate_arrays(self.manager.mesh_area_div_4pi['Old'], self.manager.mesh_area_div_4pi['New'], self.VarType), self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])

                else:
                    self.calculate_sm_DD_Green_function(self.manager.mesh_pos['New'], self.manager.mesh_norm['New'], self.manager.receivers_pos,
                                                self.manager.mesh_area_div_4pi['New'], self.k, self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])
            #print(self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'])    
            if self.manager.rel_transducer_transducer['sm_DD_Green_function'] is not None:
                return self.manager.concatenate_arrays(self.manager.rel_receivers_receivers['sm_DD_Green_function'], self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'], self.OutVarType)
            else:
                return  self.manager.rel_receivers_mesh['sm_DD_Green_function']['from_mesh']

        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.calculate_GF_mesh_to_receiver: {e}')

    def calculate_transmission_matrix(self, mesh_pression, A_calculated_with_interaction_m_r = False):
        '''
        Not prepared to add the pression generated in the receivers. Work to be done.
        '''
        try:
            
            GF_matrix = self.calculate_GF_mesh_to_receiver(A_calculated_with_interaction_m_r)
            
            self.transmission_matrix = np.empty([self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[0] , self.manager.rel_emitters_receivers['direct_contribution_pm'].shape[1]], dtype = self.out_var_type)
            
            self.config['calculate_transmission_matrix'](self.manager.rel_emitters_receivers['direct_contribution_pm'], GF_matrix, mesh_pression, self.transmission_matrix)
            print(4)
        except Exception as e:
            print(f'Error in utils.cpu.executor.executor.calculate_transmission_matrix: {e}')









