from pandas.core.arrays.arrow import dtype
import numpy as np
from numpy import float32 as f32, float64 as f64, complex64 as c64, complex128 as c128

import numba, math, cmath
from numba import cuda
from .calculator import calculator_cuda, calculate_bd_b2 as to_b2, optimize_blockdim
from .manager import manager_cuda
from .solver import solver_cuda
from .loader import loader_cuda
from .plotter import plotter

     
def copy_auxiliar_variables_noreturn_cuda (aux_pressure, pressure, aux_vx, vx, aux_vy, vy, aux_vz, vz):
    '''
	Function that saves the data of the mesh pressions
	'''
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    
    if i<pressure.shape[0] and j<pressure.shape[1] and k<pressure.shape[2]:
        pressure[i, j, k] = aux_pressure[i, j, k]
        
        vx[i, j, k] = aux_vx[i, j, k]
        vy[i, j, k] = aux_vy[i, j, k]
        vz[i, j, k] = aux_vz[i, j, k]



class executor ():
    
    def __init__ (self, config_path, print_config = False, size=None, blockdim=(16,16)):
        
        assert cuda.is_available(), 'Cuda is not available.'
        #assert stream is not None, 'Cuda not configured. Stream required.'
        
        ## In the future we can modify it to use different transducers with different Pref's.
        #self.Pref = Pref
        
        #if var_type == 'float32':
        #    self.var_type = f32
        #elif var_type == 'float64':
        #    self.var_type = f64
        #else:
        #    raise Exception (f'Bad type selected {var_type}')
        #
        self.var_type = None
        self.VarType = None
        #
        #if out_var_type == 'complex64':
        #    self.out_var_type = c64
        #elif out_var_type == 'complex128':
        #    self.out_var_type = c128
        #else:
        #    raise Exception (f'Bad type selected {out_var_type}')
        #
        self.out_var_type = None
        self.OutVarType = None
        

        self.calculator = None
        self.manager = None
        self.loader = None
        self.solver = None
        
        self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
        self.blockdim = blockdim
        self.size = size
        self.griddim = None
        ##########
        self.init_workspace(config_path, print_config)
        self.config_executor_functions()
        
        #self.auxiliar = {
        #            
        #            'vx':		        	        None,
        #            'vy':		        	        None,			
        #            'vz':		        	        None,
        #            'pressure':			            None,
        #            'Total_Mesh':                           None,
        #            'direct_contribution_emitter_mesh':     None,
        #            'mesh_pression_emitter':                None
        #    }
        
        self.time = 0.0
      
    def config_executor_functions (self):
        try:
            self.config = {
                
				'copy_auxiliar_variables':              cuda.jit('void('+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:]'
                                                                        +self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:], '+self.OutVarType+'[:,:,:])', fastmath = True)(copy_auxiliar_variables_noreturn_cuda)
            
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
            
    def config_precission(self, precission):
        try:
            
            if precission[0] == 'float32':
                self.var_type = f32
            elif precission[0] == 'float64':
                self.var_type = f64
            else:
                raise Exception (f'Bad type selected {precission[0]}')
        
            self.VarType = precission[0]
        
            if precission[1] == 'complex64':
                self.out_var_type = c64
            elif precission[1] == 'complex128':
                self.out_var_type = c128
            else:
                raise Exception (f'Bad type selected {precission[1]}')
        
            self.OutVarType = precission[1]
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.config_precission: {e}')
		     
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
            
    def config_simulation(self, dt, ds, nPoints, density, c, grid_limits, times, ratio_times):
        try:
            
            self.dt = dt
            self.ds = ds
            self.nPoints = nPoints
            self.density = density
            self.c = c
            self.grid_limits = [grid_limits, grid_limits + ds*(nPoints-1)]
            self.key_times = times
            self.ratio_times = ratio_times
                        
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.config_simulation: {e}')

    def init_workspace (self, config_path, print_config = False):
        try:
            
            self.stream = cuda.stream()
            
            self.loader = loader_cuda(config_path, print_config, stream = self.stream, var_type = self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)
            self.config_precission(self.loader.configuration['precission'])
            self.config_simulation(self.loader.configuration['grid']['sim_parameters']['dt'],
                                   self.loader.configuration['grid']['sim_parameters']['ds'],
                                   self.loader.configuration['grid']['sim_parameters']['nPoints'],
                                   self.loader.configuration['grid']['sim_parameters']['density'],
                                   self.loader.configuration['grid']['sim_parameters']['c'],
                                   self.loader.configuration['grid']['boundary']['grid_limits_min'],
                                   self.loader.configuration['times'],
                                   self.loader.configuration['ratio_sim_plot_times'])
            
            self.config_geometries(self.loader.configuration['grid']['boundary']['layer_thickness'],
                                   self.loader.configuration['grid']['boundary']['max_object_distance'],
                                   self.loader.configuration['grid']['sim_parameters']['airAbsorptivity'])
              
            self.config_executor(blockdim=self.blockdim, stream=self.stream)
        
            self.calculator = calculator_cuda(stream = self.stream, var_type = self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)
        
            self.manager = manager_cuda(stream = self.stream, var_type = self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)
            
            plotter_configuration = self.loader.load_plotter_configuration()
            if plotter_configuration is not None:
                self.plotter = plotter(plotter_configuration['mode'], plotter_configuration['region'], plotter_configuration['save_video'],plotter_configuration['value_to_plot'],
                                       self.ds, self.dt, self.nPoints, self.grid_limits, plotter_configuration['ready_to_plot'], 
                                       stream = self.stream, var_type=self.VarType, out_var_type = self.OutVarType, blockdim = self.blockdim)

            self.init_grid()
            
            self.fill_grid()
		
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.init_workspace: {e}')
 
    def init_grid(self):
        try:
            
            self.manager.pressure = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            
            self.manager.velocity_x = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.velocity_y = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.velocity_z = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            
            self.manager.emitters_amplitude = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.emitters_frequency = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.emitters_phase = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            self.manager.velocity_b = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.out_var_type), stream = self.stream)
            
            self.manager.geometry_field = cuda.to_device(np.ones((self.nPoints, self.nPoints, self.nPoints), dtype = self.var_type), stream = self.stream) #beta in the paper, init to air (1)
            if self.airAbsorptivity == 0.0:
                self.manager.absorptivity = cuda.to_device(np.zeros((self.nPoints, self.nPoints, self.nPoints), dtype = self.var_type), stream = self.stream) #sigma in the paper
            else:
                self.manager.absorptivity = cuda.to_device(self.airAbsorptivity * np.ones((self.nPoints, self.nPoints, self.nPoints), dtype = self.var_type), stream = self.stream) #sigma in the paper
                          
            self.manager.PML_limit_volume(self.manager.absorptivity, self.maxDistEffect, self.maxAbsorptivity, self.airAbsorptivity)
            
            self.plotter.data = cuda.device_array((self.nPoints, self.nPoints), dtype = self.var_type, stream =self.stream)

            #self.grid = cuda.device_array((positions_B.shape[0],positions_A.shape[0]), dtype = self.var_type, stream = self.stream)
		
        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.init_grid: {e}')

    def fill_grid(self):
        try:
            
            self.loader.load_transducers()
            
            self.loader.load_objects()

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.fill_grid: {e}')
            
    def simulation_step (self):
        try:
            
            self.calculator.step_velocity_values(self.manager.velocity_x, self.manager.velocity_b, self.manager.pressure,
                                                 self.manager.geometry_field, self.manager.absorptivity, self.dt, self.ds,
                                                 self.density, axis = 0)
            self.calculator.step_velocity_values(self.manager.velocity_y, self.manager.velocity_b, self.manager.pressure,
                                                 self.manager.geometry_field, self.manager.absorptivity, self.dt, self.ds,
                                                 self.density, axis = 1)
            self.calculator.step_velocity_values(self.manager.velocity_z, self.manager.velocity_b, self.manager.pressure,
                                                 self.manager.geometry_field, self.manager.absorptivity, self.dt, self.ds,
                                                 self.density, axis = 2)
            
            self.calculator.set_velocity_emitters(self.manager.velocity_b, self.manager.emitters_amplitude, self.manager.emitters_frequency, self.manager.emitters_phase, self.time)
            
            self.calculator.step_pressure_values(self.manager.pressure, self.manager.velocity_x, self.manager.velocity_y,
                                                 self.manager.velocity_z, self.manager.geometry_field, self.manager.absorptivity,
                                                 self.dt, self.ds, self.density*self.c**2)

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.simulation_step: {e}')
            
    def execute(self):
        try:
            
            while self.time <= self.key_times['thermalization']:
                
                self.simulation_step()
                
                self.time = self.time + self.dt
                
            recording = False
                
            while self.time - self.key_times['thermalization'] <= self.key_times['simulation']:
                
                self.simulation_step()
                
                if self.time - self.key_times['thermalization'] >= self.key_times['record'][0] and self.time - self.key_times['thermalization'] <= self.key_times['record'][1]:
                    
                    if not recording:
                        recording = True
                        self.plotter.switch_ready_to_plot()
                        
                    self.plotter.record(self.manager.pressure, self.time - self.key_times['thermalization'] - self.key_times['record'][0], self.ratio_times)
                    
                else:
                    
                    if recording:
                        recording = False
                        self.plotter.switch_ready_to_plot()
                        
                self.time = self.time + self.dt
            

        except Exception as e:
            print(f'Error in utils.cuda.executor.executor.execute: {e}')
            





