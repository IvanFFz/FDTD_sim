from matplotlib import pyplot as plt
from numba import cuda
import numpy as np, math, cmath, os, cv2
from .calculator import optimize_blockdim

def extract_data_plane_noreturn_cuda(data, field, axis, value, amp_or_phase):
	
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
	if axis == 0:
		if i<field.shape[1] and j<field.shape[2]:
			data[i, j] = cmath.polar(field[value, i, j])[amp_or_phase]

	elif axis==1:
		if i<field.shape[0] and j<field.shape[2]:
			data[i, j] = cmath.polar(field[i, value, j])[amp_or_phase]
			
	elif axis==2:
		if i<field.shape[0] and j<field.shape[1]:
			data[i, j] = cmath.polar(field[i, j, value])[amp_or_phase]
		

class plotter():
	'''
	Configurate the plotter
	'''
	def __init__(self, mode, region, save_video, value_to_plot, ds, dt, nPoints, grid_limits, ready_to_plot=False, stream=None, var_type='float64', out_var_type = 'complex128', size = None, blockdim=(16, 16)):

		assert cuda.is_available(), 'Cuda is not available.'
		assert stream is not None, 'Cuda not configured. Stream required.'
		assert self.region['axis'] in ['X', 'Y', 'Z'], f'Incorrect axis to plot.'
		#assert os.path.exists(path), f'Path {path} not valid.'
		#assert isinstance(num_emitters, int), f'Number of emitters is not valid. Inserted {num_emitters}.'
		
		self.VarType = var_type
		self.OutVarType = out_var_type
		
		self.multiProcessorCount = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
		self.blockdim = blockdim
		self.size = size
		self.griddim = None
		
		self.config_plotter_functions()
		
		if ready_to_plot==False or ready_to_plot=="False":
			self.ready_to_plot = False
		elif ready_to_plot==True or ready_to_plot=="True":
			self.ready_to_plot = True
			
		if value_to_plot == 'amplitude':
			self.amp_or_phase = 0
		elif value_to_plot == 'phase':
			self.amp_or_phase = 1
		
		self.mode = mode
		self.region = region
		if save_video['activated'] == "True":
			self.video_name = save_video['video_name']
			self.path_to_save = save_video['path_to_save']
			assert os.path.exists(self.path_to_save), f'Path {self.path_to_save} not valid.'
			self.fps = min(save_video['fps'], math.floor(1.0/dt))
			self.video_quality = save_video['video_quality']
			
			self.plot_name = self.video_name.split('.')[0] + '\n' + mode + ' mode.'
			if mode == 'plane':
				
				if self.region['axis'] == 'X':
					self.axis = 0
				elif self.region['axis'] == 'Y':
					self.axis = 1
				elif self.region['axis'] == 'Z':
					self.axis = 2

				self.plot_name = self.plot_name + ' Plane ' + self.region['axis'] + ' = ' + self.region['value']
				x_pos = [grid_limits[0] + i*ds for i in range(nPoints)]
				y_pos = [grid_limits[0] + i*ds for i in range(nPoints)]
				
				self.init_save_video()
				
		else:
			self.video_name = None
			
										
		self.dt = dt

		self.data = None
		self.X, self.Y = np.meshgrid(x_pos, y_pos, indexing = 'ij')
		
		plt.ion()
		
		self.define_plot()
		
	
	def config_plotter_functions (self):
		try:
			self.config = {
				
				'extract_data_plane':              cuda.jit('void('+self.OutVarType+'[:,:], '+self.OutVarType+'[:,:,:], int64, int64, int64)', fastmath = True)(extract_data_plane_noreturn_cuda)
                
            }
			
		except Exception as e:
			print(f'Error in utils.cuda.executor.executor.config_functions: {e}')

	def config_plotter(self, size=None, blockdim = None, stream = None):
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
       
	
	def define_plot(self):
		try:
			
			self.figure, self.ax = plt.subplots(figsize=(10,10))
			
			if self.video_name is not None:
				self.ax.set_title (self.plot_name)
				
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.define_plot: {e}')
			
	def switch_ready_to_plot (self):
		try:
			
			if self.ready_to_plot == True:
				self.ready_to_plot = False
			elif self.ready_to_plot == False:
				self.ready_to_plot=True
				
			else:
				raise Exception(f'Something went wrong. Unexpected value of ready_to_plot {self.ready_to_plot}.')
				
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.switch_ready_to_plot: {e}')
			
	def extract_data (self, input_field):
		try:
			
			if self.mode == 'plane':
				
				self.config_manager(size=(self.data.shape[0], self.data.shape[1]), blockdim=optimize_blockdim(self.multiProcessorCount, self.data.shape[0], self.data.shape[1]))
									
				self.config['extract_data_plane'](self.data, input_field, self.axis, self.region['value'], self.amp_or_phase)

		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.extract_data: {e}')
			
	def plot_plane(self):
		try:
			
			if self.ready_to_plot:
				data = self.data.copy_to_host(stream=self.stream)
				if self.amp_or_phase == 0:
					im = self.ax.pcolormesh(self.X, self.Y, data, cmap='PuRd')
					
				elif self.amp_or_phase == 1:
					im = self.ax.pcolormesh(self.X, self.Y, data, cmap='hsv')
					
				self.figure.colorbar(im, ax=self.ax)
				
				self.figure.canvas.draw()
				plt.pause(0.01)
					
			else:
				pass

		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.plot_plane: {e}')
			
	def init_save_video(self):
		try:
			
			fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
			self.out = cv2.VideoWriter(self.video_name, fourcc, self.fps, (self.video_quality[0], self.video_quality[1]))

		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.init_save_video: {e}')
			
	def save_to_video(self):
		try:
			
			if self.ready_to_plot and self.video_name is not None:
				farr = np.array(self.figure.canvas.renderer._renderer)
				farr = cv2.resize(farr, (self.video_quality[0], self.video_quality[1]))
				bgr = cv2.cvtColor(farr, cv2.COLOR_RGBA2BGR)
				self.out.write(bgr)
					
			else:
				pass

		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.save_to_video: {e}')
			
	def finish_saving_video(self):
		try:
			
			if self.ready_to_plot:
				self.out.release()					
			else:
				pass
			
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.finish_saving_video: {e}')
			
	def record(self, input_field, time, ratio_times):
		try:
			
			if self.is_frame(time, ratio_times):
				self.extract_data(input_field)
			
				self.plot_plane()
				self.save_to_video()
				
			else:
				pass
						
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.finish_saving_video: {e}')
			
	def finish(self):
		try:
			
			self.finish_saving_video()
			self.switch_ready_to_plot()
			
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.finish_saving_video: {e}')
			
	def is_frame(self, time, ratio_times):
		try:
			
			in_ratio = ratio_times*time - math.floor(ratio_times*time)
			eq_to_frame = 1.0/self.fps

			for i in range(self.fps):
				if in_ratio > i*eq_to_frame - self.dt and in_ratio < i*eq_to_frame + self.dt:
					return True
				
			return False
			
		except Exception as e:
			print(f'Error in utils.cuda.plotter.plotter.finish_saving_video: {e}')

			
	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cpu.plotter.plotter.erase_variable: {e}')
