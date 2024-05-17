import numpy as np, numba, math, cmath
from numba import njit, prange

				
'''
Implement functions used to manipulate arrays using cuda
'''

def deal_transducers_to_emitters_no_return_cpu (sel_em, trans_pos, trans_norm, em_pos, em_norm):
	
	for i in prange(trans_pos.shape[0]):
		for e in prange (sel_em.shape[0]):
			#for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				em_pos[e, 0] = trans_pos[i, 0]
				em_pos[e, 1] = trans_pos[i, 1]
				em_pos[e, 2] = trans_pos[i, 2]
				
				em_norm[e, 0] = trans_norm[i, 0]
				em_norm[e, 1] = trans_norm[i, 1]
				em_norm[e, 2] = trans_norm[i, 2]
				
		
def deal_transducers_to_receivers_no_return_cpu (sel_em, trans_pos, trans_norm, rec_pos, rec_norm):
	
	for i in prange(trans_pos.shape[0]):
		for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				break
			elif e == sel_em.shape[0]-1:
				if i > sel_em[e]:
					rec_pos[i-e-1,0] = trans_pos[i,0]
					rec_pos[i-e-1,1] = trans_pos[i,1]
					rec_pos[i-e-1,2] = trans_pos[i,2]
				
					rec_norm[i-e-1,0] = trans_norm[i,0]
					rec_norm[i-e-1,1] = trans_norm[i,1]
					rec_norm[i-e-1,2] = trans_norm[i,2]
					
				elif i < sel_em[0]:
					rec_pos[i,0] = trans_pos[i,0]
					rec_pos[i,1] = trans_pos[i,1]
					rec_pos[i,2] = trans_pos[i,2]
				
					rec_norm[i,0] = trans_norm[i,0]
					rec_norm[i,1] = trans_norm[i,1]
					rec_norm[i,2] = trans_norm[i,2]
					
				else:
					for ee in range(e):
						if i > sel_em[ee] and i < sel_em[ee+1]:#sel_em.shape[0]-1 and i < sel_em[e+1]) or (e == sel_em.shape[0] -1)):
							rec_pos[i-ee-1,0] = trans_pos[i,0]
							rec_pos[i-ee-1,1] = trans_pos[i,1]
							rec_pos[i-ee-1,2] = trans_pos[i,2]
				
							rec_norm[i-ee-1,0] = trans_norm[i,0]
							rec_norm[i-ee-1,1] = trans_norm[i,1]
							rec_norm[i-ee-1,2] = trans_norm[i,2]
					
				
					
def deal_rel_transducers_to_rel_receivers_no_return_cpu (sel_em, rel_t_t_GF, rel_r_r_GF):
	for i in prange(rel_t_t_GF.shape[0]):
		for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				break
			
			elif e == sel_em.shape[0]-1:
				if i > sel_em[e]:
				
					for j in range(sel_em[0]):
					
						rel_r_r_GF[i-e-1,j] = rel_t_t_GF[i,j]
				
					for n in range(e):
					
						for j in range(sel_em[n]+1, sel_em[n+1]):
						
							rel_r_r_GF[i-e,j-n] = rel_t_t_GF[i,j]
						
					for k in range(sel_em[e], rel_t_t_GF.shape[1]):
					
						rel_r_r_GF[i-sel_em.shape[0] +1,k-e] = rel_t_t_GF[i,k]
					
				elif i < sel_em[0]:
				
					for j in range(sel_em[0]):
					
						rel_r_r_GF[i,j] = rel_t_t_GF[i,j]
				
					for n in range(e):
					
						for j in range(sel_em[n]+1, sel_em[n+1]):
						
							rel_r_r_GF[i,j-n] = rel_t_t_GF[i,j]
						
					for k in range(sel_em[e], rel_t_t_GF.shape[1]):
					
						rel_r_r_GF[i,k-e] = rel_t_t_GF[i,k]
					
				else:
					for ee in range(e):
					
						if i > sel_em[ee] and i < sel_em[ee+1]:#sel_em.shape[0]-1 and i < sel_em[e+1]) or (e == sel_em.shape[0] -1)):
						
							for j in range(sel_em[0]):
					
								rel_r_r_GF[i -ee ,j] = rel_t_t_GF[i,j]
				
							for n in range(e):
					
								for j in range(sel_em[n]+1, sel_em[n+1]):
						
									rel_r_r_GF[i-ee,j-n] = rel_t_t_GF[i,j]
						
							for k in range(sel_em[e], rel_t_t_GF.shape[1]):
					
								rel_r_r_GF[i-ee,k-e] = rel_t_t_GF[i,k]
					
		
def deal_rel_transducers_to_rel_emitters_no_return_cpu (sel_em, rel_t_t_DC, rel_e_r_DC):
	for i in prange(rel_t_t_DC.shape[0]):
		for e in prange(sel_em.shape[0]):
			#for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				for k in range(sel_em[0]):
					rel_e_r_DC[e,k] = rel_t_t_DC[i,k]
				
				for n in range(sel_em.shape[0]-1):#sel_em.shape[0]-1):
					for j in range(sel_em[n]+1, sel_em[n+1]):
						rel_e_r_DC[e,j-n-1] = rel_t_t_DC[i,j]
					
				for k in range(sel_em[sel_em.shape[0]-1], rel_t_t_DC.shape[1]):#sel_em.shape[0]-1], rel_t_t_DC.shape[1]):
					rel_e_r_DC[e,k-sel_em.shape[0]] = rel_t_t_DC[i,k]#sel_em.shape[0]+1] = rel_t_t_DC[i,k]
						

	
def deal_transducers_noreturn_cpu (sel_em, trans_pos, trans_norm, rel_t_t_GF, rel_t_t_DC, em_pos, em_norm, rec_pos, rec_norm, rel_e_r_DC, rel_r_r_GF):
	'''
	Function that calculates the SQUARE of the distance
	'''
		
	for i in prange(trans_pos.shape[0]):
		for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				em_pos[e,0] = trans_pos[i,0]
				em_pos[e,1] = trans_pos[i,1]
				em_pos[e,2] = trans_pos[i,2]
				
				em_norm[e,0] = trans_norm[i,0]
				em_norm[e,1] = trans_norm[i,1]
				em_norm[e,2] = trans_norm[i,2]
				
				for n in range(sel_em.shape[0]-1):#sel_em.shape[0]-1):
					for j in range(sel_em[n]+1, sel_em[n+1]):
						rel_e_r_DC[e,j-n] = rel_t_t_DC[i,j]
				for k in range(sel_em[sel_em.shape[0]-1], rel_t_t_DC.shape[1]):#sel_em.shape[0]-1], rel_t_t_DC.shape[1]):
					rel_e_r_DC[e,k-sel_em.shape[0]+1] = rel_t_t_DC[i,k]#sel_em.shape[0]+1] = rel_t_t_DC[i,k]
										
			elif e == sel_em.shape[0]-1: 
				for ee in range(sel_em.shape[0]):
					if (ee == sel_em.shape[0] -1 ) and i > sel_em[ee]:
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]
					
						for nn in range(sel_em.shape[0]-1):
							for jj in range(sel_em[nn]+1, sel_em[nn+1]):
								rel_r_r_GF[i-ee,jj-nn] = rel_t_t_GF[i,jj]
						for kk in range(sel_em[sel_em.shape[0]-1], rel_t_t_GF.shape[1]):
							rel_r_r_GF[i-ee,kk-sel_em.shape[0]+1] = rel_t_t_GF[i,kk]
											
					elif i > sel_em[ee] and (ee < sel_em.shape[0]-1 and i < sel_em[ee+1]):
					#sel_em.shape[0]-1 and i < sel_em[e+1]) or (e == sel_em.shape[0] -1)):
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]

						for nn in range(sel_em.shape[0]-1):
							for jj in range(sel_em[nn]+1, sel_em[nn+1]):
								rel_r_r_GF[i-ee,jj-nn] = rel_t_t_GF[i,jj]
						for kk in range(sel_em[sel_em.shape[0]-1], rel_t_t_GF.shape[1]):
							rel_r_r_GF[i-ee,kk-sel_em.shape[0]+1] = rel_t_t_GF[i,kk]
												
					elif i < sel_em[ee] and ee==0:
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]

						for nn in range(sel_em.shape[0]-1):
							for jj in range(sel_em[nn]+1, sel_em[nn+1]):
								rel_r_r_GF[i-ee,jj-nn] = rel_t_t_GF[i,jj]
						for kk in range(sel_em[sel_em.shape[0]-1], rel_t_t_GF.shape[1]):
							rel_r_r_GF[i-ee,kk-sel_em.shape[0]+1] = rel_t_t_GF[i,kk]
						
								
def deal_transducers_no_receivers_rel_noreturn_cpu (sel_em, trans_pos, trans_norm, rel_t_t_DC, em_pos, em_norm, rec_pos, rec_norm, rel_e_r_DC):
	'''
	Function that calculates the SQUARE of the distance
	'''
		
	for i in prange(trans_pos.shape[0]):
		for e in range(sel_em.shape[0]):
			if i == sel_em[e]:
				em_pos[e,0] = trans_pos[i,0]
				em_pos[e,1] = trans_pos[i,1]
				em_pos[e,2] = trans_pos[i,2]
				
				em_norm[e,0] = trans_norm[i,0]
				em_norm[e,1] = trans_norm[i,1]
				em_norm[e,2] = trans_norm[i,2]
				
				for n in range(sel_em.shape[0]-1):#sel_em.shape[0]-1):
					for j in range(sel_em[n]+1, sel_em[n+1]):
						rel_e_r_DC[e,j-n] = rel_t_t_DC[i,j]
				for k in range(sel_em[sel_em.shape[0]-1], rel_t_t_DC.shape[1]):#sel_em.shape[0]-1], rel_t_t_DC.shape[1]):
					rel_e_r_DC[e,k-sel_em.shape[0]+1] = rel_t_t_DC[i,k]#sel_em.shape[0]+1] = rel_t_t_DC[i,k]
					
				break
					
			elif e == sel_em.shape[0]-1: 
				for ee in range(sel_em.shape[0]):
					if (ee == sel_em.shape[0] -1 ) and i > sel_em[ee]:
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]
					
						break 
				
					elif i > sel_em[ee] and ee>0 and ee < sel_em.shape[0]-1 and i < sel_em[ee+1]:#sel_em.shape[0]-1 and i < sel_em[e+1]) or (e == sel_em.shape[0] -1)):
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]
					
						break
				
					elif i < sel_em[ee] and ee==0:
						rec_pos[i-ee,0] = trans_pos[i,0]
						rec_pos[i-ee,1] = trans_pos[i,1]
						rec_pos[i-ee,2] = trans_pos[i,2]
				
						rec_norm[i-ee,0] = trans_norm[i,0]
						rec_norm[i-ee,1] = trans_norm[i,1]
						rec_norm[i-ee,2] = trans_norm[i,2]
					
						break
	
def concatenate_arrays_delete_rows_noreturn_cpu (old, new, delete_rows, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	for tx in prange(old.shape[0] + new.shape[0]):
		for ty in prange(old.shape[1]):
			for tz, value in enumerate(delete_rows):
				if (tz == 0 and tx < delete_rows[0]) or (
				tz > 0 and tz <= delete_rows.shape[0] - 1 and tx < value - tz and tx > delete_rows[tz-1] - tz):
					if tx < old.shape[0]-tz:
						#ab[tx]=0
                
						out[tx,ty] = old[tx+tz,ty]
					else:
						#ab[tx]= old.shape[0]
                
						out[tx,ty] = new[tx-old.shape[0]+tz,ty]
				elif (tz == delete_rows.shape[0]-1 and tx > value - tz -1):
					if tx < old.shape[0]-tz:
						#ab[tx]=0
                
						out[tx,ty] = old[tx+tz+1,ty]
					else:
						#ab[tx]= old.shape[0]
                
						out[tx,ty] = new[tx-old.shape[0]+tz+1,ty]
	
def concatenate_arrays_noreturn_cpu (old, new, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	for tx in prange(old.shape[0] + new.shape[0]):
		for ty in prange(old.shape[1]):
			if tx < old.shape[0]:
				#ab[tx]=0
                
				out[tx,ty] = old[tx,ty]
			else:
				#ab[tx]= old.shape[0]
                
				out[tx,ty] = new[tx-old.shape[0],ty]
			
def concatenate_matrix_noreturn_cpu (old_old, old_new, new_old, new_new, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	for tx in prange(old_old.shape[0] + new_new.shape[0]):
		for ty in prange(old_old.shape[1] + new_new.shape[1]):
			if tx < old_old.shape[0] and ty<old_old.shape[1]:
							
				out[tx,ty] = old_old[tx,ty]
			elif tx < old_old.shape[0] and ty>=old_old.shape[1]:
							
				out[tx,ty] = old_new[tx,ty-old_old.shape[1]]
    
			elif tx >= old_old.shape[0] and ty<old_old.shape[1]:
							
				out[tx,ty] = new_old[tx-old_old.shape[0],ty]
                        
			elif tx >= old_old.shape[0] and ty>=old_old.shape[1]:
							
				out[tx,ty] = new_new[tx-old_old.shape[0],ty-old_old.shape[1]]
                        				
def concatenate_matrix_delete_rowsncols_noreturn_cpu (old_old, old_new, new_old, new_new, delete_rowsncols, out):
		
	'''
	Function that concatenates two arrays of vectors deleting those vectors we have selected.
	'''
		
	for tx in prange(old_old.shape[0] + new_new.shape[0]):
		for ty in prange(old_old.shape[1] + new_new.shape[1]):
			for tz, value in enumerate(delete_rowsncols):
				if (tz == 0 and tx < delete_rowsncols[0]) or (
					tz > 0 and tz <= delete_rowsncols.shape[0] - 1 and tx < value - tz and tx > delete_rowsncols[tz-1] - tz):

					for tzz, valuee in enumerate(delete_rowsncols):
						if (tzz == 0 and ty < delete_rowsncols[0]) or (
						tzz > 0 and tzz <= delete_rowsncols.shape[0] - 1 and ty < valuee - tzz and ty > delete_rowsncols[tzz-1] - tzz):
                    
							if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = old_old[tx+tz,ty+tzz]
							elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = old_new[tx+tz,ty-old_old.shape[1]+tzz]
    
							elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = new_old[tx-old_old.shape[0]+tz,ty+tzz]
                        
							elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = new_new[tx-old_old.shape[0]+tz,ty-old_old.shape[1]+tzz]
                        
						elif (tzz == delete_rowsncols.shape[0]-1 and ty > value - tzz - 1):
							if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = old_old[tx+tz,ty+tzz+1]
							elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = old_new[tx+tz,ty-old_old.shape[1]+tzz+1]
							elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = new_old[tx-old_old.shape[0]+tz,ty+tzz+1]
							elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = new_new[tx-old_old.shape[0]+tz,ty-old_old.shape[1]+tzz+1]
                    
				elif (tz == delete_rowsncols.shape[0]-1 and tx > value - tz -1):
					for tzz, valuee in enumerate(delete_rowsncols):
						if (tzz == 0 and ty < delete_rowsncols[0]) or (
						tzz > 0 and tzz <= delete_rowsncols.shape[0] - 1 and ty < valuee - tzz and ty > delete_rowsncols[tzz-1] - tzz):
                    
							if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = old_old[tx+tz+1,ty+tzz]
							elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = old_new[tx+tz+1,ty-old_old.shape[1]+tzz]
							elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = new_old[tx-old_old.shape[0]+tz+1,ty+tzz]
							elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = new_new[tx-old_old.shape[0]+tz+1,ty-old_old.shape[1]+tzz]
						elif (tzz == delete_rowsncols.shape[0]-1 and ty > valuee - tzz - 1):
							if tx < old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = old_old[tx+tz+1,ty+tz+1]
							elif tx < old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = old_new[tx+tz+1,ty-old_old.shape[1]+tzz+1]
							elif tx >= old_old.shape[0]-tz and ty<old_old.shape[1] - tzz:
							
								out[tx,ty] = new_old[tx-old_old.shape[0]+tz+1,ty+tzz+1]
							elif tx >= old_old.shape[0]-tz and ty>=old_old.shape[1] - tzz:
							
								out[tx,ty] = new_new[tx-old_old.shape[0]+tz+1,ty-old_old.shape[1]+tzz+1]
	

class manager_cpu():
	
	def __init__ (self, var_type='float64', out_var_type = 'complex128'):
		
		self.VarType = var_type
		self.OutVarType = out_var_type
				
		self.config = None

		self.config_manager_functions()
		
		self.transducers_pos_cuda = None
		self.transducers_norm_cuda = None
		self.transducers_radius = None
		
		self.emitters_pos = None
		self.emitters_norm = None
		
		self.receivers_pos = None
		self.receivers_norm = None
		
		self.mesh_pos = { 'Old': None, 'New': None }
		self.mesh_norm = { 'Old': None, 'New': None }
		self.mesh_area_div_4pi = { 'Old': None, 'New': None }
		self.mesh_pression = { 'Old': None, 'New': None }
		
		self.rel_transducer_transducer = { 
                            'sm_DD_Green_function':	None,
                            'direct_contribution_pm': None
		}

		# Defined when preprocessing the transducers
		self.rel_emitters_receivers = { 
                            #'sm_DD_Green_function':	None,
                            'direct_contribution_pm': None
		}
		self.rel_receivers_receivers = { 
							'sm_DD_Green_function':	None #,
							#'direct_contribution_pm': None
		}
		
		# Both three must be initialized when preprocessing, once we know the sizes of the
		self.rel_emitters_mesh = { 
                            #'sq_distance':				None, #{ 'Old': None, 'New': None },                  
                            #'cos_angle_sn':				None, #{ 'Old': None, 'New': None },
                            #'sin_angle_np':				{ 'Old': None, 'New': None },
                            #'bessel_divx':				None, #{ 'Old': None, 'New': None },
                            #'sm_DD_Green_function':		{ 'Old': None, 'New': None },
                            'direct_contribution_pm':	None #, #{ 'Old': None, 'New': None },
                            #'complex_phase':			None #{ 'Old': None, 'New': None }
        }
		self.rel_receivers_mesh = { 
                            #'sq_distance':				None, #{ 'Old': None, 'New': None },                  
                            #'cos_angle_sn':				None, #{ 'Old': None, 'New': None },
                            #'sin_angle_np':				{ 'Old': None, 'New': None },
                            #'bessel_divx':				None, #{ 'Old': None, 'New': None },
                            'sm_DD_Green_function':		{ 'from_receivers': None, 'from_mesh': None },
                            #'direct_contribution_pm':	{ 'Old': None, 'New': None },
                            #'complex_phase':			None #{ 'Old': None, 'New': None }
        }
		
		self.rel_mesh_mesh = { 
                            #'sq_distance':				{ 'Old': { 'New': None }, 'New': { 'Old': None, 'New': None } },                  
                            #'cos_angle_sn':				{ 'Old': { 'New': None }, 'New': { 'Old': None, 'New': None } },  
                            #'sin_angle_np':				{ 'Old': { 'Old': None, 'New': None }, 'New': { 'Old': None, 'New': None } },
                            #'bessel_divx':				{ 'Old': { 'New': None }, 'New': { 'Old': None, 'New': None } },  
                            'sm_DD_Green_function':		{ 'Old': { 'Old': None, 'New': None }, 'New': { 'Old': None, 'New': None } }#,
                            #'direct_contribution_pm':	{ 'Old': { 'Old': None, 'New': None }, 'New': { 'Old': None, 'New': None } },
                            #'complex_phase':			{ 'Old': { 'New': None }, 'New': { 'Old': None, 'New': None } }
        }
        
	def config_manager_functions (self):
		try:
			self.config = {
				'deal_transducers':									njit('void(int64[::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '
																				+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '
																				+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(deal_transducers_noreturn_cpu),
				'deal_transducers_no_receivers_rel':				njit('void(int64[::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.OutVarType+'[:,::1], '
																				+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '
																				+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(deal_transducers_no_receivers_rel_noreturn_cpu),
				'deal_transducers_to_emitters':						njit('void(int64[::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(deal_transducers_to_emitters_no_return_cpu),
				'deal_transducers_to_receivers':					njit('void(int64[::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(deal_transducers_to_receivers_no_return_cpu),
				'deal_rel_transducers_to_rel_emitters':				njit('void(int64[::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(deal_rel_transducers_to_rel_emitters_no_return_cpu),
				'deal_rel_transducers_to_rel_receivers':			njit('void(int64[::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(deal_rel_transducers_to_rel_receivers_no_return_cpu),
				'concatenate_vartype_arrays_delete_rows':			njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], int64[::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_arrays_delete_rows_noreturn_cpu),
				'concatenate_outvartype_arrays_delete_rows':		njit('void('+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], int64[::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_arrays_delete_rows_noreturn_cpu),
				'concatenate_vartype_arrays':						njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_arrays_noreturn_cpu),
				'concatenate_outvartype_arrays':					njit('void('+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_arrays_noreturn_cpu),
				'concatenate_vartype_matrix_delete_rowsncols':		njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], int64[::1], '
																				+self.VarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_matrix_delete_rowsncols_noreturn_cpu),
				'concatenate_outvartype_matrix_delete_rowsncols':	njit('void('+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], int64[::1], '
																				+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_matrix_delete_rowsncols_noreturn_cpu),
				'concatenate_vartype_matrix':						njit('void('+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '+self.VarType+'[:,::1], '
																				+self.VarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_matrix_noreturn_cpu),
				'concatenate_outvartype_matrix':					njit('void('+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '+self.OutVarType+'[:,::1], '
																				+self.OutVarType+'[:,::1])', parallel=True, fastmath = True)(concatenate_matrix_noreturn_cpu)
				}
			
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cuda.config_manager_functions: {e}')

	'''
	Implement functions that initializes the containers of the arrays.
	'''

	def locate_transducers (self, transducers_pos, transducers_norm, transducers_radius):
		try:
			assert (type(transducers_pos) == np.ndarray and type(transducers_norm) == np.ndarray), 'Emitters must be defined as numpy array.'
			
			self.transducers_pos_cuda = transducers_pos
			self.transducers_norm_cuda = transducers_norm
			#All transducers have the same radius. Can be change to consider different transducers.
			self.transducers_radius = transducers_radius
			
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.locate_transducers: {e}')
				
	def init_new_mesh_rel(self, number_elements):
		'''
		After this, stream.sychronize()
		'''
		try:
			if self.mesh_pos['Old'] is not None:
				self.rel_emitters_mesh['direct_contribution_pm'] = np.empty([self.emitters_pos.shape[0],self.mesh_pos['Old'].shape[0]+self.mesh_pos['New'].shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				self.rel_receivers_mesh['sm_DD_Green_function']['from_receivers'] = np.empty([self.mesh_pos['Old'].shape[0]+self.mesh_pos['New'].shape[0], self.receivers_pos.shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				self.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'] = np.empty([self.receivers_pos.shape[0], self.mesh_pos['Old'].shape[0]+self.mesh_pos['New'].shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				self.rel_mesh_mesh['sm_DD_Green_function']['New']['New'] = np.empty([number_elements,number_elements],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				
			else:
				self.rel_emitters_mesh['direct_contribution_pm'] = np.empty([self.mesh_pos['New'].shape[0], self.emitters_pos.shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				self.rel_receivers_mesh['sm_DD_Green_function']['from_receivers'] = np.empty([self.mesh_pos['New'].shape[0], self.receivers_pos.shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				
				self.rel_receivers_mesh['sm_DD_Green_function']['from_mesh'] = np.empty([self.receivers_pos.shape[0], self.mesh_pos['New'].shape[0]],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
				self.rel_mesh_mesh['sm_DD_Green_function']['New']['New'] = np.empty([number_elements,number_elements],
																					dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
							
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.init_new_mesh_rel: {e}')
		
	def init_mesh(self, number_elements):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			self.mesh_pos['New'] = np.random.rand(number_elements, 3)
			
			self.mesh_norm['New'] = np.random.rand(number_elements, 3)
			
			self.mesh_area_div_4pi['New'] = np.random.rand(number_elements, 1)
			
			self.mesh_pression['New'] = np.random.rand(number_elements, self.emitters_pos.shape[0]) + 1j*np.random.rand(number_elements, self.emitters_pos.shape[0])
			
			self.init_new_mesh_rel(number_elements)
			
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.init_mesh: {e}')

	'''
	Implement functions that manipulate arrays using cuda
	'''
			
	def assign_emitters (self, selector_emitters, receivers_interacts = False):
		try:
			assert (type(selector_emitters) == np.ndarray), 'Emitters must be selected with a numpy array.'
			assert (selector_emitters.shape[0]<=self.transducers_pos_cuda.shape[0] 
					and int(max(selector_emitters)) < self.transducers_pos_cuda.shape[0] ), 'Emitters selected incorrectly.'
			
			self.selector_emitters = selector_emitters
			self.emitters_pos = np.empty([self.selector_emitters.shape[0],self.transducers_pos_cuda.shape[1]], dtype = self.transducers_pos_cuda.dtype)
			self.emitters_norm = np.empty([self.selector_emitters.shape[0],self.transducers_norm_cuda.shape[1]], dtype = self.transducers_norm_cuda.dtype)
			#self.emitters_radius = self.transducers_radius
			
			self.receivers_pos = np.empty([self.transducers_pos_cuda.shape[0]-self.selector_emitters.shape[0],self.transducers_pos_cuda.shape[1]], dtype = self.transducers_pos_cuda.dtype)
			self.receivers_norm = np.empty([self.transducers_norm_cuda.shape[0]-self.selector_emitters.shape[0],self.transducers_norm_cuda.shape[1]], dtype = self.transducers_norm_cuda.dtype)
			
			#self.receivers_pression = cuda.device_array((self.transducers_norm_cuda.shape[0]-len(selector_emitters),self.transducers_norm_cuda.shape[1]), dtype = self.transducers_norm_cuda.dtype, stream = self.stream)

			self.rel_emitters_receivers['direct_contribution_pm'] = np.empty([self.selector_emitters.shape[0],self.transducers_pos_cuda.shape[0]-self.selector_emitters.shape[0]],
																			 dtype = self.rel_transducer_transducer['direct_contribution_pm'].dtype)
			if receivers_interacts:
				self.rel_receivers_receivers['sm_DD_Green_function'] = np.empty([self.transducers_pos_cuda.shape[0]-self.selector_emitters.shape[0],self.transducers_pos_cuda.shape[0]-self.selector_emitters.shape[0]],
																			 dtype = self.rel_transducer_transducer['sm_DD_Green_function'].dtype)

			
			self.deal_transducers(receivers_interacts)#selector_emitters, receivers_interacts)
			
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.assign_emitters: {e}')
				
	def deal_transducers(self, receivers_interacts = False):#selector_emitters, receivers_interacts = False):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			self.config['deal_transducers_to_emitters'](
				self.selector_emitters, self.transducers_pos_cuda, self.transducers_norm_cuda,
				self.emitters_pos, self.emitters_norm)
			
			self.config['deal_rel_transducers_to_rel_emitters'](self.selector_emitters, self.rel_transducer_transducer['direct_contribution_pm'], self.rel_emitters_receivers['direct_contribution_pm'])
	
			self.config['deal_transducers_to_receivers'](self.selector_emitters, self.transducers_pos_cuda, self.transducers_norm_cuda,
																						 self.receivers_pos, self.receivers_norm)
			
			if receivers_interacts:
				self.config['deal_rel_transducers_to_rel_receivers'](self.selector_emitters, self.rel_transducer_transducer['sm_DD_Green_function'], self.rel_receivers_receivers['sm_DD_Green_function'])

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.deal_transducers: {e}')

	def concatenate_arrays(self, old, new, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			
			temporal = np.empty([old.shape[0] + new.shape[0], new.shape[1]], dtype = self.new.dtype)

			self.config['concatenate_'+vartype+'_arrays'](old, new, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.concatenate_arrays: {e}')

	def concatenate_arrays_delete_rows(self, old, new, rows_to_remove, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			assert rows_to_remove.shape[0] >= 1, 'No rows specified.'
			
			#d_rows_to_remove = rows_to_remove
			
			temporal = np.empty([old.shape[0] + new.shape[0] - rows_to_remove.shape[0], new.shape[1]], dtype = self.new.dtype)

			self.config['concatenate_'+vartype+'_arrays_delete_rows'](old, new, rows_to_remove, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.concatenate_arrays_delete_rows: {e}')
	
	def concatenate_matrix(self, old_old, old_new, new_old, new_new, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			
			temporal = np.empty([old_old.shape[0] + new_new.shape[0] ,
								 old_old.shape[1] + new_new.shape[1]],
								dtype = self.old_old.dtype)

			self.config['concatenate_'+vartype+'_matrix'](old_old, old_new, new_old, new_new, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.concatenate_matrix: {e}')
		
	def concatenate_matrix_delete_rowsncols(self, old_old, old_new, new_old, new_new, delete_rowsncols, vartype):
		'''
		After this, stream.sychronize()
		'''
		try:
			
			assert vartype == self.VarType or vartype == self.OutVarType, f'Invalid variable type: {vartype}.'
			assert delete_rowsncols.shape[0] >= 1, 'No rows nor cols specified.'
			
			#d_rowsncols_to_remove = delete_rowsncols
			
			temporal = np.empty([old_old.shape[0] + new_new.shape[0] - delete_rowsncols.shape[0],
								 old_old.shape[1] + new_new.shape[1] - delete_rowsncols.shape[0]],
								dtype = self.old_old.dtype, stream = self.stream)

			self.config['concatenate_'+vartype+'_matrix_delete_rowsncols'](old_old, old_new, new_old, new_new, delete_rowsncols, temporal)
			
			return temporal

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.concatenate_matrix_delete_rowsncols: {e}')

	def update_mesh_positions (self, number_new_elements, delete_elements):
		try:
			assert delete_elements.shape[0] >= 1, 'No rows specified.'
			assert number_new_elements in range(1, self.mesh_pos['New'].shape[0]), 'Bad configuration of number of new elements.'
			
			if self.mesh_pos['Old'] is None:
				self.mesh_pos['Old'] = self.mesh_pos['New']
				self.mesh_norm['Old'] = self.mesh_norm['New']
				self.mesh_area_div_4pi['Old'] = self.mesh_area_div_4pi['New']
				self.mesh_pression['Old'] = self.mesh_pression['New']
			else:
				self.mesh_pos['Old'] = self.concatenate_arrays_delete_rows(self.mesh_pos['Old'], self.mesh_pos['New'], delete_elements, self.VarType)
				self.mesh_norm['Old'] = self.concatenate_arrays_delete_rows(self.mesh_norm['Old'], self.mesh_norm['New'], delete_elements, self.VarType)
				self.mesh_area_div_4pi['Old'] = self.concatenate_arrays_delete_rows(self.mesh_area_div_4pi['Old'], self.mesh_area_div_4pi['New'], delete_elements, self.VarType)
				self.mesh_pression['Old'] = self.concatenate_arrays_delete_rows(self.mesh_pression['Old'], self.mesh_pression['New'], delete_elements, self.OutVarType)

			self.mesh_pos['New'] = np.random.rand(number_new_elements, 3).astype(self.mesh_pos['Old'].dtype)
			self.mesh_norm['New'] = np.random.rand(number_new_elements, 3).astype(self.mesh_norm['Old'].dtype)
			self.mesh_area_div_4pi['New'] = np.random.rand(number_new_elements, 1).astype(self.mesh_area_div_4pi['Old'].dtype)
			self.mesh_pression['New'] = np.random.rand(number_new_elements, 1) + 1j*np.random.rand(number_new_elements, 1)
			
			self.update_mesh_relations(number_new_elements, delete_elements)
			
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.update_mesh_positions: {e}')
			
	def update_mesh_relations (self, number_new_elements, delete_elements):
		try:
			if self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'] is None:
				self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'] = self.mesh_pos['New']['New']
			else:
				self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'] = self.concatenate_arrays_delete_rows(
									self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'], self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'], 
									self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'], self.rel_mesh_mesh['sm_DD_Green_function']['New']['New'], 
									delete_elements, self.OutVarType)
			
			self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'] = np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'].shape[0], number_new_elements).astype(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'].dtype)
			self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'] = np.random.rand(number_new_elements, self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'].shape[1]).astype(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['Old'].dtype)
			
			#self.rel_mesh_mesh['sq_distance']['Old']['New'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[1]).astype(self.rel_mesh_mesh['sq_distance']['New']['New'].dtype), stream = self.stream)
			#self.rel_mesh_mesh['sq_distance']['New']['Old'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[1]).astype(self.rel_mesh_mesh['sq_distance']['New']['New'].dtype), stream = self.stream)
			
			#self.rel_mesh_mesh['cos_angle_sn']['Old']['New'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[1]).astype(self.rel_mesh_mesh['cos_angle_sn']['New']['New'].dtype), stream = self.stream)
			#self.rel_mesh_mesh['cos_angle_sn']['New']['Old'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[1]).astype(self.rel_mesh_mesh['cos_angle_sn']['New']['New'].dtype), stream = self.stream)
			
			#self.rel_mesh_mesh['complex_phase']['Old']['New'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['Old']['New'].shape[1]).astype(self.rel_mesh_mesh['complex_phase']['New']['New'].dtype), stream = self.stream)
			#self.rel_mesh_mesh['complex_phase']['New']['Old'] = cuda.to_device(np.random.rand(self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[0], self.rel_mesh_mesh['sm_DD_Green_function']['New']['Old'].shape[1]).astype(self.rel_mesh_mesh['complex_phase']['New']['New'].dtype), stream = self.stream)
			
			self.init_new_mesh_rel(number_new_elements)
			
											 
		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.update_mesh_relations: {e}')
			
	def manually_fill_mesh (self, mesh_pos, mesh_norm, mesh_area_div_4pi):
		
		try:
			
			assert ( mesh_pos.shape == self.mesh_pos['New'].shape and mesh_norm.shape == self.mesh_norm['New'].shape
					and mesh_area_div_4pi.shape == self.mesh_area_div_4pi['New'].shape ), 'Not valid '
			
			self.mesh_pos['New'] = mesh_pos
			self.mesh_norm['New'] = mesh_norm
			self.mesh_area_div_4pi['New'] = mesh_area_div_4pi
			

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.manually_fill_mesh: {e}')
			
	def erase_variable (*vars_to_erase):
		try:
			
			for var in vars_to_erase:
				var = None			

		except Exception as e:
			print(f'Error in utils.cpu.manager.manager_cpu.manually_fill_mesh: {e}')