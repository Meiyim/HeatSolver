import numpy as np
import sys
import math
import time
import petsc4py
petsc4py.init(sys.argv)

import mesh as Mesh
import solver as Solver
import constant as Const
import utility as uti
from numba import jit
import crash_on_ipy

def summary(now, T, mv, pv):
    uti.print_with_style('[time %f] ' % now ,mode = 'bold', fore = 'red' )
    uti.print_with_style('max T %e min %e ave %e ' % (T.max(), T.min(), T.mean()) )
    uti.print_with_style('[melt]',  mode = 'bold', fore = 'red' )
    uti.print_with_style('melt_v %e pool_v %e\n' % (mv, pv))

def main():
    mat = Mesh.Material(
        Const.dict['global_lamda'], 
        Const.dict['global_rou'],
        Const.dict['global_cp'],
        Const.dict['global_melt_point'],
        Const.dict['global_epsilong'])

    mesh = Mesh.CylinderlMesh(
        Const.dict['nr'],
        Const.dict['nt'],
        Const.dict['nz'],
        Const.dict['board_radious'],
        Const.dict['radious_increase_ratio'],
        Const.dict['board_thick'] )

    T_steam = Const.dict['T_steam']

    mesh.set_basic_materal(mat)
    solver = Solver.PetscSolver(mesh)
    solver.allocate(Const.dict['T_init'])
    solver.prepare_context()

    assembly_id = {}
    for x, y, iass in Const.assembly_pos:
        idx = mesh.get_bottom_index_at_position((x, y))
        assert idx is not None
        if assembly_id.get(iass) is None:
            assembly_id[iass] = [idx]
        else:
            assembly_id[iass].append(idx)
    down_id = mesh.get_region('down')
    #geometry related
    down_area = math.pi * Const.dict['board_radious'] ** 2
    down_board_length = 2 * math.pi * Const.dict['board_radious']
    corelation_length = down_area / down_board_length

    #record object
    time_check = time.time()
    time_step = 0
    pool_volumn = 0
    melted_set_sum = set()
    melted_set = set()
    print('prepare solving')
    A, b = solver.get_template()
    solver.build_laplas_matrix(A)
    print('start solving')
    for (t, drop_list),(now_water, bottom_t, now_power_distribution) in zip(uti.parse_input_file('melt_mass.dat'), uti.core_status_generator(0.0, 1.0)):
        now = time.time()
        print('[%d] solving... time consumed%e' % (time_step, now - time_check))
        time_check = now
        pool_volumn += uti.calc_drop_volumn(drop_list)
        if len(melted_set) != 0:
            solver.set_mask(melted_set)
        T = solver.get_T()
        #temporal_term
        A_, b_ = solver.duplicate_template()
        solver.add_teporal_term(A_, b_, 1.0)
        #down_side
        T_down_mean = np.array([ T[idx] for idx in  down_id]).mean()
        h = uti.calc_hcoef(T_down_mean, corelation_length, T_steam)
        solver.set_down_side(A_, b_, 0.0, h * (T_down_mean - T_steam ), T_steam)
        #up_side
        melted_set = solver.update_mask()
        melted_set_sum = melted_set_sum | melted_set
        upper_surface_idx = mesh.get_upper_surface(melted_set_sum)    
        melted_volumn = uti.calc_melted_volumn(melted_set_sum, mesh)
        T_up_mean = np.array([ T[idx] for idx in upper_surface_idx] ).mean()
        if pool_volumn < melted_volumn + 1.e-5:    
            print('pool did not form')
            #core
            flux_from_core = uti.calc_core_flux(bottom_t, T_up_mean)
            print('core flux: %e' % flux_from_core)
            solver.set_upper_flux(b_, upper_surface_idx, flux_from_core)
            # drop 
            rod_idx, drop_heat_for_rod = uti.calc_drop_heat(drop_list, assembly_id)
            solver.set_heat_point(b_, rod_idx, drop_heat_for_rod)
        else: #pool cover the bottom
            print('pool formed')
            flux_from_pool = uti.calc_pool_heat(drop_list)        
            print('pool flux %e' % flux_from_pool )
            solver.set_upper_flux(b_, upper_surface_idx, flux_from_pool)  
        # other boundary goes here
        T = solver.solve(1.e-6, 100)
        summary(t, T, melted_volumn, pool_volumn)
        #post
        if time_step % 10 == 0:
            str_buffer = mesh.tecplot_str(T)
            open('tec/tec_%d.dat' % time_step, 'w').write(str_buffer)
        time_step += 1
        #profile
        if time_step % 10 == 0:
            break

if __name__ == '__main__':
    main()
