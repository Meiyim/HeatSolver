import numpy as np
import sys
import math
import time
import itertools as iter
import petsc4py
petsc4py.init(sys.argv)

import mesh as Mesh
import solver as Solver
import constant as Const
import utility as uti
from bintrees import RBTree
from numba import jit
import crash_on_ipy

last_mv = 0
last_pv = 0
def summary(now, T, status):
    global last_mv, last_pv
    uti.print_with_style('[time %8f] ' % now ,mode = 'bold', fore = 'red' )
    uti.print_with_style('max T %10e min %10e ave %10e ' % (T.max(), T.min(), T.mean()) )
    uti.print_with_style('[melt]',  mode = 'bold', fore = 'red' )
    uti.print_with_style('melt_v %10e(+%10e) pool_v %10e(+%10e)\n' % (status['melted_volumn'], status['melted_volumn'] - last_mv, status['pool_volumn'], status['pool_volumn']-last_pv))
    last_mv = status['melted_volumn']
    last_pv = status['pool_volumn']
    if status['board'] == 0:
        uti.print_with_style('[filling hole]\n', mode = 'bold', fore = 'green' )
    elif status['board'] == 1:
        uti.print_with_style('[jet impinging...]\n', mode = 'bold', fore = 'green' )
    elif status['board'] == 2:
        uti.print_with_style('[pooling]\n', mode = 'bold', fore = 'green' )
    else:
        pass
        
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
    status = {}
    status['board'] = 0 # filling hole,  jet, pool
    time_check = time.time()
    status['time_step'] = 0
    status['pool_volumn'] = 0
    status['melted_set'] = set()
    status['melted_set_sum'] = set()
    hole_volumn = uti.calc_hole_volumn()
    status['melted_volumn'] =  hole_volumn
    status['melted_set_tree'] = RBTree()


    uti.log('prepare solving')

    solver = Solver.PetscSolver(mesh)
    solver.allocate(Const.dict['T_init'])
    solver.prepare_context()

    A, b = solver.get_template()
    solver.build_laplas_matrix(A)
    uti.log('start solving')
    #restart
    rst_stat =  uti.load(solver.get_T())
    core_status = uti.core_status_generator(0.0, 1.0, status['time_step'])
    if rst_stat is not None:
        status = rst_stat
    for t, drop_list in uti.parse_input_file('melt_history_3.short', status['time_step']):
        now_water, bottom_t, now_power_distribution = core_status.next()
        status['time_step']  = t
        now = time.time()
        uti.log('[%d] solving... time consumed %e' % (t, now - time_check))
        time_check = now
        status['pool_volumn'] += uti.calc_drop_volumn(drop_list)
        if len(status['melted_set']) != 0:
            solver.set_mask(status['melted_set'])
        T = solver.get_T()
        #temporal_term
        A_, b_ = solver.duplicate_template()
        solver.add_teporal_term(A_, b_, 1.0 if t > 560 else 999.0)
        #down_side
        T_down_mean = np.array([ T[idx] for idx in  down_id]).mean()
        h_steam = uti.calc_hcoef(T_down_mean, corelation_length, T_steam)
        flux_down = h_steam * (T_down_mean - T_steam)
        uti.log ('heat coef steam %10e flux' % h_steam)
        solver.set_down_side(A_, b_, 0.0, flux_down, T_steam)
        #up_side
        status['melted_set'] = solver.update_mask()
        status['melted_set_sum'] = status['melted_set_sum'] | status['melted_set']
        upper_surface_idx = mesh.get_upper_surface(status['melted_set_sum'])    
        status['melted_volumn'] += mesh.calc_melted_volumn(status['melted_set'])
        T_up_mean = np.array([ T[idx] for idx in upper_surface_idx] ).mean()
        uti.log('down temp %e up temp %e' % (T_down_mean, T_up_mean))

        #from impingment
        drop_point_idx = mesh.get_drop_point_idx(assembly_id)
        drop_point_temp = solver.get_drop_point_temp(drop_point_idx)
        rod_idx, drop_heat_for_rod = uti.calc_drop_heat(drop_list, drop_point_idx, drop_point_temp)
        if drop_heat_for_rod is not None:
            uti.log('impinging')
            #print rod_idx
            assert len(set(rod_idx) & status['melted_set_sum']) == 0
            solver.set_heat_point(b_, rod_idx, drop_heat_for_rod)

        status['board'] = 1 if status['pool_volumn'] < status['melted_volumn'] else 2

        pool_bottom_surface_idx, pool_area = mesh.get_pool_bottom(status)
        #from core
        flux_from_core = uti.calc_core_flux(bottom_t, T_up_mean, h_steam)
        uti.log('core flux: %10e' % flux_from_core)
        solver.set_upper_flux(b_, list(upper_surface_idx - pool_bottom_surface_idx), flux_from_core)
        #from pool
        if len(pool_bottom_surface_idx) != 0:
            uti.log('pooling')
            flux_from_pool = uti.calc_pool_heat(drop_list, T_up_mean, pool_bottom_surface_idx, status['pool_volumn'], pool_area, now_power_distribution)
            uti.log('pool flux %10e' % flux_from_pool )
            solver.set_upper_flux(b_, pool_bottom_surface_idx, flux_from_pool)  
            # other boundary goes here
            # solve
        #debug
        '''
        for idx in upper_surface_idx:
            assert mesh.get_3d_index(idx)[2] == Const.dict['nz'] - 1
        barr = b_.getArray()
        for idx in upper_surface_idx:
            print barr[idx]
            if idx in rod_idx:
                print '************'
                raw_input()
        raw_input()
        '''
        T = solver.solve(1.e-6, 100)
        summary(t, T, status)
        #post
        if t  % Const.dict['output_step'] == 0:
            uti.print_with_style('[tecploting...]', mode = 'bold', fore = 'blue')
            str_buffer = mesh.tecplot_str(T, status, pool_bottom_surface_idx)
            open('tec/tec_%d.dat' % t, 'w').write(str_buffer)
        if t  % Const.dict['restart_step'] == 0:
            uti.save(status, T)

if __name__ == '__main__':
    main()
