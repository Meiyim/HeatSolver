import numpy as np
import sys
import math
import petsc4py
petsc4py.init(sys.argv)

import mesh as Mesh
import solver as Solver
import constant as Const
import utility as uti

def summary(now, T):
    print '[time %f] max T %e min %e ave %e' % (now, T.max(), T.min(), T.mean())

   
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
    for iass, x, y in Const.assembly_pos:
        idx = mesh.get_index_at_poristion((x, y, Const.dict['board_thick']))
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
    time_step = 0
    pool_volumn = 0
    melted_set_sum = set()
    melted_set = set()
    print 'start solving'
    #
    A, b = solver.get_template()
    solver.build_laplas_matrix(A)
    solver.add_teporal_term(A, b, 1.0)
    for (t, drop_list),(now_water, bottom_t, now_power_distribution) in zip(uti.parse_input_file('input.dat'), uti.core_status_generator(0.0, 1.0)):
        pool_volumn += uti.calc_drop_volumn(drop_list)
        if len(melted_set) != 0:
            solver.set_mask(melted_set)
        T = solver.get_T()
        #down_side
        T_down_mean = np.array([ T[idx] for idx in  down_id]).mean()
        h = uti.calc_hcoef(T_down_mean, corelation_length, T_steam)
        A_, b_ = solver.duplicate_template()
        solver.set_down_side(A_, b_, 0.0, h * (T_steam - T_down_mean), T_steam)
        #up_side
        melted_set = solver.update_mask()
        melted_set_sum = melted_set_sum | melted_set
        upper_surface_idx = mesh.get_upper_surface(melted_set_sum)    
        melted_volumn = uti.calc_melted_volumn(melted_set_sum, mesh)
        T_up_mean = np.array([ T[idx] for idx in upper_surface_idx] ).mean()
        if pool_volumn < melted_volumn:    
            #core
            flux_from_core = uti.calc_core_flux(bottom_t, T_up_mean)
            solver.set_upper_flux(b_, upper_surface_idx, flux_from_core)
            # drop 
            rod_idx, drop_heat_for_rod = uti.calc_drop_heat(drop_list, assembly_id)
            solver.set_heat_point(b_, rod_idx, drop_heat_for_rod)
        else: #pool cover the bottom
            flux_from_pool = uti.calc_pool_heat(drop_list)        
            solver.set_upper_flux(b_, upper_surface_idx, flux_from_pool)  
        # other boundary goes here
        T = solver.solve(1.e-6, 100)
        summary(t, T)
        print 'melted volumn %e, pool volumn %e' % (melted_volumn, pool_volumn)
        print 'solve done for timestep %d' % time_step
        time_step += 1

if __name__ == '__main__':
    main()
