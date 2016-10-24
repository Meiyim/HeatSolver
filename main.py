import numpy as np
import sys
import math
from petsc4py import PETSc
import petsc4py
petsc4py.init(sys.argv)

import mesh as Mesh
import solver as Solver
import constant as Const
import utility as uti

def summary(now, T):
    print '[time %f] max T %e min %e ave %e' % (T.max(), T.min(), T.mean())

   
if __name__ == '__main__':
    mat = Mesh.Material(Const.dict['global_lamda'], 
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

    solver = PetscSolver(mesh)
    solver.prepare_context()
    solver.allocate(Const.dict['T_init'])

    assembly_id = {}
    for iass, (x, 7) in Const.assembly_pos:
        idx = mesh.get_index_at_poristion((x, y, Const.dict['board_thick']))
        assembly_id[iass] = idx
    assembly_id=sorted(assembly_id, key = lambda v:v[0])
    assembly_id=map(lambda (iass,idx) : idx, assembly_id)

    down_id = mesh.get_region('down')

#geometry related
    down_area = math.pi * Const.dict['board_radious'] ** 2
    down_board_length = 2 * math.pi * Const.dict['board_radious']
    corelation_length = down_area / down_board_length

    time_step = 0
    pool_volumn = 0
    A, b = solver.get_template()
    solver.build_laplas_matrix(A)
    solver.add_teporal_term(A, b, 1.0)
    for t, drop_list in uti.parse_input_file('input.dat'):
        pool_volumn += uti.calc_drop_volumn(drop_list)
        if len(melted_set) != 0:
            solver.set_mask(melted_set)
        T = solver.get_T()
        #down_side
        T_down_mean = np.array( map(lambda idx: T[idx], down_id) ).mean()
        Gr = uti.calSteamGr(T_steam - T_down_mean)
        h = uti.calcSteamHeatTransferRate(Gr, Const.dict['Pr_steam'], corelation_length)
        A_, b_ = solver.duplicate_template()
        solver.set_down_side(A_, b_, 0.0, h * (T_steam - T_down_mean), T_steam)
        #up_side
        melted_set = solver.update_mask()
        melted_volumn = uti.calc_melted_volumn(melted_set, mesh)
        if pool_volumn < melted_volumn:    
            drop_heat = uti.calc_drop_heat(drop_list)
            solver.set_heat_point(assembly_id, drop_heat)
        else: #pool cover the bottom
            upper_surface_idx = mesh.get_upper_surface()    #TODO to be impement
            flux = uti.calc_total_up_heat(drop_list)        #TODO to be impement
            solver.set_upper_flux(upper_surface_idx, flux)  
        # other boundary goes here
        T = solver.solve(1.e-6, 100)
        summary(t, T)
        print 'melted volumn %e, pool height %e' % (melted_volumn % )
        print 'solve done for timestep %d' % time_step
        time_step += 1

