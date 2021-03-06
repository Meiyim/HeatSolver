import numpy as np
import mesh as Mesh
import utility as uti
import constant as Const
import itertools as iter
from numba import jit
from petsc4py import PETSc

class Solver(object):
    def __init__(self, mesh):
        assert isinstance(mesh, Mesh.Mesh)
        self._mesh = mesh
        self._rhs = None
        self._xsol = None
        self._coefMatrix = None
        self._T = None
    def get_T(self):
        return self._T
    def get_mesh(self):
        return self._mesh
    def allocate(self, Tinit):
        pass
    def prepare_context(self):
        pass

class PetscSolver(Solver):
    def __init__(self, mesh):
        super(PetscSolver, self).__init__(mesh)
        self._ksp = None
        self._coefMatrixTemplate = None
        self._rhsTemplate = None
        self._melted_set = set()
    def get_template(self):
        return self._coefMatrixTemplate, self._rhsTemplate
    def duplicate_template(self):
        self._coefMatrix = self._coefMatrixTemplate.duplicate(copy=True)
        self._rhs = self._rhsTemplate.duplicate()
        return self._coefMatrix, self._rhs
    def allocate(self, Tinitmin, Tinitmax):
        ncell = self._mesh.get_ncell()
        vec = PETSc.Vec().createSeq(ncell)
        self._rhsTemplate = vec.duplicate()
        self._xsol = vec.duplicate()
        self._coefMatrixTemplate = PETSc.Mat().createAIJ([ncell, ncell], nnz = 7)
        self._T = np.zeros((ncell))
        nz = Const.dict['nz']
        ny = Const.dict['nt']
        nx = Const.dict['nr']
        arr_temp = np.linspace(Tinitmin, Tinitmax, nz)
        for i in xrange(0, nx):
            for j in xrange(0, ny):
                for k in xrange(0, nz):
                    self._T[self._mesh.get_index((i, j, k))] = arr_temp[k]

    def prepare_context(self): # prepare CG with ILU Precondition
        self._ksp = PETSc.KSP().create()
        self._ksp.setType(PETSc.KSP.Type.CG)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        self._ksp.setPC(pc)
        #self.build_laplas_matrix(self._coefMatrixTemplate)

    def add_teporal_term(self, A, b, melted_set, dt):
        for i in xrange(0, self._mesh.get_ncell()):
            if i in melted_set: continue
            vol = self._mesh.get_volumn(i)
            rcp  = self._mesh.get_material_at_index(i).rcp()
            temporal_term = vol *  rcp / dt
            A.setValue(i, i, temporal_term, addv = True) 
            b.setValue(i, temporal_term * self._T[i],addv = True)
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()
    def set_down_side(self, A, b, h, q, Tf):
        idxs = self._mesh.get_region('down') 
        ncell = self._mesh.get_ncell()
        area_array = np.array([self._mesh.get_neighbour_area(idx)[5] for idx in idxs])
        for area, idx in zip(area_array, idxs):
            A.setValues( idx, idx, area * h, addv = True)
        b.setValues(idxs, area_array * (h * Tf  - q ), addv = True)
        uti.log('down_sum_heat %e' % sum(area_array * (h * Tf - q)))
        b.assemblyBegin()
        A.assemblyBegin()
        b.assemblyEnd()
        A.assemblyEnd()
    def set_heat_point(self, b, idx_array, heat_array):
        assert len(idx_array) == len(heat_array)
        b.setValues(idx_array, 0. - np.array(heat_array), addv = True)
        b.assemblyBegin()
        b.assemblyEnd()
    def set_upper_flux(self, b, idx_array, flux):
        if len(idx_array) == 0: return
        areas = [self._mesh.get_neighbour_area(idx)[4] for idx in idx_array]
        flux = flux * np.array(areas)
        b.setValues(list(idx_array), 0. - flux, addv = True)
        b.assemblyBegin()
        b.assemblyEnd()
    def update_laspack_matrix(self, melted_set, melted_set_sum):
        A = self._coefMatrixTemplate
        nei_set = set()
        for idx in melted_set:
            self._T[idx] = 1.0
            nei = filter(lambda v : v is not None,  self._mesh.get_neighbour(idx) )
            for nidx in nei:
                nei_set.add(nidx)
            A.setValues([idx], nei, [0.0] * len(nei), addv = False)
            A.setValues(nei, [idx], [0.0] * len(nei), addv = False)
            A.setValue(idx, idx, 100.0, addv = False)
        nei_set = nei_set - melted_set_sum
        for row in nei_set: #recalculate
            assert row not in melted_set
            nei = self._mesh.get_neighbour(row) 
            lens = self._mesh.get_neighbour_lenth(row)
            coef = self._mesh.get_neighbour_coef(row)
            area = self._mesh.get_neighbour_area(row)
    
            #print 'cord', self._mesh.get_3d_index(row)
            #print 'coef', coef
            #print 'area', area
            #print 'lens', lens
            #print 'nei', nei
            lens = np.array([ l for i,l in enumerate(lens) if nei[i] is not None]) #and nei[i] not in melted_set_sum])
            area = np.array([ a for i,a in enumerate(area) if nei[i] is not None]) #and nei[i] not in melted_set_sum])
            nei = filter(lambda val: val is not None, nei)

            area = np.array([ 0. if nei[i] in melted_set_sum else a  for i,a in enumerate(area)]) #and nei[i] not in melted_set_sum])

            vals = -1.0 * coef * area / lens
            #print 'vals',  vals
            A.setValues([row], nei, vals, addv = False) # off-diagnal
            center = 0. - sum(vals)
            A.setValue(row, row,  center, addv = False)#diagnal
        A.assemblyBegin()
        A.assemblyEnd()



    def build_laplas_matrix(self):
        A = self._coefMatrixTemplate
        uti.log('building laplas template...')
        for row in xrange(0, self._mesh.get_ncell()):
            nei = self._mesh.get_neighbour(row) 
            lens = self._mesh.get_neighbour_lenth(row)
            coef = self._mesh.get_neighbour_coef(row)
            area = self._mesh.get_neighbour_area(row)
    
            #print 'cord', self._mesh.get_3d_index(row)
            #print 'coef', coef
            #print 'area', area
            #print 'lens', lens
            #print 'nei', nei
            lens = np.array([ l for i,l in enumerate(lens) if nei[i] is not None])
            area = np.array([ a for i,a in enumerate(area) if nei[i] is not None])
            nei = filter(lambda val: val is not None, nei)
            vals = -1.0 * coef * area / lens
            #print 'vals',  vals
            A.setValues([row], nei, vals) # off-diagnal
            center = 0. - sum(vals)
            A.setValue(row, row,  center)#diagnal
        A.assemblyBegin()
        A.assemblyEnd()
    def set_mask(self, melted_mask): #set on the template
        A = self._coefMatrixTemplate
        b = self._rhsTemplate
        self._melted_set = self._melted_set | melted_mask #merge
        minus_dict = {}
        for idx in melted_mask:
            self._T[idx] = -1.0
            nei = filter(lambda v : v is not None and v not in melted_mask,  self._mesh.get_neighbour(idx) )
            for nid, val in zip( nei, A.getValues(idx, nei) ):
                if minus_dict.get(nid) is None:
                    minus_dict[nid] = val
                else:
                    minus_dict[nid] += val
            A.setValues([idx], nei, [0.0] * len(nei), addv = False)
            A.setValues(nei, [idx], [0.0] * len(nei), addv = False)
            A.setValue(idx, idx, 100.0, addv = False)
        A.assemblyBegin()
        A.assemblyEnd()
        for idx, val in minus_dict.items():
            A.setValue(idx, idx, 0. - val, addv = True)
        A.assemblyBegin()
        A.assemblyEnd()
    def get_mask(self):
        return self._melted_set
    def update_mask(self):
        ret = set()
        for i, temp in enumerate(list(self._T)):
            if i in self._melted_set:
                continue
            melt_point = self._mesh.get_material_at_index(i).melt_point
            if melt_point < temp:
                ret.add(i)
        return ret
    def get_drop_point_temp(self, assembly_id):
        ret = {}
        for iass, idxs in assembly_id.items():
            if len(idxs) == 0: continue # the assembly is penetrated
            ret[iass] = sum([self._T[idx] for idx in idxs ]) / len(idxs)
        return ret
    def solve(self, rtol, max_iter):
        self._coefMatrix.assemblyBegin()
        self._rhs.assemblyBegin()
        self._coefMatrix.assemblyEnd()
        self._rhs.assemblyEnd()
        self._xsol.setValues(range(0, self._mesh.get_ncell()), self._T)
        self._ksp.setInitialGuessNonzero(True)
        self._ksp.setOperators(self._coefMatrix)
        self._ksp.setTolerances(rtol=rtol, max_it=max_iter)
        self._ksp.solve(self._rhs, self._xsol)
        if self._ksp.getConvergedReason() < 0:
            raise ValueError('iteration not converged')
        #else:
           # uti.log 'iteration converged in %d step' % ksp.getIterationNumber()
        self._T[:] = self._xsol.getArray()
        return self._T

