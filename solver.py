import numpy as np
import mesh as Mesh
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
        self._coefMatrix = self._coefMatrixTemplate.duplicate()
        self._rhs = self._rhsTemplate.duplicate()
        return self._coefMatrix, self._rhs
    def allocate(self, Tinit):
        ncell = self._mesh.get_ncell()
        vec = PETSc.Vec().createSeq(ncell)
        self._rhsTemplate = vec.duplicate()
        self._xsol = vec.duplicate()
        self._coefMatrixTemplate = PETSc.Mat().createAIJ([ncell, ncell], nnz = 5)
        self._T = np.zeros((ncell))
        self._T[:] = Tinit
    def prepare_context(self): # prepare CG with ILU Precondition
        self._ksp.setType(PETSc.KSP.Type.CG)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        self._ksp.setPC(pc)
        #self.build_laplas_matrix(self._coefMatrixTemplate)
    def add_teporal_term(self, A, b, dt):
        for i in xrange(0, self._mesh.get_ncell()):
            vol = self._mesh.get_volumn(i)
            rcp  = self._mesh.get_material_at_index(i).rcp()
            temporal_term = vol *  rcp / dt
            A.setValue(i, i, temporal_term, addv = True) 
            b.setValues(i, temporal_term * self._T[i],addv = True)
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()
    def set_down_side(self, A, b, h, q, Tf):
        idxs = self._mesh.get_region('down') 
        ncell = self._mesh.get_ncell()
        area = [self._mesh.get_neighbour_area(idx)[5] for idx in idxs]
        for i in xrange():
            A.setValues( i, i, area[i] * h, addv = True)
        b.setValues(range(0, ncell), area * (h * Tf  - q ), addv = True)
        b.assemblyBegin()
        A.assemblyBegin()
        b.assemblyEnd()
        A.assemblyEnd()
    def set_heat_point(self, b, idx_array, heat_array):
        assert len(idx_array) == len(heat_array)
        b.setValues(idx_array, heat_array, addv = True)
    def set_upper_flux(self, b, idx_array, flux):
        areas = [self._mesh.get_neighbour_area(idx)[4] for idx in idx_array]
        flux = flux * areas
        b.setValues(idx_array, flux, addv = True)
    def build_laplas_matrix(self, A):
        for i in xrange(0, self._mesh.get_ncell()):
            nei = self._mesh.get_neighbour(i) 
            lens = self._mesh.get_neighbour_lenth(i)
            coef = self._mesh.get_neighbour_coef(i)
            area = self._mesh.get_neighbour_area(i)

            lens = filter(lambda (i, val): nei[i] is not None, enumerate(lens))
            area = filter(lambda (i, val): nei[i] is not None, enumerate(area))
            coef = filter(lambda (i, val): nei[i] is not None, enumerate(coef))
            nei = filter(lambda val: val is not None, nei)
            vals = coef * area / lens
            A.setValues([i], nei, vals) # off-diagnal
            center = 0. - sum(vals)
            A.setValue(i, i,  center)#diagnal
        A.assemblyBegin()
        A.assemblyEnd()
    def set_mask(self, melted_mask): #set on the template
        A = self._coefMatrixTemplate
        b = self._rhsTemplate
        self._melted_set = self._melted_set | melted_mask #merge
        for idx in melted_mask:
            self._T[idx] = -1.0
            nei = filter(lambda v : v is not None,  self._mesh.get_neighbour(idx) )
            A.setValues([idx], nei, 0.0, addv = False)
            A.setValues(nei, [idx], 0.0, addv = False)
        b.setValues(list(melted_mask), 0.0, addv = False)
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()
    def get_mask(self):
        return self._melted_set
    def update_mask(self):
        ret = set()
        for i, temp in self._T:
            if i in self._melted_set:
                continue
            melt_point = self._mesh.get_material_at_index(i).melt_point
            if melt_point < temp:
                ret.add(i)
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
           # print 'iteration converged in %d step' % ksp.getIterationNumber()
        self._T[:] = self._xsol.getArray()
        return self._T

