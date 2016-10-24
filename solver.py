import numpy
import sys
import mesh
import numpy as np
from petsc4py import PETSc

class Solver(objet):
    def __init__(self, mesh):
        assert isinstance(mesh, Mesh)
        self.__mesh = mesh
        self.__rhs = None
        self.__xsol = None
        self.__coefMatrix = None
        self.__T = None
    def get_T(self):
        return self.__T
    def get_mesh(self):
        return self.__mesh
    def allocate(self):
        pass
    def prepare_context(self):
        pass

class PetscSolver(Solver):
    def __init__(self, mesh):
        super(PetscSolver, self).__init__(mesh)
        self.__ksp = None
        self.__coefMatrixTemplate = None
        self.__rhsTemplate = None
        self.__melted_set = set()
    def get_template(self):
        return self.__coefMatrixTemplate, self.__rhsTemplate
    def duplicate_template(self):
        self.__coefMatrix = self.__coefMatrixTemplate.duplicate()
        self.__rhs = self.__rhsTemplate.duplicate()
        return self.__coefMatrix, self.__rhs
    def allocate(self, Tinit):
        ncell = self.__mesh.get_ncell()
        vec = PETSc.Vec().createSeq(ncell)
        self.__rhsTemplate = vec.duplicate()
        self.__xsol = vec.duplicate()
        self.__coefMatrixTemplate = PETSc.Mat().createAIJ([ncell, ncell], nnz = 5)
        self.__T = np.zeros((ncell))
        self.__T[:] = Tinit
    def prepare_context(self): # prepare CG with ILU Precondition
        self.__ksp.setType(PETSc.KSP.Type.CG)
        pc = self.__ksp.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        self.__ksp.setPC(pc)
        #self.build_laplas_matrix(self.__coefMatrixTemplate)
    def add_teporal_term(self, A, b, dt):
        for i in xrange(0, self.__mesh.get_ncell()):
            vol = self.__mesh.get_volumn(i)
            rcp  = self.__mesh.get_material_at_index(i).rcp()
            temporal_term = vol *  rcp / dt
            A.setValue(i, i, temporal_term, addv = True) 
            b.setValues(i, temporal_term * self.__T[i],addv = True)
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()
    def set_down_side(A, b, h, q, Tf):
        idx = self.__mesh.get_region('down') 
        ncell = self.__mesh.get_ncell()
        area = map(lambda v: self.__mesh.get_neighbour_area(v)[5], idx)
        dT = self.__T - Tf
        for i in xrange()
            A.setValues( i, i, area[i] * h, addv = True)
        b.setValues(range(0, ncell), area * (h * Tf  - q ), addv = True)
        b.assemblyBegin()
        A.assemblyBegin()
        b.assemblyEnd()
        A.assemblyEnd()
    def set_heat_point(idx_array, heat_array):
        assert len(idx_array) == len(heat_array)
        b.setValues(idx_array, heat_array, addv = True)
    def set_upper_flux(idx_array, flux):
        areas = map(lambda idx: self.__mesh.get_neighbour_area(idx)[4], idx_array)
        flux = flux * areas
        b.setValues(idx_array, flux, addv = True)
    def build_laplas_matrix(self, A):
        for i in xrange(0, self.__mesh.get_ncell()):
            nei = self.__mesh.get_neighbour(i) 
            lens = self.__mesh.get_neighbour_lenth(i)
            coef = self.__mesh.get_neighbour_coef(i)
            area = self.__mesh.get_neighbour_area(i)

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
        A = self.__coefMatrixTemplate
        b = self.__rhsTemplate
        self.__melted_set = self.__melted_set | melted_mask #merge
        for idx in melted_idx:
            self.__T[idx] = -1.0
            nei = filter(lambda v : v is not None,  self.__mesh.get_neighbour(idx) )
            A.setValues([idx], nei, 0.0, addv = False)
            A.setValues(nei, [idx], 0.0, addv = False)
        b.setValues(list(melted_mask), 0.0, addv = False)
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()
    def update_mask(self):
        ret = set()
        for i, t in self.__T:
            if i in self.__melted_set:
                continue
            melt_point = self.__mesh.get_material_at_index(i).melt_point
            if melt_point < T:
                ret.add(i)
        return ret
    def solve(rtol, max_iter):
        self.__coefMatrix.assemblyBegin()
        self.__rhs.assemblyBegin()
        self.__coefMatrix.assemblyEnd()
        self.__rhs.assemblyEnd()
        self.__xsol.setValues(range(0, self.__mesh.get_ncell()), self.__T)
        self.__ksp.setInitialGuessNonzero(True)
        self.__ksp.setOperators(self.__coefMatrix)
        self.__ksp.setTolerances(rtol=rtol, max_it=max_iter)
        self.__ksp.solve(self.__rhs, self.__xsol)
        if self.__ksp.getConvergedReason() < 0:
            raise ValueError, 'iteration not converged'
        #else:
           # print 'iteration converged in %d step' % ksp.getIterationNumber()
        self.__T[:] = self.__xsol.getArray()
        return self.__T




