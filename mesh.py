import math
import numpy as np
import itertools
import utility as uti

class Material(object):
    def __init__(self, lamda, rou, cp, mp, sigma = 0.0):
        self.lamda = lamda
        self.rou = rou
        self.cp = cp
        self.melt_point = mp
        self.sigma = sigma
    def rcp(self):
        return self.rou * self.cp

class Mesh(object):
    def __init__(self, ncell): 
        self._ncell = ncell
        self._region = {}
        self._basic_material = None
    def get_ncell(self):
        return self._ncell
    def get_index_at_poristion(self, pos):
        pass
    def get_region(self, name):
        return list(self._region.get(name))
    def get_volumn(self, i):
        pass
    def get_neighbour(self, i):
        pass
    def get_neighbour_lenth(self, i):
        pass
    def get_neighbour_coef(self, i):
        pass
    def get_neighbour_area(self, i):
        pass

class StructuredMesh3D(Mesh):
    def __init__(self, x, y, z, x1, x2, y1, y2, z1, z2):
        super(StructuredMesh3D, self).__init__(x * y * z)
        self._nx = x
        self._cordinatex = np.linspace(x1, x2, x)
        self._ny = y
        self._cordinatey = np.linspace(y1, y2, y)
        self._nz = z
        self._cordinatez = np.linspace(z1, z2, z)
        self._basic_material = None
    def set_basic_materal(self, mat):
        assert isinstance(mat, Material)
        self._basic_material = mat
    def get_material_at_index(self, i): #temperally uniform material
        return self._basic_material
    def set_region_3d(self, name, region):
        nodes_index = [self.get_index(idx) for idx in region]
        assert None not in nodes_index
        self._region[name] = set(nodes_index)
    def get_3d_index(self, i):
        if i >= self._ncell or i < 0 :
            return None
        else:
            return  (i / (self._ny * self._nz), \
                     i / (self._nz), \
                     i )
    def get_index(self, cord): #None able
        x, y, z = cord
        if 0 <= x < self._nx or 0<= y < self._ny or 0 < z <= self._nz :
            return x * self._ny * self._nz + \
                   y * self._nz + \
                   z
        else:
            return None
    def get_neighbour3d(self, cord):
        i, j, k = cord
        return (   self.get_index((i+1, j, k)),
                self.get_index((i-1, j, k)),
                self.get_index((i, j+1, k)),
                self.get_index((i, j-1, k)),
                self.get_index((i, j, k+1)),
                self.get_index((i, j, k-1)),
              )
    def get_neighbour(self, idx):
        cord = self.get_3d_index(idx)
        return self.get_neighbour3d(cord)

    def get_neighbour_length3d(self, cord):
        i, j, k = cord
        ret =  ( self._cordinatex[(i + 1) % self._nx] - self._cordinatex[i],
                 self._cordinatex[(i - 1) % self._nx] - self._cordinatex[i],
                 self._cordinatey[(j + 1) % self._ny] - self._cordinatey[j],
                 self._cordinatey[(j - 1) % self._ny] - self._cordinatey[j],
                 self._cordinatez[(k + 1) % self._nz] - self._cordinatez[k],
                 self._cordinatez[(k - 1) % self._nz] - self._cordinatez[k],
               )
        return ret
    def get_neighbour_lenth(self, idx):
        cord = self.get_3d_index(idx)
        return self.get_neighbour_length3d(cord)
    def d_cordinate_center(self, cord):
        dx1, dx2, dy1, dy2, dz1, dz2 = self.get_neighbour_length3d(cord)
        return (dx1+dx2)/2, (dy1+dy2)/2, (dz1+dz2)/2
    def get_volumn3d(self, cord):
        dx, dy, dz = self.d_cordinate_center(cord)
        return dx * dy * dz
    def get_volumn(self, idx):
        cord = self.get_3d_index(idx)
        return self.get_volumn3d(cord)
    def get_neighbour_area3d(self, cord):
        dx, dy, dz = self.d_cordinate_center(cord)
        return dy*dz, dy*dz, dx*dz, dx*dz, dy*dz, dy*dz
    def get_neighbour_area(self, i):
        cord = self.get_3d_index(i)
        return self.get_neighbour_area3d(cord)
    def get_neighbour_coef(self, i):
        return self._basic_material.lamda

class CylinderlMesh(StructuredMesh3D):
    def __init__(self, ir, it, iz, r, qr, z):
        dr = r / float(ir)
        dr /= 2
        dz = z / float(iz)
        print 'begin %f end %f ratio %e' % (dr, r-dr, qr)
        super(CylinderlMesh, self).__init__(ir, it, iz, dr, r-dr, 0., math.pi, dz, z - dz)
        self._cordinatex = uti.stupid_method(dr, r-dr, qr, ir)
        print 'CHECK CORDINATE R: %s' % str(self._cordinatex)
        upper_bound = set(
            [self.get_index(cord)  for cord in itertools.product(xrange(0, ir), xrange(0, it), xrange(iz - 1, iz)) ]
        )
        self._upper_boundary = upper_bound 
        self.set_region_3d('up', itertools.product(xrange(0, self._nx), xrange(0, self._ny), xrange(iz - 1, self._nz)))
        self.set_region_3d('down', itertools.product(xrange(0, self._nx), xrange(0, self._ny), xrange(0, 1)))
        self.set_region_3d('side', itertools.product(xrange(ir-1, self._nx), xrange(0, self._ny), xrange(0, self._nz)))

    def get_neighbour3d(self, cord):
        i, j, k = cord
        return filter(lambda v : v is not None,
            (
                self.get_index((i+1, j, k)),
                self.get_index((i-1, j, k)),
                self.get_index((i, (j+1)%self._ny, k)),
                self.get_index((i, (j-1)%self._ny, k)),
                self.get_index((i, j, k+1)),
                self.get_index((i, j, k-1)),
            ))
    def get_index_at_poristion(self, pos):
        x, y, z = pos
        dz = int((z - self._cordinatez[0]) / self._cordinatez.shape[0])
        r = math.sqrt(x**2 + y**2)
        theta = math.acos( x/r )
        dt = int((theta - self._cordinatey[0]) / self._cordinatey.shape[0])
        dr = 0
        for the_r in self._cordinatex:
            dr += 1
            if the_r > r:
                break
        return self.get_index((dz, dt, dr))
 
    def d_cordinate(self, cord):
        i, j, k = cord
        dr1 =       self._cordinatex[(i + 1) % self._nx] - self._cordinatex[i],
        dr2 =       self._cordinatex[(i - 1) % self._nx] - self._cordinatex[i],
        dtheta1 =   self._cordinatey[(j + 1) % self._ny] - self._cordinatey[j],
        dtheta2 =   self._cordinatey[(j - 1) % self._ny] - self._cordinatey[j],
        dz1 =       self._cordinatez[(k + 1) % self._nz] - self._cordinatez[k],
        dz2 =       self._cordinatez[(k - 1) % self._nz] - self._cordinatez[k],
        return dr1, dr2, dtheta1, dtheta2, dz1, dz2
    def d_cordinate_center(self, cord):
        dr1, dr2, dtheta1, dtheta2, dz1, dz2 = self.d_cordinate(cord)
        return ((dr1+dr2)/2, (dtheta1 + dtheta2)/2, (dz1+dz2)/2)

    def get_neighbour_length3d(self, cord):
        dr1, dr2, dtheta1, dtheta2, dz1, dz2 = self.d_cordinate(cord)
        r = self._cordinatex[cord[0]]
        ret = ( 
            dr1,
            dr2, 
            r * dtheta1,
            r * dtheta2,
            dz1,
            dz2,
        )
        return ret

    def get_neighbour_area3d(self, cord):
        dr, dtheta, dz = self.d_cordinate_center(cord)
        r = self._cordinatex[cord[0]]
        ret = ( (r + dr/2) * dtheta * dz,
                (r - dr/2) * dtheta * dz,
                dr * dz,
                dr * dz,
                r * dtheta * dr,
                r * dtheta * dr,
        )
        return ret
    def get_volumn3d(self, cord):
        dr, dtheta, dz = self.d_cordinate_center(cord)
        r = self._cordinatex[cord[0]]
        return r * dtheta * dr * dz
    def get_upper_surface(self, melted_set):
        def _find_upper(idx):
            while idx in melted_set:
                idx += 1#step
            return idx
        self._upper_boundary = [_find_upper(idx) for idx in self._upper_boundary ]
        return self._upper_boundary

