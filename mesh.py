import sys
import math
import numpy as np
import itertools

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
        self.__ncell = ncell
        self.__region = {}
        self.__basic_material = None
    def get_ncell(self):
        return self.__ncell
    def get_index_at_poristion(pos):
        pass
    def get_region(self, name):
        return list(self.__region.get(name))
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
        self.__nx = x
        self.__cordinatex = np.linspace(x1, x2, x)
        self.__ny = y
        self.__cordinatey = np.zeros(y1, y2, y)
        self.__nz = z
        self.__cordinatez = np.zeros(z1, z2, z)
        self.__basic_material = None
    def set_basic_materal(self, mat):
        assert isinstance(mat, Material)
        self.__basic_material = mat
    def get_material_at_index(self, i): #temperally uniform material
        return self.__basic_material
    def set_region_3d(self, name, region):
        nodes_index = map(lambda val: self.get_index(val), region)
        assert None not in nodes_index
        self.__region[name] = set(nodes_index)
    def get_3d_index(self, i):
        if i >= self.__ncell or i < 0 :
            return None
        else:
            return  (i / (self.__ny * self.__nz), \
                     i / (self.nz), \
                     i )
    def get_index(self, cord): #None able
        x, y, z = cord
        if 0 <= x < self.__nx or 0<= y < self.__ny or 0 < z <= self.__nz :
            return x * self.__ny * self.__nz + \
                   y * self.__nz + \
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
        i, j, k = self.get_3d_index(idx)
        return self.get_neighbour3d(i, j, k)

    def get_neighbour_length3d(self, cord):
        i, j, k = cord
        ret =  ( self.__cordinatex[(i + 1) % self.__nx] - self.__cordinatex[i],
                 self.__cordinatex[(i - 1) % self.__nx] - self.__cordinatex[i],
                 self.__cordinatey[(j + 1) % self.__ny] - self.__cordinatey[j],
                 self.__cordinatey[(j - 1) % self.__ny] - self.__cordinatey[j],
                 self.__cordinatez[(k + 1) % self.__nz] - self.__cordinatez[k],
                 self.__cordinatez[(k - 1) % self.__nz] - self.__cordinatez[k],
               )
        return ret
    def get_neighbour_lenth(self, idx):
        cord = self.get_3d_index(idx)
        return self.get_neighbour_length3d(cord)
    def d_cordinate_center(self, cord):
        dx1, dx2, dy1, dy2, dz1, dz2 = self.get_neighbour_length3d(self, cord):
        return (dx1+dx2)/2, (dy1+dy2)/2, (dz1+dz2)/2
    def get_volumn3d(self, cord):
        dx, dr, dz = self.d_cordinate_center(cord)
        return dx * dy * dz
    def get_volumn(self, idx):
        cord = self.get_3d_index(idx)
        return get_volumn3d(cord)
    def get_neighbour_area3d(self, cord):
        dx, dr, dz = self.d_cordinate_center(cord)
        return dy*dz, dy*dz, dx*dz, dx*dz, dy*dz, dy*dz
    def get_neighbour_area(self, i):
        cord = self.get_3d_index(i)
        return self.get_neighbour_area3d(cord)
    def get_neighbour_coef(self, i):
        return self.__basic_material.lamda

class CylinderlMesh(StructuredMesh3D):
    def __init__(self, ir, it, iz, r, qr, z):
        dr = r / float(ir)
        dr /= 2
        dz = z / float(iz)
        nr = (math.log(r-dr) - math.log(dr)) * (math.log(b) / math.log(q) ) + 1
        nr = int(nr)
        super(CylinderlMesh, self).__init__(ir, it, iz, dr, r-dr, 0., math.pi, dz, z - dz)
        self.__cordinatex = np.logspace(math.log(dr), math.log(r - dr), nr, base = math.e) #rewrite
        self.set_region_3d('up', itertools(xrange(0, ir), xrange(0, it), xrange(iz - 0, iz))
        self.set_region_3d('down', itertools(xrange(0, ir), xrange(0, it), xrange(0, 1))
        self.set_region_3d('side', itertools(xrange(ir-1, ir), xrange(0, it), xrange(0, iz))

    def get_neighbour3d(self, cord):
        i, j, k = cord
        return map(lambda v : v is not None,
            (
                self.get_index((i+1, j, k)),
                self.get_index((i-1, j, k)),
                self.get_index((i, (j+1)%self.__ny, k)),
                self.get_index((i, (j-1)%self.__ny, k)),
                self.get_index((i, j, k+1)),
                self.get_index((i, j, k-1)),
            ))
    def get_index_at_poristion(pos):
        x, y, z = pos
        dz = int((z - self.__cordinatez[0]) / self.__cordinatez.shape[0])
        r = sqrt(x**2 + y**2)
        theta = math.acos( x/r )
        dt = int((theta - self.__cordinatey[0]) / self.__cordinatey.shape[0])
        dr = 0
        for the_r in self.__cordinatex:
            dr += 1
            if th_r > r:
                break;
        return self.get_index((dz, dt, dr))
 
    def d_cordinate(self, cord):
        i, j, k = cord
        dr1 =       self.__cordinatex[(i + 1) % self.__nx] - self.__cordinatex[i],
        dr2 =       self.__cordinatex[(i - 1) % self.__nx] - self.__cordinatex[i],
        dtheta1 =   self.__cordinatey[(j + 1) % self.__ny] - self.__cordinatey[j],
        dtheta2 =   self.__cordinatey[(j - 1) % self.__ny] - self.__cordinatey[j],
        dz1 =       self.__cordinatez[(k + 1) % self.__nz] - self.__cordinatez[k],
        dz2 =       self.__cordinatez[(k - 1) % self.__nz] - self.__cordinatez[k],
        return dr1, dr2, dtheta1, dtheta2, dz1, dz2
    def d_cordinate_center(self, cord):
        dr1, dr2, dtheta1, dtheta2, dz1, dz2 = self.d_cordinate(cord)
        return ((dr1+dr2)/2, (dtheta1 + dtheta2)/2, (dz1+dz2)/2)

    def get_neighbour_length3d(self, cord):
        dr1, dr2, dtheta1, dtheta2, dz1, dz2 = self.d_cordinate(cord)
        r = self.__cordinatex[cord[0]]
        ret = ( dr1,
                dr2, 
                r * dtheta1,
                r * dtheta2,
                dz1,
                dz2,
        )
        return ret

    def get_neighbour_area3d(self, cord):
        dr, dtheta, dz = self.d_cordinate_center(cord)
        r = self.__cordinatex[cord[0]]
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
        r = self.__cordinatex[cord[0]]
        return r * dtheta1 * dr * dz
        

