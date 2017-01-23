import math
import numpy as np
import itertools as iter
import itertools
import utility as uti
import constant as Const
from copy import deepcopy
from numba import jit
from numba import int64, float64

#declare
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
    def get_position(self, idx):
        pass
    def get_volumn(self, i):
        pass
    def get_neighbour(self, i):
        pass
    def get_neighbour_lenth(self, i):
        pass
    def get_neighbour_coef(self, i): pass 
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
    def get_3d_index(self, idx):
        if idx >= self._ncell or idx < 0 :
            return None
        else:
            i = idx / (self._ny * self._nz)
            yu = idx % (self._ny * self._nz)
            j =  yu / self._nz
            k =  yu % self._nz
            return  i, j, k
    def get_position(self, idx):
        cord = self.get_3d_index(idx)
        return self.get_position3d(cord)
    def get_position3d(self, cord):
        pass

    def get_index(self, cord): #None able
        x, y, z = cord
        if 0 <= x < self._nx and 0<= y < self._ny and 0 <= z < self._nz :
            return x * self._ny * self._nz + \
                   y * self._nz + \
                   z
        else:
            return None
    def get_neighbour3d(self, cord):
        i, j, k = cord
        return (self.get_index((i+1, j, k)),
                self.get_index((i-1, j, k)),
                self.get_index((i, j+1, k)),
                self.get_index((i, j-1, k)),
                self.get_index((i, j, k+1)),
                self.get_index((i, j, k-1)),
               )
    def get_neighbour(self, idx):
        cord = self.get_3d_index(idx)
        ret = self.get_neighbour3d(cord)
        return ret

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
        dz /= 2
        uti.log('begin %f end %f ratio %e' % (dr, r-dr, qr) )
        super(CylinderlMesh, self).__init__(ir, it, iz, dr, r-dr, 0., 0.5 * math.pi, dz, z - dz)
        self._cordinatex = uti.stupid_method(dr, r-dr, qr, ir)
        self._cordinatex = np.hstack((self._cordinatex, np.array((r+dr, -dr))))
        self._cordinatez = np.hstack((self._cordinatez, np.array((z+dz, -dz))))
        uti.log('CHECK CORDINATE R: %s' % str(self._cordinatex))
        print self._cordinatey
        upper_bound = set(
            [self.get_index(cord)  for cord in itertools.product(xrange(0, ir), xrange(0, it), xrange(iz - 1, iz)) ]
        )
        self._upper_boundary = upper_bound 
        self.set_region_3d('up', itertools.product(xrange(0, self._nx), xrange(0, self._ny), xrange(iz - 1, self._nz)))
        self.set_region_3d('down', itertools.product(xrange(0, self._nx), xrange(0, self._ny), xrange(0, 1)))
        self.set_region_3d('side', itertools.product(xrange(ir-1, self._nx), xrange(0, self._ny), xrange(0, self._nz)))


    def get_position3d(self, cord):
        ix, iy, iz = cord
        r = self._cordinatex[ix]
        theta = self._cordinatey[iy]
        z = self._cordinatez[iz]
        return r * math.cos(theta), r * math.sin(theta), z

    def get_neighbour3d(self, cord):
        i, j, k = cord
        return (
                self.get_index((i+1, j, k)),
                self.get_index((i-1, j, k)),
                self.get_index((i, (j+1)%self._ny, k)),
                self.get_index((i, (j-1)%self._ny, k)),
                self.get_index((i, j, k+1)),
                self.get_index((i, j, k-1)),
            )
    def get_bottom_index_at_position(self, pos): #optimizeable
        x, y = pos
        epsi = 1.e-9
        r = math.sqrt(x**2 + y**2)
        theta = math.atan(y / (x + epsi))
        assert -epsi  < theta < 0.5 * math.pi + epsi
        dt = int((theta - self._cordinatey[0]) / (self._cordinatey[1] - self._cordinatey[0])) 
        dr = 0
        for the_r in self._cordinatex:
            dr += 1
            if the_r > r:
                break
        return (dr, dt)
 
    def d_cordinate(self, cord):
        i, j, k = cord
        dr1 =       self._cordinatex[i + 1] - self._cordinatex[i]
        dr2 =       self._cordinatex[i] - self._cordinatex[i - 1]
        dtheta1 =   self._cordinatey[1] - self._cordinatey[0]
        dtheta2 =   self._cordinatey[1] - self._cordinatey[0]
        dz1 =       self._cordinatez[k + 1] - self._cordinatez[k]
        dz2 =       self._cordinatez[k] - self._cordinatez[k - 1]
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

    def down_step(self, idx):
        return idx - 1 if self.get_3d_index(idx)[2] > 0 else None
    def add_vertical_surface_and_area(self, ret, melted_set_sum, pool_idxs):
        pool_area = sum([self.get_neighbour_area(idx)[4] for idx in ret])
        iter_idx = iter.imap(lambda idx : self.get_neighbour(idx), pool_idxs)
        '''
        iter_area = iter.imap(lambda idx : self.get_neighbour_area(idx), pool_idxs)
        iter2 = iter.imap(lambda (idxs, area): ((idxs[0], area[0]), (idxs[1], area[1]), (idxs[2], area[2]), (idxs[3], area[3])), zip(iter_idx, iter_area))
        for (i1, a1), (i2, a2), (i3, a3), (i4, a4) in iter2:
            if i1 in ret or i2 in ret or i3 in ret or i4 in ret:
                continue
            if i1 is not None and i1 not in melted_set_sum:
                ret.add(i1)
                pool_area += a1
            if i2 is not None and i2 not in melted_set_sum:
                ret.add(i2)
                pool_area += a2
            if i3 is not None and i3 not in melted_set_sum:
                ret.add(i3)
                pool_area += a3
            if i4 is not None and i4 not in melted_set_sum:
                ret.add(i4)
                pool_area += a4
        '''
        for i1, i2, i3, i4, i5, i6 in iter_idx:
            if i1 in ret or i2 in ret or i3 in ret or i4 in ret:
                continue
            if i1 is not None and i1 not in melted_set_sum:
                ret.add(i1)
            if i2 is not None and i2 not in melted_set_sum:
                ret.add(i2)
            if i3 is not None and i3 not in melted_set_sum:
                ret.add(i3)
            if i4 is not None and i4 not in melted_set_sum:
                ret.add(i4)
        #pool_area /= 4
        return ret, pool_area

    def get_upper_surface(self, melted_set_sum):
        def _find_upper(idx):
            while idx in melted_set_sum:
                idx = self.down_step(idx)#step
            return idx
        self._upper_boundary = set([_find_upper(idx) for idx in self._upper_boundary ])
        if None in self._upper_boundary: self._upper_boundary.remove(None)
        ret = deepcopy(self._upper_boundary)
        self.add_vertical_surface_and_area(ret, melted_set_sum, melted_set_sum)
        return ret

    def get_pool_bottom(self, status):
        melted_set = status['melted_set'] 
        melted_set_tree = status['melted_set_tree']
        pool_volumn = status['pool_volumn'] - uti.calc_hole_volumn()
        # insert
        iter_cord = iter.imap(lambda idx : (self.get_3d_index(idx), idx), melted_set)
        #make h first
        cord_idx_pair = [ ((z, y, x), idx) for ((x, y ,z), idx) in iter_cord]
        for (cord, idx) in cord_idx_pair:
            melted_set_tree[cord] = idx

        ret = None
        if len(melted_set_tree) == 0 :
            print 'pool covering not melted yet'
            ret = deepcopy(self._upper_boundary) 
            r = Const.dict['board_radious']
            area = math.pi * r ** 2 / 4
            return ret, area
        else:
            vol = 0
            pool_idxs  = set()
            for cord, idx in melted_set_tree.items():
                vol += self.get_volumn(idx)
                if vol > pool_volumn:
                    break
                else:
                    pool_idxs.add(idx)
            lowest =  melted_set_tree.keys().next()
            penetrate_iz = lowest[0] - 1
            uti.log('penetrate-deep %e idx %d lowest-pool %s' % (self._cordinatez[penetrate_iz], penetrate_iz, str(lowest)))
            if vol > pool_volumn:
                uti.log('pool covering')
                r = Const.dict['board_radious']
                area = math.pi * r ** 2 / 4
                return deepcopy(self._upper_boundary), area
            else:
                uti.log('pool not cover yet')
                ret = set([ self.down_step(idx) for idx in pool_idxs]) & self._upper_boundary
                return self.add_vertical_surface_and_area(ret, status['melted_set_sum'], pool_idxs)


    def calc_melted_volumn(self, melted_set):
        return sum(map(lambda idx: self.get_volumn(idx), melted_set))

    def get_drop_point_idx(self, xy_idx):
        upper_surface_idx = self._upper_boundary
        xyz_upper = iter.imap(lambda idx: self.get_3d_index(idx), upper_surface_idx)
        xy_dict = dict([ ((x, y), z) for (x, y, z) in xyz_upper])
        assert len(xy_dict) == len(upper_surface_idx)
        ret = {}
        for iass, idxs in xy_idx.items():
            arr = []
            for (x,y) in idxs:
                if (x, y) in xy_dict: arr.append(self.get_index((x, y, xy_dict[x, y])))
            ret[iass] = arr
        return ret

    def tecplot_str(self, var, status, boundary_idx):
        tec_text = []
        tec_text.append('title = supporting board')
        tec_text.append('ZONE I=%d, J=%d, K=%d, F=point' % (self._nz, self._ny, self._nx + 1))
        #a fake center
        center_idx  = [self.get_index((0, 0, k)) for k in xrange(0, self._nz)]
        varcenter = np.array([var[idx] for idx in center_idx])
        statcenter = map(lambda idx: 0 if idx in status['melted_set_sum'] else 1, center_idx)
        for j in xrange(0, self._ny):
            for k in xrange(0, self._nz):
                tec_text.append('%e %e %e %e %d' % (0., 0., 0., varcenter[k], statcenter[k]))
        for i in xrange(0, self._nx):
            for j in xrange(0, self._ny):
                for k in xrange(0, self._nz):
                    idx = self.get_index((i, j, k))
                    stat = 0 
                    if idx in boundary_idx:
                        stat = 1
                    elif idx in status['melted_set_sum']:
                        stat = 2
                    tec_text.append('%e %e %e %e %d' % (self.get_position3d((i, j, k)) + (var[idx], stat)))
        return '\n'.join(tec_text)

