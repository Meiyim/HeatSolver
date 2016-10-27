import numpy as np
import math
import constant as Const

class Drop_mass(object):
    def __init__(self, f, c, g):
        self.fuel_mass = f
        self.clad_mass = c
        self.gray_mass = g
    def sum(self):
        return self.fuel_mass + self.clad_mass + self.gray_mass

def parse_input_file(filename):
    f = open(filename, 'r')
    line_counter = 0
    ret = {}
    t = 0
    for line in f:
        if line_counter % 53 == 0:
            yield t, ret
            t = float(line)
            ret = {}
        else:
            arr = line.split()
            iass = int(arr[0])
            ret[iass] =  Drop_mass(float(arr[1]), float(arr[2]), float(arr[3]))
        line_counter += 1
    yield t, ret

def core_status_generator(t_start, t_step):
    t = t_start
    water_history = np.array(Const.water_history)
    power_history = np.array(Const.power_history)
    bottom_history = np.array(Const.bottom_history)
    radial_factor = np.array(Const.power_distribute)
    radial_factor /= sum(radial_factor)
    basic_power_sum = Const.dict['total_power']
    while True:
        now_water = np.interp(t, water_history[:, 0], water_history[:, 1])
        now_power = np.interp(t, power_history[:, 0], power_history[:, 1])
        bottom_temp = np.interp(t, bottom_history[:, 0], bottom_history[:, 1])
        now_power_distribution = basic_power_sum * now_power * np.array(radial_factor)
        yield  now_water, bottom_temp,  now_power_distribution
        t += t_step

def calSteamGr(dT,L):
    dT = abs(dT)
    beta = 0.00268
    g = 9.8
    niu = 12.37e-6
    rou = 0.598
    if dT < 1.e-10 or L < 1.e-10:
        print 'Gr is zero or negative: dt: %f, L: %f' %(dT,L)
    return g*beta*dT*(L**3) /((niu/rou)**2) 
def calcSteamHeatTransferRate(Gr, Prf, L):
    mul = Gr*Prf
    lamda = Const.dict['lambda_steam']
    Nu = 0.0
    if Gr < 6.37e5 or Gr > 1.12e8:
        print 'Gr did not confront corelation Gr: %e' % Gr
    Nu = 0.747 * (mul) ** (1./6.)
    return   Nu * lamda / L

def calc_hcoef(T, L, Tf):
    Gr = calSteamGr(T - Tf, L)
    Pr_fluid = Const.dict['Pr_steam']
    Pr_wall = Const.dict['Pr_wall']
    h = calcSteamHeatTransferRate(Gr, Pr_fluid, L)
    return h

def calc_melted_volumn(mask, mesh):
    return sum( map(lambda idx: mesh.get_volumn(idx), mask) )

def calc_drop_volumn(drop_list):
    fuel_dense = Const.dict['fuel_dense']
    clad_dense = Const.dict['clad_dense']
    gray_dense = Const.dict['gray_dense']
    drop_vol = 0
    for item in drop_list.values():
        drop_vol += item.fuel_mass / fuel_dense
        drop_vol += item.clad_mass / clad_dense
        drop_vol += item.gray_mass / gray_dense
    return drop_vol

def calc_core_flux(bottom_t, board_t):
    #distance = Const.dict['bottom_board_distance']
    sigma = Const.dict['bottom_sigma']
    epsi = Const.dict['bottom_epsi']
    r = Const.dict['board_radious']
    area = math.pi * r * r
    qrad = area * sigma * epsi * (bottom_t ** 4 - board_t ** 4)
    qcov = 0.0
    return  qrad + qcov


def calc_drop_heat(drop_list, assembly_id):
    assert isinstance(drop_list, dict)
    assert isinstance(assembly_id, dict)
    drop_heat_for_each_assembly = [ (iass, item.sum()) for (iass, item) in drop_list.items() ] 
    rod_idx = []
    drop_heat_for_rod = []
    for (iass, assemblyHeat) in drop_heat_for_each_assembly:
        drop_heat_for_rod += ([ assemblyHeat/len(assembly_id[iass]) ] * len(assembly_id[iass]) )
        rod_idx += assembly_id[iass]
    return rod_idx, drop_heat_for_rod


def calc_pool_heat(drop_list):
    r = Const.dict['board_radious']
    sum = 0
    for item in drop_list.values():
        sum += item.sum()
    sum /= (math.pi * r *r)
    return sum

def stupid_method(begin, end, q, n):
    p =  (1 - q ** n)/(1 - q) 
    ret = np.empty((n))
    s = end - begin
    ret[0] = s/p
    total = 0
    for i in xrange(1, n):
        ret[i] = ret[i-1] * q  + total
    for i in xrange(1, n):
        ret[i] += ret[i-1]
    print sum(ret)
    return ret

