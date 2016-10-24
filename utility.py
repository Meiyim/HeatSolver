import sys
import numpy as np
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
            ret = []
        else:
            to_append = {}
            arr = line.split('\t')
            ret[float(arr[0])] = Drop_mass(float(arr[1]), float(arr[2]), float(arr[3])
        line_counter += 1
    yield t, ret
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
    if Gr < 6.37e5 or Gr <1.12e8:
        print 'Gr did not confront corelation'
    Nu = 0.747 * (mul) ** (1./6.)
    return   Nu * lamda / L

def calc_hcoef(T, L, Tf):
    Gr = calSteamGr(T - Tf)
    Pr_fluid = Const.dict['Pr_steam']
    Pr_wall = Const.dict['Pr_wall']
    h = calcSteamHeatTransferRate(Gr, Pr_fluid, Pr_wall, L)
    return h

def calc_melted_volumn(mask, mesh):
    vol = 0
    for idx in mask:
        vol += mesh.get_neighbour(idx)
    return volk

def calc_drop_volumn(drop_list):
    fuel_dense = Const.dict['fuel_dense']
    clad_dense = Const.dict['clad_dense']
    gray_dense = Const.dict['gray_dense']
    drop_vol = 0
    for item in drop_list.values():
        drop_vol += item.fuel_mass / fuel_dense
        drop_vol += item.clad_mass / clad_dense
        drop_vol += item.gray__mass / gray_dense
    return drop_vol

def calc_total_up_heat(drop_list):
    r = Const.dict['board_radious']
    sum = 0
    for item in drop_list.values():
        sum += item.sum()
    return 
def 
