import numpy as np
import struct
import os
import math
import pickle
import constant as Const
from bintrees import RBTree
from numba import jit
from numba import float64, int64
import sys

@jit (float64(float64, float64, float64, 
     float64, float64, float64,
     float64, float64, float64,
     float64, float64, float64,
     float64, float64, float64, 
     float64, float64))
def _calc_drop_heat(mf, mc, mg, rf, rc, rg, cpf, cpc, cpg, mpf, mpc, mpg, g, height, r, St, selft):
    mass_sum = mf + mc + mg 
    if mass_sum < 1.e-8:
        return 0
    vsum = mf / rf + \
           mc / rc + \
           mg / rg
    dense = mass_sum / vsum
    cp = (mf * cpf) / mass_sum + \
         (mc * cpc) / mass_sum + \
         (mg * cpg) / mass_sum

    mp = (mf * mpf) / mass_sum + \
         (mc * mpc) / mass_sum + \
         (mg * mpg) / mass_sum

    height /= 2
    velocity = math.sqrt(2 * g * height)
    hcoef = St * dense * velocity * cp
    area = vsum / height
    #area = math.pi * r * r
    ret =  4 * 17 * 17 * area * hcoef  * (selft - mp)
    #print 'impingmeng hcoef %e heat %e' % (hcoef, ret)
    return ret

class Drop_mass(object):
    def __init__(self, f, c, g):
        self.fuel_mass = f
        self.clad_mass = c
        self.gray_mass = g
    def drop_heat(self, imping_temp):
        return _calc_drop_heat(self.fuel_mass, self.clad_mass, self.gray_mass,
                              Const.dict['fuel_dense'], Const.dict['clad_dense'], Const.dict['gray_dense'],
                              Const.dict['fule_cp'], Const.dict['clad_cp'], Const.dict['gray_cp'],
                              Const.dict['fule_mp'], Const.dict['clad_mp'], Const.dict['gray_mp'],
                              Const.dict['gravity'], Const.dict['core_height'], Const.dict['rod_radious'], 
                              Const.dict['reference_ST_number'], imping_temp)
    def mass_sum(self):
        return self.fuel_mass + self.clad_mass + self.gray_mass
    def toString(self):
        return '[fule] %e [clad] % e [gray] %e' % (self.fuel_mass, self.clad_mass, self.gray_mass)
 
def parse_input_file(filename, restart_step):
    f = open(filename, 'r')
    line_counter = 0
    ret = {}
    t = 0
    for line in f:
        if line_counter % 53 == 0:
            if t  >= restart_step and line_counter != 0:
                yield t, ret
            t = float(line)
            ret = {}
        else:
            arr = line.split()
            iass = int(arr[0])
            ret[iass] =  Drop_mass(float(arr[1]), float(arr[2]), float(arr[3]))
        line_counter += 1
    yield t, ret
 
def core_status_generator(t_start, t_step, restart_time):
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
        if t >= restart_time:
            yield  now_water, bottom_temp,  now_power_distribution
        t += t_step

@jit(float64(float64, float64), nopython=True)
def calSteamGr(dT,L):
    dT = abs(dT)
    beta = 0.00268
    g = 9.8
    niu = 12.37e-6
    rou = 0.598
    if dT < 1.e-10 or L < 1.e-10:
        print ('Gr is zero or negative')
    return g*beta*dT*(L**3) /((niu/rou)**2) 

 
@jit(float64(float64, float64, float64, float64), nopython=True)
def calcSteamHeatTransferRateCore(Gr, Prf, L, lamda):
    mul = Gr*Prf
    Nu = 0.0
    if Gr < 6.37e5 or Gr > 1.12e8:
        pass
    Nu = 0.747 * (mul) ** (1./6.)
    return   Nu * lamda / L

@jit(float64(float64, float64, float64, float64, float64), nopython=True)
def calcSteamHeatTransferRateCore2(Gr, Prf, Prw, L, lamda):
    mul = Gr*Prf
    Nu = 0.0
    if mul<1e3:
        pass
        #uti.mpi_print('Gr, Pr steam  didnt confront Correlation\n Pr * Gr == %f!' , mul, my_rank)
    if 1e3 < mul < 1e10:
        Nu = 0.6 * (mul)**0.25 * (Prf/Prw) ** (0.25)
    if mul >=1e10:
        Nu = 0.15 * (mul)**0.333 * (Prf/Prw) ** (0.25)
    return   Nu * lamda / L

def calcSteamHeatTransferRate(Gr, Prf, Prw, L): 
    return calcSteamHeatTransferRateCore2(Gr, Prf, Prw, L, Const.dict['lambda_steam'])
 
def calc_hcoef(T, L, Tf):
    Gr = calSteamGr(T - Tf, L)
    Pr_fluid = Const.dict['Pr_steam']
    Pr_wall = Const.dict['Pr_wall']
    h = calcSteamHeatTransferRate(Gr, Pr_fluid, Pr_wall, L)
    return h

def calc_drop_volumn(drop_list):
    fuel_dense = Const.dict['fuel_dense']
    clad_dense = Const.dict['clad_dense']
    gray_dense = Const.dict['gray_dense']
    drop_vol = 0
    for item in drop_list.values():
        #print item.toString()
        drop_vol += item.fuel_mass / fuel_dense
        drop_vol += item.clad_mass / clad_dense
        drop_vol += item.gray_mass / gray_dense
    return drop_vol

def calc_core_flux(bottom_t, board_t, h):
    #distance = Const.dict['bottom_board_distance']
    sigma = Const.dict['bottom_sigma']
    epsi = Const.dict['bottom_epsi']
    r = Const.dict['board_radious']
    area = math.pi * r * r
    epsi = Const.dict['core_epsi']
    qrad = sigma * epsi * (board_t ** 4 - bottom_t ** 4 ) / (1 / epsi + 1 / epsi - 1 )
    qcov = h * (board_t - bottom_t)
    return  qrad + qcov

#TODO OPTIMIZE!
def calc_drop_heat(drop_list, assembly_id, drop_point_temp):
    #assert isinstance(drop_list, dict)
    #assert isinstance(assembly_id, dict)
    drop_heat_for_each_assembly = [(iass, item.drop_heat(temp)) for ((iass, item), (iass2, temp)) in zip( drop_list.items(), drop_point_temp.items())] 
    sum_heat = 0
    for (iass, heat) in drop_heat_for_each_assembly:
        sum_heat += heat
    sum_mass = sum([item.mass_sum() for (iass, item) in drop_list.items() ])
    print_with_style('[drop]', mode = 'bold', fore = 'purple')
    rod_imping_heat = {}
    for iass, idxs in assembly_id.items():
        for idx in idxs:
            rod_imping_heat[idx] = 0
    for (iass, assemblyHeat) in drop_heat_for_each_assembly:
        for idx in assembly_id[iass]:
            rod_imping_heat[idx] += assemblyHeat / len(assembly_id[iass])
    for k, v in rod_imping_heat.items():
        if abs(v) < 1.e-8:
            del rod_imping_heat[k]
    log( 'drop heat %e mass %e per_posi %e' % (sum_heat, sum_mass, sum_heat / len(rod_imping_heat)) )
    if abs(sum_heat) < 1.e-5:
        return None, None
    else:
        #print rod_imping_heat.values()
        return rod_imping_heat.keys(), np.array(rod_imping_heat.values())

def calc_pool_heat(drop_list, pool_area):
    return 0.0

def calc_hole_volumn():
    r = Const.dict['rod_radious']
    areahole = math.pi * r ** 2
    nrod = len(Const.assembly_pos)
    return nrod * areahole * Const.dict['board_thick']
 
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
    return ret

def save(status, temp_array):
    title = 'sav/rst_%d.npy' % status['time_step']
    _f = open(title,'wb')
    #data = struct.pack('i',step)
    pickle.dump(status, _f)
    pickle.dump(list( status['melted_set_tree'].items() ), _f)
    np.save(_f, temp_array)

def load(temp_array):
    ret = os.popen("ls -lt sav/ | awk '{print $9}'").read()
    arr = ret.split()
    if len(arr) > 0 :
        filename = arr[0]
        print 'restarting from %s... ?' % filename
        input = raw_input()
        while input != 'yes' and input != 'y':
            input = raw_input()
        try: 
            _f = open('sav/'+filename)
            status = pickle.load(_f)
            tree = RBTree()
            for k,v in pickle.load(_f):
                tree[k] = v
            status['melted_set_tree'] = tree
            temp_array[:]  = np.load(_f)[:] 
            print status
            return status
        except IOError:
            print 'cannot open file %s... abort' % filename 
            exit()
    else:
        print 'new start'

STYLE = {
    'fore':
     {  
         'black'    : 30,   
         'red'      : 31,  
         'green'    : 32, 
         'yellow'   : 33,
         'blue'     : 34, 
         'purple'   : 35, 
         'cyan'     : 36, 
         'white'    : 37, 
     },
     
     'back' :
     {  
         'black'     : 40, 
         'red'       : 41, 
         'green'     : 42, 
         'yellow'    : 43, 
         'blue'      : 44, 
         'purple'    : 45, 
         'cyan'      : 46,
         'white'     : 47,
     },
     
     'mode' :
     { 
         'mormal'    : 0, 
         'bold'      : 1,
         'underline' : 4,
         'blink'     : 5,
         'invert'    : 7,
         'hide'      : 8,
     },
     
     'default' :
     {
         'end' : 0,
     },
 }
 
 
def print_with_style(string, mode = '', fore = '', back = ''):
    mode  = '%s' % STYLE['mode'][mode] if STYLE['mode'].has_key(mode) else ''
    fore  = '%s' % STYLE['fore'][fore] if STYLE['fore'].has_key(fore) else ''
    back  = '%s' % STYLE['back'][back] if STYLE['back'].has_key(back) else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end   = '\033[%sm' % STYLE['default']['end'] if style else ''
    sys.stdout.write('%s%s%s' % (style, string, end))
    logfile.write(string.strip() + '\n')

logfile = open('log', 'w')
def log(str):
    text = str.strip() + '\n'
    logfile.write(text)
    sys.stdout.write(text)

    
 
