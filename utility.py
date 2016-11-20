import numpy as np
import struct
import os
import math
import pickle
import constant as Const
import itertools as iter
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
    area = vsum / 0.079
    #area = math.pi * r * r
    ret = 4 * area * hcoef  * (selft - mp)
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
                              Const.dict['fuel_cp'], Const.dict['clad_cp'], Const.dict['gray_cp'],
                              Const.dict['fuel_mp'], Const.dict['clad_mp'], Const.dict['gray_mp'],
                              Const.dict['gravity'], Const.dict['core_height'], Const.dict['rod_radious'], 
                              Const.dict['reference_ST_number'], imping_temp)
    def mass_oxide(self):
        return self.fuel_mass
    def mass_metal(self):
        return self.gray_mass + self.clad_mass
    def mass_sum(self):
        return self.fuel_mass + self.clad_mass + self.gray_mass
    def mp(self):
        m = self.mass_sum()
        if m < 1.e-8 :
            return 0
        else:
            return Const.dict['fuel_mp'] * self.fuel_mass / m +\
               Const.dict['clad_mp'] * self.clad_mass / m +\
               Const.dict['gray_mp'] * self.gray_mass / m
    def toString(self):
        return '[fuel] %e [clad] % e [gray] %e' % (self.fuel_mass, self.clad_mass, self.gray_mass)
 
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
    vol = math.pi * Const.dict['rod_radious'] ** 2 * Const.dict['core_height']
    m = vol * Const.dict['fuel_dense']
    m *= 17 * 17
    while True:
        now_water = np.interp(t, water_history[:, 0], water_history[:, 1])
        now_power = np.interp(t, power_history[:, 0], power_history[:, 1])
        bottom_temp = np.interp(t, bottom_history[:, 0], bottom_history[:, 1])
        now_power_distribution = (basic_power_sum * now_power / m)  * np.array(radial_factor) 
        if t >= restart_time:
            yield  now_water, bottom_temp,  now_power_distribution
        t += t_step

def record_mass(mass_record, drop_list):
    for iass, item in drop_list.items():
        mass_record.get(iass)[0] += drop_list[iass].mass_sum()
        mass_record.get(iass)[1] += drop_list[iass].mass_metal()

@jit(float64(float64, float64, float64, float64, float64), nopython=True)
def calcGr(dT, L, beta, niu, rou):
    dT = abs(dT)
    g = 9.8
    if dT < 1.e-10 or L < 1.e-10:
        print ('Gr is zero or negative')
    return g*beta*dT*(L**3) /((niu/rou)**2) 

@jit(float64(float64, float64, float64, float64, float64, float64), nopython=True)
def calcRa(dT, L, beta, niu, rou, Pr):
    Gr = calcGr(dT, L, beta, niu, rou)
    return Gr * Pr

@jit(float64(float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
def calcRa_p(dT, L, beta, niu, rou, Pr, Q, lamda):
    g = 9.8
    return Pr * (g * beta * Q * L ** 5) / (lamda * niu ** 2)

@jit(float64(float64, float64, float64, float64), nopython=True)
def calcMoltenMetalFlux(Ra, Pr, lamda, L):
    if not 0.02 < Pr < 8750:
        print 'Pr didnt confont correlation'
    if not 3e5 < Ra < 7e9:
        print 'Ra didnt confont correlation'
    Nu = 0.069 * Ra ** (1/3) * Pr ** (0.074)
    return Nu * lamda / L

@jit(float64(float64, float64, float64, float64), nopython=True)
def calcMoltenOxideFlux(Ra_p, Q, V, A):
    if not  Ra_p < 2e16:
        print 'Ra_p didnt confont correlation',  Ra_p
        Ra_p = 2e16
    if not  Ra_p > 1e12:
        Ra_p = 1e12
        print 'Ra_p didnt confont correlation',  Ra_p
        return 0
    #Nu = 0.1857 * Ra_p ** 0.2304
    #return Nu * lamda / L
    ratio = 0.63 * Ra_p ** 0.0333
    ret = Q * V / (A *(1 + ratio))
    if ret > 1.e9:
        ret = 0.
    #print Q, V, A, ratio
    #print ret
    return ret

 
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

 
def calc_hcoef(T, L, Tf):
    Gr = calcGr(T - Tf, L, Const.dict['beta_steam'], Const.dict['niu_steam'], Const.dict['rou_steam'])
    Pr_fluid = Const.dict['Pr_steam']
    Pr_wall = Const.dict['Pr_wall']
    return calcSteamHeatTransferRateCore2(Gr, Pr_fluid, Pr_wall, L, Const.dict['lambda_steam'])

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
    area = math.pi * r * r / 4
    epsi = Const.dict['core_epsi']
    qrad = sigma * epsi * (board_t ** 4 - bottom_t ** 4 ) / (1 / epsi + 1 / epsi - 1 )
    qcov = h * (board_t - bottom_t)
    return  qrad + qcov

#TODO OPTIMIZE!
def calc_drop_heat(drop_list, assembly_id, drop_point_temp):
    #assert isinstance(drop_list, dict)
    #assert isinstance(assembly_id, dict)
    drop_heat_for_each_assembly = {}
    for iass, temp in drop_point_temp.items():
        drop_heat_for_each_assembly[iass] = drop_list[iass].drop_heat(temp)
    sum_heat = 0
    for (iass, heat) in drop_heat_for_each_assembly.items():
        sum_heat += heat
    if abs(sum_heat) < 1.e-5:
        return list(), list()
    else:
        sum_mass = sum([item.mass_sum() for (iass, item) in drop_list.items() ])
        print_with_style('[drop]', mode = 'bold', fore = 'purple')
        rod_imping_heat = {}
        for iass, idxs in assembly_id.items():
            for idx in idxs:
                rod_imping_heat[idx] = 0
        for iass, assemblyHeat in drop_heat_for_each_assembly.items():
            for idx in assembly_id[iass]:
                rod_imping_heat[idx] += assemblyHeat / len(assembly_id[iass])
        for k, v in rod_imping_heat.items():
            if abs(v) < 1.e-8:
                del rod_imping_heat[k]
        log('drop heat %e mass %e per_posi %e' % (sum_heat, sum_mass, sum_heat / len(rod_imping_heat)) )
        #print rod_imping_heat.values()
        return rod_imping_heat.keys(), np.array(rod_imping_heat.values())

def calc_decay_heat(power_distribution, mass_record, pool_volumn):
    sum = 0
    for iass, mass in mass_record.items():
        sum += mass[0] * power_distribution[iass - 1]
    return sum / pool_volumn

def calc_pool_heat(drop_list, T_up, surface_idx, pool_volumn, pool_area, decay_heat):
    iter_mass = iter.imap(lambda item: (item.mass_oxide(), item.mass_metal()), drop_list.values())
    mass_oxide, mass_metal = reduce(lambda (a, b), (c, d): (a+c, b+d), iter_mass)
    mass_sum = mass_oxide + mass_metal
    T_corium = sum(iter.imap(lambda item: item.mp(), drop_list.values()))

    Q = decay_heat
    Q /= pool_volumn
    H = pool_volumn / pool_area
    H /= 2
    Ra_p = calcRa_p(70, H, #T_up - T_corium, H, 
                    Const.dict['liquid_oxide_beta'], Const.dict['liquid_oxide_niu'], Const.dict['liquid_oxide_rou'], 
                    Const.dict['liquid_oxide_Pr'], Q, Const.dict['liquid_oxide_lambda'])
    print 'pool Height %f area %f' % (H, pool_area)
    return 0. - calcMoltenOxideFlux(Ra_p, Q, pool_volumn, pool_area)
    #return h * 70#(T_up - T_corium)

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
            status['time_step'] += 1
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

cacher = {}
def function_cacher(func):
    def wrapper(*args, **kw):
        if func.__name__ not in cacher:
            cacher[func.__name__] = {}
        cache_dict = cacher[func.__name__]
        if args not in cache_dict:
            ret = func(*args, **kw)
            cache_dict[args] = ret
            return ret
        else: 
            return cache_dict[args]
    return wrapper
        
 
