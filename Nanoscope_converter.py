import re
import numpy as np
import csv
import itertools
#import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # Savitzky Golay filter for smoothing data with too much noise
import datetime

def nanoscope_converter(file_path):
    with open(file_path, 'rb') as path:
        path.seek(40960) # where the binary data begins
        # testo = str(path.read()) # useless
        all_deflections = np.fromfile(path, dtype=np.int16)
        # ha trovato i valori in LSB delle deflection sia approach che retraction infatti è lunga 8192
        path.seek(0)
        header_info = str(path.read())

    # find z sensitivity in nm/V
    index = header_info.find('@Sens. Zsens: V')
    sensitivity = float(re.findall("\d+\.\d+", header_info[index:])[0])

    # find the size of the probed area and the x and y position of the curve
    index = header_info.find('\Scan Size: ')
    scan_size = float(re.findall("\d+", header_info[index:])[0])
    index = header_info.find('\X Offset: ')
    x_pos = float(re.findall("\d+", header_info[index:])[0])   # x position in nm (* -1 is for displaying them correctly)
    index = header_info.find('\Y Offset: ')
    y_pos = float(re.findall("\d+", header_info[index:])[0])   # y position in nm

    # find the indentation rate in Hz
    index = header_info.find('\Scan rate: ')
    ind_rate = float(re.findall("\d+\.\d+", header_info[index:])[0])

    # find the forward velocity in nm/s
    index = header_info.find('\Forward vel.: ')
    forw_vel = float(re.findall("\d+\.\d+", header_info[index:])[0])*100

    index = header_info.find('@4:Ramp size Zsweep: V')
    ramp_size_in_Volt = float(re.findall("\d+\.\d+", header_info[index:])[1]) # interested value is the second after @4:Ramp size..
    ramp_size_in_nm = sensitivity*ramp_size_in_Volt

    index = header_info.find('Samps/line: ', index)
    Rt_npoints = int(re.findall("\d+", header_info[index:])[0]) # n° of points in the retraction curve
    Ex_npoints = int(re.findall("\d+", header_info[index:])[1]) # n° of points in the approach curve
    Rt_step = ramp_size_in_nm/Rt_npoints # step increment along the retraction ramp
    Ex_step = ramp_size_in_nm/Ex_npoints # step increment along the approach ramp

    # building up the approach ramp array
    Ex_ramp = np.array(0)
    new, i = 0, 0
    while i < Ex_npoints-1:
        new = new + Ex_step
        Ex_ramp = np.append(Ex_ramp, new)
        i += 1

    # building up the retraction ramp array
    Rt_ramp = np.array(ramp_size_in_nm)
    i, new = 0, ramp_size_in_nm
    while i < Rt_npoints-1:
        new = new - Rt_step
        Rt_ramp = np.append(Rt_ramp, new)
        i += 1

    # building up the APPROACH DEFLECTION ARRAY
    Ex_deflection = all_deflections[0:Ex_npoints]
    index = header_info.find('@4:Z scale:')
    from_LSB_to_Volt = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # find the spring constant nN/nm
    index = header_info.find('Spring Constant:')
    spring_constant = float(re.findall("\d+\.\d+", header_info[index:])[0])
    #find the Deflection sensitivity  nm/V
    index = header_info.find('@Sens. DeflSens: V')
    defl_sensitivity = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # deflection in nm and force in pN arrays for approach
    Ex_deflection = Ex_deflection*from_LSB_to_Volt*defl_sensitivity
    Ex_deflection = np.flip(Ex_deflection) # inverting the array, it's in the wrong sense
    Ex_force = Ex_deflection*spring_constant*1000


    # building up the RETRACTION DEFLECTION ARRAY
    Rt_deflection = all_deflections[Rt_npoints:2*Rt_npoints]
    # deflection in nm and force in pN arrays for retraction
    Rt_deflection = Rt_deflection*from_LSB_to_Volt*defl_sensitivity
    Rt_force = Rt_deflection*spring_constant*1000

    #------SAVITZKY GOLAY FILTERING IN THE CASE THE SIGNAL PRESENTS TOO MUCH NOISE
    Ex_deflection_filtered = savgol_filter(Ex_deflection, 7, 3)
    Rt_deflection_filtered = savgol_filter(Rt_deflection, 7, 3)
    # Ex_deflection_filtered = Ex_deflection
    # Rt_deflection_filtered = Rt_deflection

    Ex_force_filtered = Ex_deflection_filtered*spring_constant*1000
    Rt_force_filtered = Rt_deflection_filtered*spring_constant*1000
    Rt_force = np.flip(Rt_force)
    Rt_force_filtered = np.flip(Rt_force_filtered)


    # calculating the baseline in order to shift to zero the initial force values
    sum = 0
    Rt_sum = 0
    for i in range(1000):
        sum = sum + Ex_force[i]
        Rt_sum = Rt_sum + Rt_force[i]

    Ex_force_shifted = Ex_force - sum/1000
    Rt_force_shifted = Rt_force - Rt_sum/1000

    Ex_force_filtered = Ex_force_filtered - sum/1000
    Rt_force_filtered = Rt_force_filtered - Rt_sum/1000


    # EXPOPORTING A TEXT FILE OF THE RAW DATA
    # newpath = str(file_path) + '_rawCurveInfo.txt'
    # with open(newpath, 'w') as text:
    #     text.write("Calc_Ramp_Ex_nm     Calc_Ramp_Rt_nm     Defl_nm_Ex     Defl_nm_Rt     Defl_pN_Ex     Defl_pN_Rt\n")
    #     for xyz in itertools.zip_longest(Ex_ramp, Rt_ramp, Ex_deflection, Rt_deflection, Ex_force, Rt_force, fillvalue=''):
    #         text.write('%-20s   %-20s   %-20s   %-20s   %-20s   %-20s\n'%xyz)
    # #exporting a text file of the separation vs Force both Ex and Rt
    # newpath2 = str(file_path) + '_Ex_ForceSeparation.txt'
    # newpath3 = str(file_path) + '_Rt_ForceSeparation.txt'
    # with open(newpath2, 'w') as text:
    #     text.write("Separation_Ex_nm     Force_Ex_pN\n")
    #     for xyz in itertools.zip_longest(Ex_ramp-Ex_deflection,Ex_force, fillvalue=''):
    #         text.write('%-20s   %-20s\n'%xyz)
    #
    # with open(newpath3, 'w') as text:
    #     text.write("Separation_Rt_nm     Force_Rt_pN\n")
    #     for xyz in itertools.zip_longest(Rt_ramp-Rt_deflection,Ex_force, fillvalue=''):
    #         text.write('%-20s   %-20s\n'%xyz)

    Ex_separation = np.array(Ex_ramp-Ex_deflection)
    Rt_separation = np.array(Rt_ramp-Rt_deflection)

    Ex_separation_filtered = np.array(Ex_ramp-Ex_deflection_filtered)
    Rt_separation_filtered = np.array(Rt_ramp-Rt_deflection_filtered)


    Ex_max = np.max(Ex_separation)
    aligned_Ex_sep = np.array((Ex_separation - Ex_max)*-1)
    Rt_max = np.max(Rt_separation)
    aligned_Rt_sep = np.flip(np.array((Rt_separation - Rt_max)*-1))

    Ex_max_filtered = np.max(Ex_separation_filtered)
    aligned_Ex_sep_filtered = np.array((Ex_separation_filtered - Ex_max_filtered)*-1)
    Rt_max_filtered = np.max(Rt_separation_filtered)
    aligned_Rt_sep_filtered = np.flip(np.array((Rt_separation_filtered - Rt_max_filtered)*-1))


    curve = [aligned_Ex_sep, Ex_force_shifted, aligned_Rt_sep, Rt_force_shifted, Ex_ramp, Rt_ramp,
             aligned_Ex_sep_filtered, Ex_force_filtered, aligned_Rt_sep_filtered, Rt_force_filtered,
             forw_vel, ind_rate, scan_size, x_pos, y_pos]


    return curve
