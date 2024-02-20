import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter # Savitzky Golay filter for smoothing data with too much noise

def nanoscope_converter(file_path):
    '''Parser for a force curve raw file. Takes the path to the file as input and outputs a list containing all
    relevant information about a curve for further analysis.'''

    # Read the curve file as a binary
    with open(file_path, 'rb') as path:
        # first part of the file is a header, put the index where the binary data starts
        path.seek(40960)
        # converting to integer values these data are in the format called "LeastSignificantBit" (LSB) and refer to
        # the deflection signal of both approach and retraction curves
        all_deflections = np.fromfile(path, dtype=np.int16)
        # extracting header info from first part
        path.seek(0)
        header_info = str(path.read())

    '''The header contains some info and values necessary for calculating the force curves, this info needs to be 
    parsed and extracted.'''
    # find z sensitivity in nm/V
    index = header_info.find('@Sens. Zsens: V')
    sensitivity = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # find the size of the probed area and the x and y position of the curve
    index = header_info.find('\Scan Size: ')
    scan_size = float(re.findall("\d+", header_info[index:])[0])
    # indentation x position (nm)
    index = header_info.find('\X Offset: ')
    x_pos = float(re.findall("\d+", header_info[index:])[0])
    # indentation y position (nm)
    index = header_info.find('\Y Offset: ')
    y_pos = float(re.findall("\d+", header_info[index:])[0])
    # indentation rate in Hz
    index = header_info.find('\Scan rate: ')
    ind_rate = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # find the forward velocity in nm/s
    index = header_info.find('\Forward vel.: ')
    forw_vel = float(re.findall("\d+\.\d+", header_info[index:])[0])*100
    # Ramp size (V), the interesting value is the second after @4:Ramp size..
    index = header_info.find('@4:Ramp size Zsweep: V')
    ramp_size_in_Volt = float(re.findall("\d+\.\d+", header_info[index:])[1])
    ramp_size_in_nm = sensitivity*ramp_size_in_Volt
    # n° of points in the retraction curve
    index = header_info.find('Samps/line: ', index)
    Rt_npoints = int(re.findall("\d+", header_info[index:])[0])
    # n° of points in the approach curve
    Ex_npoints = int(re.findall("\d+", header_info[index:])[1])

    '''The binary data only contains the info about the cantilever deflection, the height (ramp) values need to be 
    computed from the data of the header.'''
    # step increment along the retraction ramp
    Rt_step = ramp_size_in_nm/Rt_npoints
    # step increment along the approach ramp
    Ex_step = ramp_size_in_nm/Ex_npoints
    # Approach ramp array
    Ex_ramp = np.zeros(Ex_npoints)
    for i in range(1, Ex_npoints, 1):
        Ex_ramp[i] = Ex_ramp[i-1] + Ex_step
    # Retraction ramp array (should be in the reverse order)
    Rt_ramp = np.full(Rt_npoints, ramp_size_in_nm)
    for i in range(1, Rt_npoints, 1):
        Rt_ramp[i] = Rt_ramp[i - 1] - Rt_step

    # Approach deflections array
    Ex_deflection = all_deflections[0:Ex_npoints]
    index = header_info.find('@4:Z scale:')
    from_LSB_to_Volt = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # find the spring constant nN/nm
    index = header_info.find('Spring Constant:')
    spring_constant = float(re.findall("\d+\.\d+", header_info[index:])[0])
    #find the Deflection sensitivity nm/V
    index = header_info.find('@Sens. DeflSens: V')
    defl_sensitivity = float(re.findall("\d+\.\d+", header_info[index:])[0])
    # Approach deflections in nm
    Ex_deflection = Ex_deflection*from_LSB_to_Volt*defl_sensitivity
    # from the raw file the array is reversed wrt the ramp array so needs to be flipped
    Ex_deflection = np.flip(Ex_deflection)
    # Approach force in pN
    Ex_force = Ex_deflection*spring_constant*1000

    # Retraction deflections array
    Rt_deflection = all_deflections[Rt_npoints:2*Rt_npoints]
    # Retraction deflections in nm
    Rt_deflection = Rt_deflection*from_LSB_to_Volt*defl_sensitivity
    # from the raw file the array is reversed wrt the ramp array so needs to be flipped
    Rt_deflection = np.flip(Rt_deflection)
    # Retraction force in pN
    Rt_force = Rt_deflection*spring_constant*1000

    # Savitzky Golay filter to reduce noise
    Ex_deflection_filtered = savgol_filter(Ex_deflection, 307, 3)
    Rt_deflection_filtered = savgol_filter(Rt_deflection, 307, 3)
    # Forces filtered
    Ex_force_filtered = Ex_deflection_filtered*spring_constant*1000
    Rt_force_filtered = Rt_deflection_filtered*spring_constant*1000

    # Estimating the baseline in order to shift to zero the initial force values
    Ex_force_shifted = Ex_force - np.mean(Ex_force[:1000])
    Rt_force_shifted = Rt_force - np.mean(Rt_force[:1000])
    Ex_force_filtered = Ex_force_filtered - np.mean(Ex_force_filtered[:1000])
    Rt_force_filtered = Rt_force_filtered - np.mean(Rt_force_filtered[:1000])

    # Separation arrays are built by subtracting deflection form the height
    Ex_separation = np.array(Ex_ramp-Ex_deflection)
    Rt_separation = np.array(Rt_ramp+Rt_deflection)
    Ex_separation_filtered = np.array(Ex_ramp-Ex_deflection_filtered)
    Rt_separation_filtered = np.array(Rt_ramp+Rt_deflection_filtered)

    # Setting the "surface" as zero of the curve
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

    # Plotting the curve
    # fig, ax = plt.subplots()
    # ax.plot(aligned_Ex_sep, Ex_force_shifted, 'b', zorder=1)
    # ax.plot(aligned_Rt_sep, Rt_force_shifted, 'r', zorder=2)
    # ax.grid()
    # ax.set_xlabel("Separation (nm)")
    # ax.set_ylabel("Force (pN)")
    # curve_name = str(file_path).split("/")[-1]
    # plt.savefig(r"Example_figures/" + "Nanoscope_converter_" + curve_name + ".png", dpi=300)
    # plt.show()

    return curve

# Testing the two functions with two random force curves
# nanoscope_converter(r"Example_RawData/cr00028.000")
# nanoscope_converter(r"Example_RawData/cr00082.000")
