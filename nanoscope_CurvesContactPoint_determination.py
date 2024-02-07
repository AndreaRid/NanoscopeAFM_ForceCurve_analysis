# import matplotlib.pyplot as plt
# from Nanoscope_converter import nanoscope_converter
# import glob
# import os
import numpy as np
#
'''Lines before function definition are just for local diagnostic tests. The script contains 2 functions for 
identification of the contact point in the force curves (more precisely in the indenting force curve), the use of one 
function over the other depends on the shape of the curve, in the case of roughly a monotonic indenting curve, 
the second function works better (for each of them the values related to the signal noise have to be adjusted 
according to specific cases).'''
# folders = [f for f in glob.glob("Example_RawData/Chromosomes" + '**/*', recursive=True)]
# folder = "Example_RawData/Chromosome_PS161143"
# storing the directories of all the files
# files = []
# for folder in folders:
#     file = [f for f in glob.glob(folder + '**/*', recursive=True) if '.000' in f]
#     files.extend(file)
# title = False   # control variable used for writing the header in the text file only once
# # HOW MANY POINTS TO USE FOR CORRECTING THE SENSITIVITY?
# # points = input('How many points to use for correcting the sensitivity? Type "no" for avoid sensitivity correction ')
# points = 'no'
# for path in files:
#     raw_separation = nanoscope_converter(path)[0]  # script per lettura e conversione dei file Bruker
#     force = nanoscope_converter(path)[1]       # script per lettura e conversione dei file Bruker
#     curve_name = os.path.basename(path)        # obtaining the name of the file
#
#     # #-------CORRECTION FOR SENSITIVITY--------
#
#     def fit_func_linear(x, a, b):    # sensitivity is corrected by applying a linear fit to the last part of the curve
#        return a*x + b
#
#     if points == 'no':
#         separation = raw_separation
#     else:
#         points = int(points)
#         corrected_separation = raw_separation[(len(raw_separation)-points):]
#         corrected_force = force[(len(force)-points):]
#         fit_param = curve_fit(fit_func_linear, corrected_separation, corrected_force,  maxfev=100000)
#         correction = (force - fit_param[0][1])/fit_param[0][0]
#         separation = raw_separation - correction
def contact_pointFinder1(separation, force):
        '''Identify the contact point as the point at which the force > 2*baseline_noise. The function calculates the
        average value of the baseline noise (as its st. dev) and find the first point where the force becomes greater
        than this value, this point will be the contact point.'''
        # percentage of the curve that should be regarded as baseline
        pc = 25
        bs_points = int(len(separation)*pc/100)
        # calculate the baseline value and the baseline noise as the standard deviation of the baseline
        bs_value = np.mean(force[:bs_points])
        bs_noise = np.std(force[:bs_points])
        # finding all the points in contact with the baseline
        contact_array = [p for p in force if p <= bs_value + 2*bs_noise
                         and p >= bs_value - 2*bs_noise ]
        # The contact point will be the last point "in contact" with the baseline +- noise
        cp_index = len(contact_array)
        cp_x = separation[len(contact_array)]
        cp_y = force[len(contact_array)]
        # Plotting
        # fig = plt.figure()
        # ax = fig.subplots()
        # ax.grid()
        # ax.scatter(separation, force, s=2, c='b')
        # ax.scatter(cp_x, cp_y, s=25, c='r', marker='X')
        # plt.show()

        return cp_x, cp_y, cp_index


def contact_pointFinder2(separation, force):
        '''This function estimates the contact point as the first point where the force becomes > 50 pN, which is a
        reasonable estimation for the noise of the baseline. In the case of snap-in events where the force decreases
        towards zero but then increases again, this function might generate wrong results. The threshold for the
        noise value has to be adjusted depending on the instrumentation and type of measurement.'''
        i = 0
        while force[i] <= 550:
                i += 1
        while force[i] > 50:
                i -= 1
        cp_index = i
        cp_x = separation[cp_index]
        cp_y = force[cp_index]
        return cp_x, cp_y, cp_index
