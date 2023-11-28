# import matplotlib.pyplot as plt
import numpy as np
# from Nanoscope_converter import nanoscope_converter
# import glob
# import os
#
# COMMENTED LINES ARE JUST FOR RUNNING THE FUNCTION INDEPENDENTLY AND CHECKING IT
# # folder_path = input('inserire directory della cartella contenente le curve da analizzare: ')
# folder_path = r"C:\Users\Andrea\Desktop\VU_postdoc\Lab\20220224_Chromosomes_IndentationTests_PABuffer\PS241224\curves"
# # create a list named files that contains all the path of all the files
# files = [f for f in glob.glob(folder_path + '**/*', recursive=True)]
#
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
        i = 0
        while force[i] <= 550:
                i += 1
        while force[i] > 50:
                i -= 1
        cp_index = i
        cp_x = separation[cp_index]
        cp_y = force[cp_index]
        return cp_x, cp_y, cp_index
