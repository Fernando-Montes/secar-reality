import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from emcee.backends import HDFBackend
import multiprocessing as mp
from functools import partial
import corner

from config import (
    viewers, element_names_stripped, results_directory,                      # Defines system geometry
    fox_file, matrix_directory, pkl_directory
)
from helper_functions import runECAT

groupsInfo = {
    '1': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '2': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '3': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-21, -7),  "label": "FP1 Slits: -21 to -7 mm"},
        },
    '4': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (7, 21),    "label": "FP1 Slits: 7 to 21 mm"},
        },
    '5': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '6': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-50, 50),  "label": "FP1 Slits: ±50 mm"},
        },
    '7': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": [-7, 7], "label": "FP1 Slits: ±7 mm"},
        },
    '8': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-21, -7),  "label": "FP1 Slits: -21 to -7 mm"},
        },
    '9': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (7, 21),    "label": "FP1 Slits: 7 to 21 mm"},
        },
    '10': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.1284349631 / 1000, "Y": 0.1781644397 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '11': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.1284349631 / 1000, "Y": 0.1781644397 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '12': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -8.068629955 / 1000, "Y": 0.030415035 / 1000, "R": 0.5 / 1000, "label": "1 MM (+15 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '13': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -4.239219775 / 1000, "Y": 0.431691050 / 1000, "R": 0.5 / 1000, "label": "1 MM (+7.5 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '14': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 8.061907576 / 1000, "Y": -0.310614354 / 1000, "R": 0.5 / 1000, "label": "1 MM (-15 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '15': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 4.029862244 / 1000, "Y": -0.504692800 / 1000, "R": 0.5 / 1000, "label": "1 MM (-7.2 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '16': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -10.62294395 / 1000, "Y": -0.18692655 / 1000, "R": 0.5 / 1000, "label": "1 MM (+20 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '17': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 10.30665605 / 1000, "Y": -0.18692655 / 1000, "R": 0.5 / 1000, "label": "1 MM (-20 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    # '18': {
    #     'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
    #     'pp_alignment' : {"X": 0.311246954 / 1000, "Y": 8.5930652  / 1000, "R": 0.5 / 1000, "label": "3 hole (+Y-position grp 18) -2% dE/E"},
    #     'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
    #     },
    '18': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.253681624/1000, "Y": -4.113076555/1000, "R": 0.5/1000, "label": "3 hole (-Y-position grp 18) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    # '18': {
    #     'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
    #     'pp_alignment' : {"X": [0.311246954/1000, -6.355652212/1000, -0.253681624/1000], "Y": [8.5930652/1000, 2.7191336/1000, -4.113076555/1000], "R": [0.5/1000, 0.5/1000, 0.5/1000], "label": "3 hole Grp 18) -2% dE/E"},
    #     'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
    #     },
    '19': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 0.071253860 / 1000, "Y": 7.806471915  / 1000, "R": 0.5 / 1000, "label": "3 hole (+Y-position grp 19) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    }

# Dictionary that sets the max. variance a given type of parameter can possibly have (to be used as bounds in the
# parameter optimization
# bounds = {
#     'mu_y' : [-0.006, 0.006],    # +- 6mrad     Initial distribution
#     'mu'   : [1.9/100, 2.1/100], # 2+-0.1%    mean dE/E Initial distribution
#     'sigma': [5/100, 11/100],    # 8+-3%      sigma dE/E Initial distribution
#     'Roll' : [-0.003, 0.003],    # +-1.1deg rotation around beam axis
#     'X'    : [-0.001, 0.001],    # +-1mm
#     'Y'    : [-0.001, 0.001],    # +-1mm
#     'B_SC' : [-0.03, 0.03],      # +-3.0% field
#     'dist' : [-0.003, 0.003],    # +-3mm
#     'dB'   : [-0.011, 0.011],    # Possible miscalibration +-1.1% according to 2024 tuning
# }

bounds = {
    'target_rho': [-0.5, 0.5],
    'ax_sigma' : [0, 0.008], # +- 3mrad     Initial distribution
    'ay_mu' : [-0.006, 0.006],    # +- 3mrad     Initial distribution
    'ay_sigma' : [0, 0.008], # +- 3mrad     Initial distribution
    'angle_rho': [-0.5, 0.5],
    'dE_mu'   : [-1/100, 1/100], # 2+-0.1%    mean dE/E Initial distribution
    'dE_sigma': [0/100, 2/100],    # 8+-3%      sigma dE/E Initial distribution
    'Roll' : [-0.003, 0.003],    # +-1.1deg rotation around beam axis
    'X'    : [-0.001, 0.001],    # +-1mm
    'Y'    : [-0.001, 0.001],    # +-1mm
    'B_SC' : [-0.03, 0.03],      # +-3.0% field
    'dist' : [-0.003, 0.003],    # +-3mm
    'dB'   : [-0.011, 0.011],    # Possible miscalibration +-1.1% according to 2024 tuning
}

# Trouble shooting... can be deleted
def load_2d_image(gorup_number, viewer, image_path, plot = True, save_path = None):

    data = np.load(image_path)
    corrected_full_image = data["corrected_full_image"]
    xf_range = data["xf_range"]
    yf_range = data["yf_range"]

    if plot:
        plt.imshow(
            corrected_full_image,
            extent = (xf_range[0], xf_range[-1], yf_range[-1], yf_range[0]),
            cmap = 'viridis', aspect = 'auto'
        )
        plt.colorbar()

        plt.xlim(xf_range[0], xf_range[-1])
        plt.ylim(yf_range[-1], yf_range[0])
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')

        # Title depending on POV
        if "_cosy" in image_path:
            pov = "Looking at Beam / COSY POV"
        else:
            pov = "Looking with Beam / Viola POV"

        plt.title(f"Group {gorup_number} - {viewer}\n({pov})", fontweight = 'bold')

        if save_path:
            plt.savefig(save_path, dpi = 300)
            plt.close()
        else:
            plt.show()
            plt.close()

    return corrected_full_image, xf_range, yf_range

####################################
##### UPDATED FUNCTIONS FOR E3 #####
####################################
# Loading projections from csv files AND/OR returns metric value when comparing viewer and ECAT simulations
# viewerImages = {"VDXXXX": {"x_projection": '----.csv', "y_projection": '----.csv', "2d_image": '----.npz',
#                           "fp1_slits": {"slits_mm": (SLL, SLR),    "label": "FP1 Slits: SLL to SLR mm"}}, ...}
def load_projection(viewerImages, all_x = None, all_y = None, transmission_indices = None, plot = True, metric = None, save_fig = False, all_ax = None, all_dE = None):
    '''
    metric : metric to compare viewer images against ECAT results. Can be: 'chisq_1d (X and Y projections of a viewer), chisq_2d (viewer)'
    returns metric value
    '''
    chisq_x, chisq_y = 0, 0
    metric_value = 0
    for viewer in viewerImages:   # go viewer by viewer
        if viewer.startswith('VD'):

            dataPath = viewerImages['data_path']
            # print(dataPath)
            # print(viewerImages[viewer])
            # print(viewerImages[viewer]['x_projection'])
            # print(os.path.join(dataPath, viewerImages[viewer]['x_projection']))

            if metric == 'chisq_1d' or plot == True:
                x_projection_path = os.path.join(dataPath, viewerImages[viewer]['x_projection'])
                y_projection_path = os.path.join(dataPath, viewerImages[viewer]['y_projection'])
                # Load 1d viewer projections and skip any lines starting with a '#'
                df_x = pd.read_csv(x_projection_path, comment = "#")
                df_y = pd.read_csv(y_projection_path, comment = "#")

                # Extract the 1d viewer projections data
                x_pos = -df_x.iloc[:, 0].to_numpy()
                x_intensity = df_x.iloc[:, 1].to_numpy()
                y_pos = df_y.iloc[:, 0].to_numpy()
                y_intensity = df_y.iloc[:, 1].to_numpy()

                x_mean = np.average(x_pos, weights=x_intensity)
                x_sigma = np.sqrt( np.average((x_pos-x_mean)**2, weights=x_intensity) )
                y_mean = np.average(y_pos, weights=y_intensity)
                y_sigma = np.sqrt( np.average((y_pos-y_mean)**2, weights=y_intensity) )
                # print(f"VIEWER - {viewer}, X mean [mm]: {x_mean:.2f},  X sigma [mm]: {x_sigma:.2f}")
                # print(f"VIEWER - {viewer}, Y mean [mm]: {y_mean:.2f},  Y sigma [mm]: {y_sigma:.2f}")

                # Creating 1d viewer histograms (binned)
                x_bins = 50
                x_scale = x_bins/len(x_pos)
                x_intensityC = [i*x_scale for i in x_intensity]
                x_intensityC_sum = np.sum(x_intensityC)
                x_intensityC = [max(0, round(100*x_bins*i/x_intensityC_sum)) for i in x_intensityC]   # Scale binned viewer so sum is 100*x_bins
                x_intensityC[0] = max(1, x_intensityC[0] )                                            # Making sure the extrems are included
                x_intensityC[-1] = max(1, x_intensityC[-1] )                                          # Making sure the extrems are included
                x_intensity  = [100*x_bins*i/x_intensityC_sum for i in x_intensity]                   # Scale orig viewer so agrees with binned
                x_vd = [ p for p,n in zip(x_pos, x_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

                y_bins = 50
                y_scale = y_bins/len(y_pos)
                y_intensityC = [i*y_scale for i in y_intensity]
                y_intensityC_sum = np.sum(y_intensityC)
                y_intensityC = [max(0, round(100*y_bins*i/y_intensityC_sum)) for i in y_intensityC]   # Scale binned viewer so sum is 100*y_bins
                y_intensityC[0] = max(1, y_intensityC[0] )                                            # Making sure the extrems are included
                y_intensityC[-1] = max(1, y_intensityC[-1] )                                          # Making sure the extrems are included
                y_intensity  = [100*y_bins*i/y_intensityC_sum for i in y_intensity]                   # Scale orig viewer so agrees with binned
                y_vd = [ p for p,n in zip(y_pos, y_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

                # Creating ecat histograms if they are used
                if all_x is not None:
                    # Find position index using viewer
                    posV = viewers[viewer.replace("VD", "") ]["index"]
                    posS = element_names_stripped.index('FP1 Slits')  # position slits
                    if posS < posV:   # Viewer is after fp1 slits
                        x = []
                        y = []
                        aX = []
                        dE = []
                        valid_indices = []
                        for i in range(len(all_x)):
                            if transmission_indices[i] >= posV \
                              and viewerImages['fp1_slits']['slits_mm'][0] <= -all_x[i][posS]*1000 \
                              and -all_x[i][posS]*1000 <= viewerImages['fp1_slits']['slits_mm'][1]:
                                valid_indices.append(i)
                                x.append( -all_x[i][posV]*1000 )
                                aX.append( -all_ax[i][0]*1000 )   # Append the angles at target that arrive at this location!
                                y.append( all_y[i][posV]*1000 )
                                dE.append( all_dE[i][posV]*100 )
                        x = np.array(x)
                        aX = np.array(aX)
                        y = np.array(y)
                        dE = np.array(dE)
                    else:
                        x  = np.array([-xx[posV]*1000 for i, xx in enumerate(all_x)  if transmission_indices[i] >= posV]) # in mm
                        aX = np.array([ -ax[0]*1000    for i, ax in enumerate(all_ax) if transmission_indices[i] >= posV]) # in mm
                        y  = np.array([yy[posV]*1000 for i, yy in enumerate(all_y)  if transmission_indices[i] >= posV]) # in mm
                        dE  = np.array([de[posV]*100 for i, de in enumerate(all_dE) if transmission_indices[i] >= posV]) # in mm
                    # print(f"ECAT - {viewer}, X mean [mm]: {np.mean(x):.2f},  X std [mm]: {np.std(x):.2f}")
                    # print(f"ECAT - Arriving at {viewer}, aX at target mean [mm]: {np.mean(aX):.2f},  aX std [mm]: {np.std(aX):.2f}")
                    # print(f"ECAT - {viewer}, Y mean [mm]: {np.mean(y):.2f},  Y std [mm]: {np.std(y):.2f}")
                    # print(f"ECAT - {viewer}, dE mean [mm]: {np.mean(dE):.2f},  dE std [mm]: {np.std(dE):.2f}")

                    # Scale ecat data to viewer counts
                    x_edges = np.histogram_bin_edges(x_vd, bins=x_bins)
                    vd_counts, _ = np.histogram(x_vd, bins=x_edges)
                    ecat_counts, _ = np.histogram(x, bins=x_edges)
                    widths = np.diff(x_edges)
                    area_vd   = np.sum(vd_counts * widths)
                    area_ecat = np.sum(ecat_counts * widths)
                    x_scale = area_vd / area_ecat if area_ecat > 0 else 0.0   # weights of ECAT x-histogram
                    # Calculating chi-square
                    viewer_counts, edges = np.histogram(x_vd, bins=x_bins)
                    ecat_counts, edges = np.histogram(x, bins=x_edges, weights=np.full(len(x), x_scale))

                    if np.abs(np.mean(x) - np.mean(x_vd)) > np.max(x_pos): # if ECAT mean minus viewer mean is more than size of the viewer
                        chisq_x = np.inf
                    else:
                        chisq_x = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)

                if all_y is not None and all_x is not None:
                    # Find position index using viewer
                    if element_names_stripped.index('FP1 Slits') < posV:   # Viewer is after fp1 slits
                        y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if i in valid_indices]) # in mm
                    else:
                        y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if transmission_indices[i] >= posV]) # in mm
                    # Scale ecat data to viewer counts
                    y_edges = np.histogram_bin_edges(y_vd, bins=y_bins)
                    vd_counts, _ = np.histogram(y_vd, bins=y_edges)
                    ecat_counts, _ = np.histogram(y, bins=y_edges)
                    widths = np.diff(y_edges)
                    area_vd   = np.sum(vd_counts * widths)
                    area_ecat = np.sum(ecat_counts * widths)
                    y_scale = area_vd / area_ecat if area_ecat > 0 else 0.0  # weights of ECAT y-histogram
                    # Calculating chi-square
                    viewer_counts, edges = np.histogram(y_vd, bins=y_bins)
                    ecat_counts, edges = np.histogram(y, bins=y_edges, weights=np.full(len(y), y_scale))
                    chisq_y = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)

            if metric == 'chisq_2d' or plot == True:
                image_path = os.path.join(dataPath, viewerImages[viewer]["2d_image"])
                # Load 2d viewer data
                data = np.load(image_path)
                corrected_full_image = data["corrected_full_image"]
                corrected_full_image = corrected_full_image[:, ::-1]
                x2d_range = data["xf_range"]
                y2d_range = data["yf_range"]

        	# Optionally plot the projection
            if plot:
                fig, axs = plt.subplots(2, 2, figsize=(17,9))
                labelSlits = viewerImages['fp1_slits']['label']
                plt.suptitle(f"{viewer} - {labelSlits}\n(Viewing with the beam)", fontsize=16)

                axs[0,0].plot(x_pos, x_intensity, label = 'Viewer original')
                axs[0,0].hist(x_vd, bins=x_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
                if all_x is not None:
                    axs[0,0].hist(x, bins=x_edges, histtype='step', linewidth=4, weights=np.full(len(x), x_scale),  # uniformly scale ecat
                        label=f'ecat × {x_scale:.3g}', color = 'orange')
                axs[0,0].set_xlabel("X [mm]")
                axs[0,0].set_ylabel("Intensity")
                axs[0,0].set_title("X projection", fontweight = 'bold')
                axs[0,0].set_xlim(min(x_pos), max(x_pos))
                axs[0,0].legend(loc='best')   # place legend
                axs[0,0].text( 0.01, 0.98, f"chi-sq = {chisq_x:.1f}", transform=axs[0,0].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')

                axs[0,1].plot(y_pos, y_intensity, label = 'Viewer original')
                axs[0,1].hist(y_vd, bins=y_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
                if all_y is not None:
                    axs[0,1].hist(y, bins=y_edges, histtype='step', linewidth=4, weights=np.full(len(y), y_scale),  # uniformly scale ecat
                        label=f'ecat × {y_scale:.3g}', color = 'orange')
                axs[0,1].set_xlabel("Y [mm]")
                axs[0,1].set_ylabel("Intensity")
                axs[0,1].set_title("Y projection", fontweight = 'bold')
                axs[0,1].set_xlim(min(y_pos), max(y_pos))
                axs[0,1].legend(loc='best')   # place legend
                axs[0,1].text( 0.01, 0.98, f"chi-sq = {chisq_y:.1f}", transform=axs[0,1].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')

                im = axs[1,0].imshow( corrected_full_image, extent = (-x2d_range[-1], -x2d_range[0], y2d_range[-1], y2d_range[0]), \
                    cmap = 'viridis', aspect = 'auto')
                cbar = fig.colorbar(im, ax=axs[1,0])
                axs[1,0].set_xlim(-x2d_range[-1], -x2d_range[0])
                axs[1,0].set_ylim(y2d_range[0], y2d_range[-1])
                axs[1,0].set_xlabel('X (mm)')
                axs[1,0].set_ylabel('Y (mm)')
                axs[1,0].set_title("Viewer", fontweight = 'bold')

                if all_x is not None and all_y is not None:
                    axs[1,1].hist2d(x, y, bins=(50, 50), cmap='viridis',range=[[-x2d_range[-1], -x2d_range[0]],[y2d_range[0], y2d_range[-1]]])
                    axs[1,1].set_xlabel('X (mm)')
                    axs[1,1].set_ylabel('Y (mm)')
                    axs[1,1].set_title("ECAT", fontweight = 'bold')
                plt.tight_layout()
                if save_fig:
                    save_path = f"{results_directory}/imageCompared_VD{viewer}"
                    plt.savefig(save_path, dpi = 300)
                else:
                    plt.show()
            metric_value = metric_value + chisq_x + chisq_y
    return metric_value

# Loading projections from csv files and obtain information from viewer data ONLY
def load_projection_viewer(images2compare, plot = True):

    for groupImages in images2compare:   # go group by group
        group_number = groupImages[0]
        imagesByGroup = groupImages[1]
        group_path = f"PartD_ViewerAnalysis/Group{group_number}"
        for viewer in imagesByGroup:   # go viewer by viewer
            x_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_xf_cosy.csv")
            y_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_yf_cosy.csv")
            # Load 1d viewer projections and skip any lines starting with a '#'
            df_x = pd.read_csv(x_projection_path, comment = "#")
            df_y = pd.read_csv(y_projection_path, comment = "#")
            # Extract the 1d viewer projections data
            x_pos = -df_x.iloc[:, 0].to_numpy()
            x_intensity = df_x.iloc[:, 1].to_numpy()
            y_pos = df_y.iloc[:, 0].to_numpy()
            y_intensity = df_y.iloc[:, 1].to_numpy()

            x_mean = np.sum([x_pos[i]*x_intensity[i] for i in range(len(x_pos))])/np.sum(x_intensity)
            y_mean = np.sum([y_pos[i]*y_intensity[i] for i in range(len(y_pos))])/np.sum(y_intensity)
            print(f"Group {group_number} - {viewer}, X mean [mm]: {x_mean:.2f},  Y mean [mm]: {y_mean:.2f}")

            # Creating 1d viewer histograms (binned)
            x_bins = 50
            x_scale = x_bins/len(x_pos)
            x_intensityC = [i*x_scale for i in x_intensity]
            x_intensityC_sum = np.sum(x_intensityC)
            x_intensityC = [max(0, round(100*x_bins*i/x_intensityC_sum)) for i in x_intensityC]   # Scale binned viewer so sum is 100*x_bins
            x_intensityC[0] = max(1, x_intensityC[0] )                                            # Making sure the extrems are included
            x_intensityC[-1] = max(1, x_intensityC[-1] )                                          # Making sure the extrems are included
            x_intensity  = [100*x_bins*i/x_intensityC_sum for i in x_intensity]                   # Scale orig viewer so agrees with binned
            x_vd = [ p for p,n in zip(x_pos, x_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

            y_bins = 50
            y_scale = y_bins/len(y_pos)
            y_intensityC = [i*y_scale for i in y_intensity]
            y_intensityC_sum = np.sum(y_intensityC)
            y_intensityC = [max(0, round(100*y_bins*i/y_intensityC_sum)) for i in y_intensityC]   # Scale binned viewer so sum is 100*y_bins
            y_intensityC[0] = max(1, y_intensityC[0] )                                            # Making sure the extrems are included
            y_intensityC[-1] = max(1, y_intensityC[-1] )                                          # Making sure the extrems are included
            y_intensity  = [100*y_bins*i/y_intensityC_sum for i in y_intensity]                   # Scale orig viewer so agrees with binned
            y_vd = [ p for p,n in zip(y_pos, y_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

            if plot == True:
                image_path = os.path.join(group_path, f"Group{group_number}_VD{viewer}_2D_cosy.npz")
                # Load 2d viewer data
                data = np.load(image_path)
                corrected_full_image = data["corrected_full_image"]
                corrected_full_image = corrected_full_image[:, ::-1]
                x2d_range = data["xf_range"]
                y2d_range = data["yf_range"]
                fig, axs = plt.subplots(2, 2, figsize=(17,9))
                labelPP    = groupsInfo[group_number]['pp_alignment']['label']
                labelSlits = groupsInfo[group_number]['fp1_slits']['label']
                plt.suptitle(f"Group {group_number} - {labelPP} - {labelSlits} - {viewer}\n(Viewing with the beam)", fontsize=16)

                axs[0,0].plot(x_pos, x_intensity, label = 'Viewer original')
                axs[0,0].hist(x_vd, bins=x_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
                axs[0,0].set_xlabel("X [mm]")
                axs[0,0].set_ylabel("Intensity")
                axs[0,0].set_title("X projection", fontweight = 'bold')
                axs[0,0].set_xlim(min(x_pos), max(x_pos))
                axs[0,0].legend(loc='best')   # place legend

                axs[0,1].plot(y_pos, y_intensity, label = 'Viewer original')
                axs[0,1].hist(y_vd, bins=y_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
                axs[0,1].set_xlabel("Y [mm]")
                axs[0,1].set_ylabel("Intensity")
                axs[0,1].set_title("Y projection", fontweight = 'bold')
                axs[0,1].set_xlim(min(y_pos), max(y_pos))
                axs[0,1].legend(loc='best')   # place legend

                im = axs[1,0].imshow( corrected_full_image, extent = (-x2d_range[-1], -x2d_range[0], y2d_range[-1], y2d_range[0]), \
                    cmap = 'viridis', aspect = 'auto')
                cbar = fig.colorbar(im, ax=axs[1,0])
                axs[1,0].set_xlim(-x2d_range[-1], -x2d_range[0])
                axs[1,0].set_ylim(y2d_range[0], y2d_range[-1])
                axs[1,0].set_xlabel('X (mm)')
                axs[1,0].set_ylabel('Y (mm)')
                axs[1,0].set_title("Viewer", fontweight = 'bold')

                plt.tight_layout()
                save_path = f"{results_directory}/imageCompared_Group{group_number}_VD{viewer}"
                plt.savefig(save_path, dpi = 300)
                plt.close()

# Loading projections from ECAT simulations
def load_projection_ecat(imagesByGroup, group_number, all_x = None, all_y = None, transmission_indices = None, plot = True, save_fig = False, all_ax = None, all_dE = None):

    line = ""
    for viewer in imagesByGroup:   # go viewer by viewer

        if all_x is not None:
            # Find position index of the viewer
            posV = viewers[viewer]["index"]
            posS = element_names_stripped.index('FP1 Slits')  # position slits
            if posS < posV:   # Viewer is after fp1 slits
                x = []
                y = []
                aX = []
                dE = []
                valid_indices = []
                for i in range(len(all_x)):
                    if transmission_indices[i] >= posV \
                      and groupsInfo[group_number]['fp1_slits']["slits_mm"][0] <= -all_x[i][posS]*1000 \
                      and -all_x[i][posS]*1000 <= groupsInfo[group_number]['fp1_slits']["slits_mm"][1]:
                        valid_indices.append(i)
                        x.append( -all_x[i][posV]*1000 )
                        aX.append( -all_ax[i][0]*1000 )   # Append the angles at target that arrive at this location!
                        y.append( all_y[i][posV]*1000 )
                        dE.append( all_dE[i][posV]*100 )
                x = np.array(x)
                aX = np.array(aX)
                y = np.array(y)
                dE = np.array(dE)
            else:
                x  = np.array([-xx[posV]*1000 for i, xx in enumerate(all_x)  if transmission_indices[i] >= posV]) # in mm
                aX = np.array([ -ax[0]*1000    for i, ax in enumerate(all_ax) if transmission_indices[i] >= posV]) # in mm
                y  = np.array([yy[posV]*1000 for i, yy in enumerate(all_y)  if transmission_indices[i] >= posV]) # in mm
                dE  = np.array([de[posV]*100 for i, de in enumerate(all_dE) if transmission_indices[i] >= posV]) # in mm
            #print(f"ECAT Group {group_number} - {viewer}, X mean [mm]: {np.mean(x):.2f},  X std [mm]: {np.std(x):.2f}")

        line = line + f"{np.mean(x):.2f} "

        if all_y is not None and all_x is not None:
            # Find position index using viewer
            if element_names_stripped.index('FP1 Slits') < posV:   # Viewer is after fp1 slits
                y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if i in valid_indices]) # in mm
            else:
                y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if transmission_indices[i] >= posV]) # in mm

    	# Optionally plot the projection
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(17,9))
            labelPP    = groupsInfo[group_number]['pp_alignment']['label']
            labelSlits = groupsInfo[group_number]['fp1_slits']['label']
            plt.suptitle(f"Group {group_number} - {labelPP} - {labelSlits} - {viewer}\n(Viewing with the beam)", fontsize=16)

            if all_x is not None:
                axs[0,0].hist(x, bins=50, histtype='step', linewidth=4, label='ecat', color = 'orange')
            axs[0,0].set_xlabel("X [mm]")
            axs[0,0].set_ylabel("Intensity")
            axs[0,0].set_title("X projection", fontweight = 'bold')
            axs[0,0].legend(loc='best')   # place legend

            if all_y is not None:
                axs[0,1].hist(y, bins=50, histtype='step', linewidth=4, label='ecat', color = 'orange')
            axs[0,1].set_xlabel("Y [mm]")
            axs[0,1].set_ylabel("Intensity")
            axs[0,1].set_title("Y projection", fontweight = 'bold')
            axs[0,1].legend(loc='best')   # place legend

            if all_x is not None and all_y is not None:
                axs[1,1].hist2d(x, y, bins=(50, 50), cmap='viridis')
                axs[1,1].set_xlabel('X (mm)')
                axs[1,1].set_ylabel('Y (mm)')
                axs[1,1].set_title("ECAT", fontweight = 'bold')
            plt.tight_layout()
            if save_fig:
                save_path = f"{results_directory}/imageCompared_Group{group_number}_VD{viewer}"
                plt.savefig(save_path, dpi = 300)
            else:
                plt.show()

    return line

# Loading projections from ECAT simulations without considering data from E2
def load_projection_ecat_raw(locations, all_x = None, all_y = None, transmission_indices = None, all_ax = None, all_dE = None):

    line = ""
    for location in locations:   # go location by location

        if all_x is not None:
            # Find position index of the viewer
            posV = element_names_stripped.index(location)

            x  = np.array([-xx[posV]*1000 for i, xx in enumerate(all_x)  if transmission_indices[i] >= posV]) # in mm
            aX = np.array([ -ax[0]*1000    for i, ax in enumerate(all_ax) if transmission_indices[i] >= posV]) # in mrad
            y  = np.array([yy[posV]*1000 for i, yy in enumerate(all_y)  if transmission_indices[i] >= posV]) # in mm
            dE  = np.array([de[posV]*100 for i, de in enumerate(all_dE) if transmission_indices[i] >= posV]) # in %
            #print(f"ECAT Group {group_number} - {viewer}, X mean [mm]: {np.mean(x):.2f},  X std [mm]: {np.std(x):.2f}")

        line = line + f" {np.mean(x):.2f} " +  f"{np.std(x):.2f}"

    return line


# Function that counts the number of SECAR parameters to be optimized and creates an array of bounds
def countSECARparams(params):
    count = 0
    pbounds = []
    for p in params:
        if p['par'] == 'XY':
            pbounds.append(bounds['X'])
            pbounds.append(bounds['Y'])
            count += 2
        else:
            pbounds.append(bounds[p['par']])
            count += 1
    return count, pbounds

# Creates labels for corner plot
def make_param_labels(params):
    labels = []
    for p in params:
        if p['par'] == "XY":
            labels.append(f"{p['elem']}_X")
            labels.append(f"{p['elem']}_Y")
        else:
            labels.append(f"{p['elem']}_{p['par']}")
    return labels

# Function to convert emcee input to something runECAT can use
def convert_theta_to_values(params, theta):
    """Map a flat theta vector onto params structure, expanding XY -> [x,y]."""
    values, j = [], 0
    for p in params:
        if p["par"] == "XY":
            values.append([theta[j], theta[j+1]])
            j += 2
        else:
            values.append(theta[j])
            j += 1
    return values

# Function to create arrays tuneChanges and initialDistChanges (parameters in runECAT)
def prepareMCChanges(params, values):
    tuneChanges = []
    initialDistChanges = []
    flagCosy = False
    for i, p in enumerate(params):
        if p['elem'] != 'initialDist':
            tuneChanges.append( [ p['elem'], p['par'], values[i] ] )
            if p['par'] != 'dB':
                flagCosy = True
        else:
            initialDistChanges.append( [ p['par'], values[i] ] )
    return tuneChanges, initialDistChanges, flagCosy

# Prior to penalize large changes
def log_prior(theta, pbounds):
    i = 0
    for i, t in enumerate(theta):
        if t < pbounds[i][0] or t > pbounds[i][1] : # penalty if out of bounds
                return -np.inf
    return 0

# Updated log_posterior function specific to E3 beam time for determining initial beam parameters
def log_posterior(theta, params, pbounds, viewerImages):
    if log_prior(theta, pbounds) == 0:
        try:
            TEMPERATURE = 160000.0
            values = convert_theta_to_values(params, theta)
            tuneChanges, initialDistChanges, run_cosy_flag = prepareMCChanges(params, values)
            cosyFile = fox_file
            initialDistribution = { 'type': 'mixed_rays',
                                    'nRays': 500,
                                    'target' : {'option': 'gaussian_2d', 'param': {'mean': (0.1304/1000, -0.24/1000), 'sigma': (0.6652/1000,0.5291/1000), 'rho': 0.0}}, \
                                    'angles' : {'option': 'gaussian_2d', 'param': {'mean': (-0.223/1000, 0), 'sigma': (6.75/2/1000, 6.75/2/1000), 'rho': 0.0}}, \
                                    'dE' : {'option': 'normal', 'param' : {'mu':  0.001, 'sigma': 0.1277/100}}, \
                                    'dZ' : {'option': 'fixed',  'param': 0}, }

            # # Make changes to initialDistribution if needed
            for initDist in initialDistChanges:
                # Accessing ay for optimizing μ and/or σ (if not assuming the same σ as ax from the D1477 scan)
                if initDist[0] == 'target_rho':
                    initialDistribution['target']['param']['rho'] = initDist[1]
                elif initDist[0] == 'ax_sigma':
                    initialDistribution['angles']['param']['sigma'] = (initDist[1], initialDistribution['angles']['param']['sigma'][1])
                elif initDist[0] == 'ay_mu':
                    initialDistribution['angles']['param']['mean'] = (initialDistribution['angles']['param']['mean'][0], initDist[1])
                elif initDist[0] == 'ay_sigma':
                    initialDistribution['angles']['param']['sigma'] = (initialDistribution['angles']['param']['sigma'][0], initDist[1])
                elif initDist[0] == 'angle_rho':
                    initialDistribution['angles']['param']['rho'] = initDist[1]
                # Accessing dE/E for optimizing μ and/or σ
                elif initDist[0] == 'dE_mu':
                    initialDistribution['dE']['param']['mu'] = initDist[1]
                elif initDist[0] == 'dE_sigma':
                    initialDistribution['dE']['param']['sigma'] = initDist[1]

            all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
                    runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, foxFile = cosyFile, tuneChanges = tuneChanges,
                        save_rays_flag = False, final_index = viewers[f"{max(int(num) for key in viewerImages for num in re.findall(r'\d+', key))}"]["index"] )
            metric = load_projection(viewerImages, all_x = all_x, all_y = all_y, all_ax = all_ax, all_dE = all_dE, transmission_indices = transmission_indices, plot = False, metric = 'chisq_1d')
            if cosyFile != fox_file:
                os.system(f"rm -rf {matrix_directory}/{cosyFile}")
                os.system(f"rm -rf {pkl_directory}/{cosyFile}.pkl")
            return -metric/TEMPERATURE
        except Exception:
            return -np.inf
    else:
        return log_prior(theta, pbounds)

# class created so multiprocessing can work more efficiently: restart processes
# that hang and after a certain number of tasks (prevents memory leaks)
class TimedRestartingPool:
    """
    Drop-in replacement for emcee's pool=... argument.
    - Provides map(func, iterable) with a per-task timeout.
    - Uses multiprocessing.Pool with maxtasksperchild to restart leaky workers.
    """
    def __init__(self, processes=None, timeout=120, maxtasksperchild=200, start_method="spawn"):
        self.timeout = timeout
        self._ctx = mp.get_context(start_method)
        self._pool = self._ctx.Pool(processes=processes, maxtasksperchild=maxtasksperchild)

    def map(self, func, iterable):
        # one task per item so timeout applies individually
        asyncs = [self._pool.apply_async(func, (x,)) for x in iterable]
        out = []
        for a in asyncs:
            try:
                out.append(a.get(timeout=self.timeout))
            except mp.TimeoutError:
                # Treat timeouts as invalid points so the sampler can keep going
                out.append(-np.inf)
            except Exception:
                # You can choose to re-raise; returning -inf keeps the run alive
                out.append(-np.inf)
        return out

    def close(self):
        self._pool.close()
        self._pool.join()

    # context-manager support
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

# Run MCMC optimization
def optimizeMCMC(params, viewerImages, fresh_start = True):
    '''
    params: list of parameters to optimize
    fresh_start: start from scratch or from previous optimization True/False
    '''
    rng = np.random.default_rng(42)

    # --- Run MCMC ---
    ndim, pbounds = countSECARparams(params)
    # nwalkers, nsteps = 26, 2000
    nwalkers, nsteps = 12, 200

    if fresh_start:
        os.system(f"rm -f chain.h5")
        backend = HDFBackend("chain.h5")    # Save as you go in case we want to restart optimization
        backend.reset(nwalkers, ndim)
        p0 = np.column_stack([rng.uniform(pbounds[i][0], pbounds[i][1], nwalkers) for i in range(ndim)])
    else:
        backend = HDFBackend("chain.h5")    # Save as you go in case we want to restart optimization
        p0 = backend.get_last_sample()

    # logp = partial(log_posterior, params=params, pbounds = pbounds, images2compare = images2compare)
    logp = partial(log_posterior, params=params, pbounds = pbounds, viewerImages = viewerImages)
    nprocs = max(1, mp.cpu_count() - 1)

    # Per-eval timeout (seconds) and periodic worker restarts
    EVAL_TIMEOUT = 240          # ~2–3× typical runECAT time
    MAXTASKS_PER_CHILD = 200    # restart workers every ~200 tasks
    with TimedRestartingPool(processes=nprocs,timeout=EVAL_TIMEOUT, maxtasksperchild=MAXTASKS_PER_CHILD, start_method="spawn") as pool:
        #moves = [ (emcee.moves.StretchMove(a=1.5), 0.5), (emcee.moves.DEMove(gamma0=0.8), 0.3), (emcee.moves.KDEMove(), 0.2),]
        moves = [(emcee.moves.StretchMove(a=2.0), 0.7), (emcee.moves.DEMove(gamma0=0.8), 0.3)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, backend=backend, moves=moves, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

# OLD FUNCTIONS:

# Loading projections from csv files AND/OR returns metric value when comparing viewer and ECAT simulations
# def load_projection(imagesByGroup, group_number, all_x = None, all_y = None, transmission_indices = None, plot = True, metric = None, save_fig = False, all_ax = None, all_dE = None):
#     '''
#     metric : metric to compare viewer images against ECAT results. Can be: 'chisq_1d (X and Y projections of a viewer), chisq_2d (viewer)'
#     returns metric value
#     '''
#     chisq_x, chisq_y = 0, 0
#     metric_value = 0
#     group_path = f"PartD_ViewerAnalysis/Group{group_number}"
#     for viewer in imagesByGroup:   # go viewer by viewer
#         if metric == 'chisq_1d' or plot == True:
#             x_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_xf_cosy.csv")
#             y_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_yf_cosy.csv")
#             # Load 1d viewer projections and skip any lines starting with a '#'
#             df_x = pd.read_csv(x_projection_path, comment = "#")
#             df_y = pd.read_csv(y_projection_path, comment = "#")
#
#             # Extract the 1d viewer projections data
#             x_pos = -df_x.iloc[:, 0].to_numpy()
#             x_intensity = df_x.iloc[:, 1].to_numpy()
#             y_pos = df_y.iloc[:, 0].to_numpy()
#             y_intensity = df_y.iloc[:, 1].to_numpy()
#
#             x_mean = np.average(x_pos, weights=x_intensity)
#             x_sigma = np.sqrt( np.average((x_pos-x_mean)**2, weights=x_intensity) )
#             y_mean = np.average(y_pos, weights=y_intensity)
#             y_sigma = np.sqrt( np.average((y_pos-y_mean)**2, weights=y_intensity) )
#             print(f"VIEWER Group {group_number} - {viewer}, X mean [mm]: {x_mean:.2f},  X sigma [mm]: {x_sigma:.2f}")
#             print(f"VIEWER Group {group_number} - {viewer}, Y mean [mm]: {y_mean:.2f},  Y sigma [mm]: {y_sigma:.2f}")
#
#             # Creating 1d viewer histograms (binned)
#             x_bins = 50
#             x_scale = x_bins/len(x_pos)
#             x_intensityC = [i*x_scale for i in x_intensity]
#             x_intensityC_sum = np.sum(x_intensityC)
#             x_intensityC = [max(0, round(100*x_bins*i/x_intensityC_sum)) for i in x_intensityC]   # Scale binned viewer so sum is 100*x_bins
#             x_intensityC[0] = max(1, x_intensityC[0] )                                            # Making sure the extrems are included
#             x_intensityC[-1] = max(1, x_intensityC[-1] )                                          # Making sure the extrems are included
#             x_intensity  = [100*x_bins*i/x_intensityC_sum for i in x_intensity]                   # Scale orig viewer so agrees with binned
#             x_vd = [ p for p,n in zip(x_pos, x_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed
#
#             y_bins = 50
#             y_scale = y_bins/len(y_pos)
#             y_intensityC = [i*y_scale for i in y_intensity]
#             y_intensityC_sum = np.sum(y_intensityC)
#             y_intensityC = [max(0, round(100*y_bins*i/y_intensityC_sum)) for i in y_intensityC]   # Scale binned viewer so sum is 100*y_bins
#             y_intensityC[0] = max(1, y_intensityC[0] )                                            # Making sure the extrems are included
#             y_intensityC[-1] = max(1, y_intensityC[-1] )                                          # Making sure the extrems are included
#             y_intensity  = [100*y_bins*i/y_intensityC_sum for i in y_intensity]                   # Scale orig viewer so agrees with binned
#             y_vd = [ p for p,n in zip(y_pos, y_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed
#
#             # Creating ecat histograms if they are used
#             if all_x is not None:
#                 # Find position index using viewer
#                 posV = viewers[viewer]["index"]
#                 posS = element_names_stripped.index('FP1 Slits')  # position slits
#                 if posS < posV:   # Viewer is after fp1 slits
#                     x = []
#                     y = []
#                     aX = []
#                     dE = []
#                     valid_indices = []
#                     for i in range(len(all_x)):
#                         if transmission_indices[i] >= posV \
#                           and groupsInfo[group_number]['fp1_slits']["slits_mm"][0] <= -all_x[i][posS]*1000 \
#                           and -all_x[i][posS]*1000 <= groupsInfo[group_number]['fp1_slits']["slits_mm"][1]:
#                             valid_indices.append(i)
#                             x.append( -all_x[i][posV]*1000 )
#                             aX.append( -all_ax[i][0]*1000 )   # Append the angles at target that arrive at this location!
#                             y.append( all_y[i][posV]*1000 )
#                             dE.append( all_dE[i][posV]*100 )
#                     x = np.array(x)
#                     aX = np.array(aX)
#                     y = np.array(y)
#                     dE = np.array(dE)
#                 else:
#                     x  = np.array([-xx[posV]*1000 for i, xx in enumerate(all_x)  if transmission_indices[i] >= posV]) # in mm
#                     aX = np.array([ -ax[0]*1000    for i, ax in enumerate(all_ax) if transmission_indices[i] >= posV]) # in mm
#                     y  = np.array([yy[posV]*1000 for i, yy in enumerate(all_y)  if transmission_indices[i] >= posV]) # in mm
#                     dE  = np.array([de[posV]*100 for i, de in enumerate(all_dE) if transmission_indices[i] >= posV]) # in mm
#                 print(f"ECAT Group {group_number} - {viewer}, X mean [mm]: {np.mean(x):.2f},  X std [mm]: {np.std(x):.2f}")
#                 print(f"ECAT Group {group_number} - Arriving at {viewer}, aX at target mean [mm]: {np.mean(aX):.2f},  aX std [mm]: {np.std(aX):.2f}")
#                 print(f"ECAT Group {group_number} - {viewer}, Y mean [mm]: {np.mean(y):.2f},  Y std [mm]: {np.std(y):.2f}")
#                 print(f"ECAT Group {group_number} - {viewer}, dE mean [mm]: {np.mean(dE):.2f},  dE std [mm]: {np.std(dE):.2f}")
#
#                 # Scale ecat data to viewer counts
#                 x_edges = np.histogram_bin_edges(x_vd, bins=x_bins)
#                 vd_counts, _ = np.histogram(x_vd, bins=x_edges)
#                 ecat_counts, _ = np.histogram(x, bins=x_edges)
#                 widths = np.diff(x_edges)
#                 area_vd   = np.sum(vd_counts * widths)
#                 area_ecat = np.sum(ecat_counts * widths)
#                 x_scale = area_vd / area_ecat if area_ecat > 0 else 0.0   # weights of ECAT x-histogram
#                 # Calculating chi-square
#                 viewer_counts, edges = np.histogram(x_vd, bins=x_bins)
#                 ecat_counts, edges = np.histogram(x, bins=x_edges, weights=np.full(len(x), x_scale))
#
#                 if np.abs(np.mean(x) - np.mean(x_vd)) > np.max(x_pos): # if ECAT mean minus viewer mean is more than size of the viewer
#                     chisq_x = np.inf
#                 else:
#                     chisq_x = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)
#
#             if all_y is not None and all_x is not None:
#                 # Find position index using viewer
#                 if element_names_stripped.index('FP1 Slits') < posV:   # Viewer is after fp1 slits
#                     y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if i in valid_indices]) # in mm
#                 else:
#                     y = np.array([yy[posV]*1000 for i, yy in enumerate(all_y) if transmission_indices[i] >= posV]) # in mm
#                 # Scale ecat data to viewer counts
#                 y_edges = np.histogram_bin_edges(y_vd, bins=y_bins)
#                 vd_counts, _ = np.histogram(y_vd, bins=y_edges)
#                 ecat_counts, _ = np.histogram(y, bins=y_edges)
#                 widths = np.diff(y_edges)
#                 area_vd   = np.sum(vd_counts * widths)
#                 area_ecat = np.sum(ecat_counts * widths)
#                 y_scale = area_vd / area_ecat if area_ecat > 0 else 0.0  # weights of ECAT y-histogram
#                 # Calculating chi-square
#                 viewer_counts, edges = np.histogram(y_vd, bins=y_bins)
#                 ecat_counts, edges = np.histogram(y, bins=y_edges, weights=np.full(len(y), y_scale))
#                 chisq_y = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)
#
#         if metric == 'chisq_2d' or plot == True:
#             image_path = os.path.join(group_path, f"Group{group_number}_VD{viewer}_2D_cosy.npz")
#             # Load 2d viewer data
#             data = np.load(image_path)
#             corrected_full_image = data["corrected_full_image"]
#             corrected_full_image = corrected_full_image[:, ::-1]
#             x2d_range = data["xf_range"]
#             y2d_range = data["yf_range"]
#
#     	# Optionally plot the projection
#         if plot:
#             fig, axs = plt.subplots(2, 2, figsize=(17,9))
#             labelPP    = groupsInfo[group_number]['pp_alignment']['label']
#             labelSlits = groupsInfo[group_number]['fp1_slits']['label']
#             plt.suptitle(f"Group {group_number} - {labelPP} - {labelSlits} - {viewer}\n(Viewing with the beam)", fontsize=16)
#
#             axs[0,0].plot(x_pos, x_intensity, label = 'Viewer original')
#             axs[0,0].hist(x_vd, bins=x_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
#             if all_x is not None:
#                 axs[0,0].hist(x, bins=x_edges, histtype='step', linewidth=4, weights=np.full(len(x), x_scale),  # uniformly scale ecat
#                     label=f'ecat × {x_scale:.3g}', color = 'orange')
#             axs[0,0].set_xlabel("X [mm]")
#             axs[0,0].set_ylabel("Intensity")
#             axs[0,0].set_title("X projection", fontweight = 'bold')
#             axs[0,0].set_xlim(min(x_pos), max(x_pos))
#             axs[0,0].legend(loc='best')   # place legend
#             axs[0,0].text( 0.01, 0.98, f"chi-sq = {chisq_x:.1f}", transform=axs[0,0].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')
#
#             axs[0,1].plot(y_pos, y_intensity, label = 'Viewer original')
#             axs[0,1].hist(y_vd, bins=y_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
#             if all_y is not None:
#                 axs[0,1].hist(y, bins=y_edges, histtype='step', linewidth=4, weights=np.full(len(y), y_scale),  # uniformly scale ecat
#                     label=f'ecat × {y_scale:.3g}', color = 'orange')
#             axs[0,1].set_xlabel("Y [mm]")
#             axs[0,1].set_ylabel("Intensity")
#             axs[0,1].set_title("Y projection", fontweight = 'bold')
#             axs[0,1].set_xlim(min(y_pos), max(y_pos))
#             axs[0,1].legend(loc='best')   # place legend
#             axs[0,1].text( 0.01, 0.98, f"chi-sq = {chisq_y:.1f}", transform=axs[0,1].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')
#
#             im = axs[1,0].imshow( corrected_full_image, extent = (-x2d_range[-1], -x2d_range[0], y2d_range[-1], y2d_range[0]), \
#                 cmap = 'viridis', aspect = 'auto')
#             cbar = fig.colorbar(im, ax=axs[1,0])
#             axs[1,0].set_xlim(-x2d_range[-1], -x2d_range[0])
#             axs[1,0].set_ylim(y2d_range[0], y2d_range[-1])
#             axs[1,0].set_xlabel('X (mm)')
#             axs[1,0].set_ylabel('Y (mm)')
#             axs[1,0].set_title("Viewer", fontweight = 'bold')
#
#             if all_x is not None and all_y is not None:
#                 axs[1,1].hist2d(x, y, bins=(50, 50), cmap='viridis',range=[[-x2d_range[-1], -x2d_range[0]],[y2d_range[0], y2d_range[-1]]])
#                 axs[1,1].set_xlabel('X (mm)')
#                 axs[1,1].set_ylabel('Y (mm)')
#                 axs[1,1].set_title("ECAT", fontweight = 'bold')
#             plt.tight_layout()
#             if save_fig:
#                 save_path = f"{results_directory}/imageCompared_Group{group_number}_VD{viewer}"
#                 plt.savefig(save_path, dpi = 300)
#             else:
#                 plt.show()
#         if metric == 'chisq_1d':
#             if (group_number == "10" and viewer == "1783") or (group_number == "14" and viewer == "1783"):
#                 metric_value = metric_value + chisq_x
#             else:
#                 metric_value = metric_value + chisq_x + chisq_y
#
#     return metric_value

# def log_posterior(theta, params, pbounds, images2compare):
#     if log_prior(theta, pbounds) == 0:
#         try:
#             TEMPERATURE = 160000.0
#             values = convert_theta_to_values(params, theta)
#             # tuneChangesPrev = [['Q1', 'Y', -0.000346253403085111], ['Q1', 'B_SC', 0.012213815413722453], ['Q2', 'Y', -0.0004883738170837214], ['Q2', 'B_SC', 0.014579957615870023], ['B2 Exit', 'dB', -0.0029242376594304244], ['Q3', 'B_SC', 0.00455452181737245], ['Q4', 'B_SC', 0.0028608744765512744], ['Q5', 'B_SC', -0.02148827880573774]]
#             # tuneChangesPrev = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125] ]
#             #tuneChangesPrev = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125], ['B5 Exit', 'dB', -0.005], ['B6 Exit', 'dB', -0.005], ['WF2 Exit', 'dB', -0.005]]
#             #tuneChangesPrev = [ ['Q1', 'Y', -0.00037575856111501927], ['Q1', 'B_SC', 0.011297666768643053], ['Q2', 'Y',  -0.0005914722202508483], ['Q2', 'B_SC', -0.001612967967599542], ['Q3', 'B_SC', -0.02234262988240511], ['Q4', 'B_SC', 0.019381765829497632], ['Q5', 'B_SC', -0.0029501097224348145] ]
#             #tuneChangesPrev = [ ['Q1', 'Y', -0.00037575856111501927], ['Q1', 'B_SC', 0.011297666768643053], ['Q2', 'Y',  -0.0005914722202508483], ['Q2', 'B_SC', -0.001612967967599542], ['Q3', 'B_SC', -0.02234262988240511], ['Q4', 'B_SC', 0.019381765829497632], ['Q5', 'B_SC', -0.0029501097224348145], ['B3 Exit', 'dB', +0.001], ['B4 Exit', 'dB', +0.001], ['WF1 Exit', 'dB', 0.001], ['B5 Exit', 'dB', -0.0015], ['B6 Exit', 'dB', -0.0015], ['WF2 Exit', 'dB', -0.0004] ]
#             tuneChangesPrev = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125], ['B5 Exit', 'dB', -0.0053], ['B6 Exit', 'dB', -0.0053], ['WF2 Exit', 'dB', -0.0059]]
#
#             tuneChanges, initialDistChanges, run_cosy_flag = prepareMCChanges(params, values)
#             tuneChanges = tuneChangesPrev + tuneChanges
#             metric = 0
#             cosyFile = fox_file
#             for groupImages in images2compare:
#                 group = groupImages[0]
#                 imagesByGroup = groupImages[1]
#                 initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
#                                         'nRays': 7000,
#                                         'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y'], "R": groupsInfo[group]['target_alignment']['R']},
#                                         'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
#                                         'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
#                                         'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},      # 2, 8
#                                         'dZ' : {'option': 'fixed',  'param': 0}, }
#                 for initDist in initialDistChanges:  # Make changes to initialDistribution if needed
#                     if initDist[0] == 'mu_y':
#                         initialDistribution['angles']['mu_y'] = initDist[1]
#                     elif initDist[0] == 'mu':
#                         initialDistribution['dE']['param']['mu'] = initDist[1]
#                     elif initDist[0] == 'sigma':
#                         initialDistribution['dE']['param']['sigma'] = initDist[1]
#                 all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
#                     runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, foxFile = cosyFile, tuneChanges = tuneChanges,
#                         save_rays_flag = False, final_index = viewers[f"{max(int(n) for _, nums in images2compare for n in nums)}"]["index"] )
#                 run_cosy_flag = False # Only needs to run cosy once for every MCMC iteration
#                 metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
#                     plot = False, metric = 'chisq_1d')
#             if cosyFile != fox_file:
#                 os.system(f"rm -rf {matrix_directory}/{cosyFile}")
#                 os.system(f"rm -rf {pkl_directory}/{cosyFile}.pkl")
#             return -metric/TEMPERATURE
#         except Exception:
#             return -np.inf
#     else:
#         return log_prior(theta, pbounds)
