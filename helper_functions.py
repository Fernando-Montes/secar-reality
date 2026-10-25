import io, os, re, pickle, zipfile, matplotlib, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import random
from os.path import exists
from matplotlib.gridspec import GridSpec
import warnings
import time
from scipy.integrate import trapezoid
from shapely.geometry.polygon import orient
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

# Import Cython functions for ray generation and processing [See "ray_functions.pyx" & "setup.py"]
from ray_functions import stepped_uniform_rays, mixed_rays, aperture_rays, process_all_rays, extract_coordinates
from fast_tracer import trace_rays
from misalign_blocks import MISALIGN_BLOCKS, MISALIGN_PARS, block_changes

# Import system configuration [See "config.py"]
from config import (
    ECAT_directory, COSY_directory, matrix_directory, pkl_directory,    # Defines base directories
    results_directory, fox_file, cosy_lines, matrix_file, beamline_geometry, z_coordinates,
    element_names_stripped,                 # Defines system geometry
    save_rays_file,
    viewers, detectors, faraday_cups   		# Dimensions
)

###########################################
########## FOR THE COSY MATRICES ##########
###########################################

# Functions that separates between changes in the .fox file and mistunes to dipoles/WF through field/energies changes to the rays
def prepareTuneChanges(tuneChanges):
    '''
    tuneChanges = [ ['Q1', 'Roll', -0.08], ['B1 Exit', 'dB', 0.01] ]
    '''
    params = []
    values = []
    dipoleWFchanges = []
    for change in tuneChanges:
        if change[1] != 'dB':
            if change[1] == 'XY':
                params.append( {'elem': change[0], 'par': 'XY'} )
                values.append( [change[2][0], change[2][1]] )
            elif change[1] == 'rhoEB':
                params.append( {'elem': change[0], 'par': 'WFR'} )
                values.append( [change[2][0], change[2][1]] )
            else:
                params.append( {'elem': change[0], 'par': change[1]} )
                values.append( change[2] )
        else:
            dipoleWFchanges.append( [ element_names_stripped.index(change[0])-1, change[2] ] )
    # AgentSearch fix (2026-08-16): process_single_ray (ray_functions.pyx)
    # walks the beamline with a single forward-only pointer into
    # dipoleWFchanges, advancing only when it hits that entry's element
    # index -- it silently requires dipoleWFchanges to already be sorted
    # by ascending element index, or every entry after the first
    # out-of-order one is never applied (the pointer can't go backwards).
    # Confirmed real: fp2_family_mc.py's v7 KNOBS lists WF1 (index 33)
    # before B1-B4 (indices 7/9/22/24), so throughout the entire v7
    # search B1-B4 were silently never applied to a single ray --
    # verified by recomputing one v7 candidate's M12/M16 with the dB
    # entries in caller order vs sorted order and finding they differ
    # (verify_dipole_order_bug.py). Sorting here fixes every caller at
    # once, regardless of what order they build tuneChanges in.
    dipoleWFchanges.sort(key=lambda c: c[0])
    return params, values, dipoleWFchanges

# Number formats for the values prepareCOSYchanges writes into the .fox file. The defaults
# reproduce the decks written so far; queue_eval.py (eval-queue SPEC v0.2, 2026-10-23) raises
# them so that ridge-solve values are not rounded in the deck.
DECK_FMT = ".6f"
WFR_FMT = ".3f"

# Function that creates array CosyChanges that is used when running ECAT when there are differences to the nominal
# COSY file
def prepareCOSYchanges(params, values):
    '''
    params: Array of SECAR parameters to change
    values: Values used to be added/subtracted to nominal cosy values
    '''
    cosyChanges = []
    cosyFile = ''
    if params != None:
        # FP3-section elements: all misalignment knobs of an element are
        # written together as one block (see misalign_blocks.py)
        block_deltas = {}
        for p, v in zip(params, values):
            if p['elem'] in MISALIGN_BLOCKS and p['par'] in MISALIGN_PARS:
                block_deltas.setdefault(p['elem'], {})[p['par']] = v
        for elem, deltas in block_deltas.items():
            cosyChanges.extend(block_changes(elem, deltas))
        for i in range(len(params)):
            if params[i]['elem'] in block_deltas and params[i]['par'] in MISALIGN_PARS:
                continue
            cosyFile = cosyFile + params[i]['elem']
            if params[i]['par'] == 'Roll':
                cosyFile = cosyFile + "R" + str(f"{1000*values[i]:.1f}")
                val = cosy_lines[ params[i]['elem'] ][ 'Roll' ]['nom']+values[i]
                text = cosy_lines[ params[i]['elem'] ][ 'Roll' ]['text1'] + f"{val:{DECK_FMT}}" + \
                       cosy_lines[ params[i]['elem'] ][ 'Roll' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'Roll' ]['line'][0] , text] )
                text = cosy_lines[ params[i]['elem'] ][ 'Roll' ]['text1'] + f"{-val:{DECK_FMT}}" + \
                       cosy_lines[ params[i]['elem'] ][ 'Roll' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'Roll' ]['line'][1] , text] )
            elif params[i]['par'] == 'dist':
                cosyFile = cosyFile + "d" + str(f"{1000*values[i]:.1f}")
                val = cosy_lines[ params[i]['elem'] ][ 'dist' ]['nom'][0]+values[i]
                text = cosy_lines[ params[i]['elem'] ][ 'dist' ]['text1'] + f"{val:{DECK_FMT}}" + \
                       cosy_lines[ params[i]['elem'] ][ 'dist' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dist' ]['line'][0] , text] )
                text = cosy_lines[ params[i]['elem'] ][ 'dist' ]['text1'] + f"{-val:{DECK_FMT}}" + \
                       cosy_lines[ params[i]['elem'] ][ 'dist' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dist' ]['line'][1] , text] )
            elif params[i]['par'] == 'B_SC':
                cosyFile = cosyFile + "B" + str(f"{100*values[i]:.2f}")
                val = cosy_lines[ params[i]['elem'] ][ 'B_SC' ]['nom']+values[i]
                text = cosy_lines[ params[i]['elem'] ][ 'B_SC' ]['text1'] + f"{val:{DECK_FMT}}" + \
                       cosy_lines[ params[i]['elem'] ][ 'B_SC' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'B_SC' ]['line'] , text] )
            elif params[i]['par'] == 'X':
                cosyFile = cosyFile + "X" + str(f"{1000*values[i]:.1f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][0]+values[i]
                val2 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][1]
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{val1:{DECK_FMT}}" + " " + f"{val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][0] , text] )
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{-val1:{DECK_FMT}}" + " " + f"{-val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][1] , text] )
            elif params[i]['par'] == 'Y':
                cosyFile = cosyFile + "Y" + str(f"{1000*values[i]:.1f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][0]
                val2 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][1]+values[i]
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{val1:{DECK_FMT}}" + " " + f"{val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][0] , text] )
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{-val1:{DECK_FMT}}" + " " + f"{-val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][1] , text] )
            elif params[i]['par'] == 'XY':
                cosyFile = cosyFile + str(f"X{1000*values[i][0]:.1f}Y{1000*values[i][1]:.1f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][0]+values[i][0]
                val2 = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['nom'][1]+values[i][1]
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{val1:{DECK_FMT}}" + " " + f"{val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][0] , text] )
                text = cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text1'] + f"{-val1:{DECK_FMT}}" + " " + f"{-val2:{DECK_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'dXY' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'dXY' ]['line'][1] , text] )
            elif params[i]['par'] == 'rhoE':
                cosyFile = cosyFile + "rhoE" + str(f"{values[i]:.2f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][0]+values[i]
                val2 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][1]
                text = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text1'] + f"{val1:{WFR_FMT}}" + " " + f"{val2:{WFR_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'WFR' ]['line'] , text] )
            elif params[i]['par'] == 'rhoB':
                cosyFile = cosyFile + "rhoB" + str(f"{values[i]:.2f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][0]
                val2 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][1]+values[i]
                text = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text1'] + f"{val1:{WFR_FMT}}" + " " + f"{val2:{WFR_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'WFR' ]['line'] , text] )
            elif params[i]['par'] == 'rhoEB':
                cosyFile = cosyFile + str(f"rhoE{values[i][0]:.2f}B{values[i][1]:.2f}")
                val1 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][0]+values[i][0]
                val2 = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['nom'][1]+values[i][1]
                text = cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text1'] + f"{val1:{WFR_FMT}}" + " " + f"{val2:{WFR_FMT}}"  + \
                       cosy_lines[ params[i]['elem'] ][ 'WFR' ]['text2']
                cosyChanges.append( [ cosy_lines[ params[i]['elem'] ][ 'WFR' ]['line'] , text] )
            elif params[i]['par'] == 'length' and params[i]['elem'] == 'WF1':
                # AgentSearch addition, 2026-08-08 (not in the original ECAT
                # codebase): WF1's physical length (WC line 593), with
                # compensating adjustments to the drifts immediately before
                # (line 575, "DL18") and after (line 609, "DL19") so the
                # overall SECAR beamline length is unchanged. Confirmed with
                # user: shrink each neighboring drift by delta/2.
                # Nominal values, read directly from COSY10.2/pg_June2025.fox:
                #   line 575: DL 0.496188;              (DL18)
                #   line 593: WC WF1R WF1R 2.365 0.11 NE1 NM1 4 ;  (WF1 length = 2.365)
                #   line 609: DL 0.498516-0.1;           (DL19, "part of WF1")
                delta = values[i]
                cosyFile = cosyFile + "WF1len" + str(f"{100*delta:.2f}")
                WF1_LEN_NOM = 2.365
                DL18_NOM = 0.496188
                DL19_NOM = 0.498516 - 0.1  # = 0.398516; written as a plain number, not the original expression
                new_wf1_len = WF1_LEN_NOM + delta
                new_dl18 = DL18_NOM - delta / 2
                new_dl19 = DL19_NOM - delta / 2
                cosyChanges.append([575, f"DL {new_dl18:{DECK_FMT}}; {{DL18}}\n"])
                cosyChanges.append([593, f"WC WF1R WF1R {new_wf1_len:{DECK_FMT}} 0.11 NE1 NM1 4 ;                       {{WF1}}\n"])
                cosyChanges.append([609, f"DL {new_dl19:{DECK_FMT}}; {{THIS IS PART OF WF1!}}                    {{DL19}}\n"])
    # AgentSearch fix (2026-08-16): the descriptive cosyFile string built
    # above rounds each value to 2 decimal places and used to be truncated
    # to its last 40 characters (a real COSY filename-length limit) --
    # with several B_SC-type knobs concatenated, this silently collides
    # between DIFFERENT tune combinations (confirmed: far fewer unique
    # Matrix_Files folders exist than genuinely distinct compiles run
    # across this project, and folder names show visible truncation
    # artifacts), especially since truncation drops the EARLIEST knobs
    # and 2-decimal-percent rounding gives only ~100 distinguishable
    # values per knob. Under the multiprocessing.Pool parallelism used
    # throughout, two different tasks landing on the same truncated name
    # is a genuine race condition, not just an overwrite. Fixed by hashing
    # the exact (params, values) pair instead of truncating the lossy
    # display string -- deterministic, well under the 40-char limit, and
    # collision probability is negligible for any realistic run count.
    # fox_file (the template) is part of the key so the same knob change on
    # two different base lattices never shares a cache name.
    if params:
        digest = hashlib.md5(repr((fox_file, list(zip(params, values)))).encode()).hexdigest()
        cosyFile = "cs" + digest[:14]
    else:
        cosyFile = cosyFile[-40:]
    return cosyChanges, cosyFile

# Function that creates COSY file if needed
def createCosyFile(cosyChanges, file_name = "temp"):
    if cosyChanges != None:
        current_directory = os.getcwd()
        os.chdir(COSY_directory)
        lines = open(f"{fox_file}.fox", 'r').readlines()
        for i in range(len(cosyChanges)):
            lines[cosyChanges[i][0]-1] = cosyChanges[i][1]
        out = open(f"{file_name}.fox", 'w')
        out.writelines(lines)
        out.write('\n')
        out.close()

        os.chdir(current_directory)
        return file_name
    else:
        return fox_file

# Reading the COSY fort files
def read_file(file_content):
    file_io = io.StringIO(file_content)
    COLUMN_NAMES = ["I", "Coefficient", "Order", "x", "ax", "y", "ay", "l", "dE", "dm", "dz"]
    df = pd.read_csv(file_io, delimiter = '\s+', skiprows = 4, header = None, names = COLUMN_NAMES, na_values = ['---', '--', '-'])
    df.dropna(inplace = True)
    return df

# Dividing the fort files into the individual matrices for each ray parameter
def split_fort(df):

    sections = []
    start_idx = None

    for idx, row in df.iterrows():
        if row['I'] == 1 or row['I'] == '1':
            if start_idx is not None:
                sections.append(df.loc[start_idx:idx - 1])
            start_idx = idx

    # Handling the last section if it ends with 'I == 1'
    if start_idx is not None and start_idx != len(df) - 1:
        sections.append(df.loc[start_idx:])

    return sections

# Processing the matrices
def process_section(df):

    columns_to_convert = ["Coefficient", "Order", "x", "ax", "y", "ay", "l", "dE", "dm", "dz"]

    for column in columns_to_convert:

        mask = pd.notnull(df[column])
        df.loc[mask, column] = pd.to_numeric(df.loc[mask, column], errors = 'coerce')

        # Convert the column to NumPy float64 type using .loc[]
        df.loc[:, column] = df.loc[:, column].astype(np.float64)

    return df

# Run COSY and save fort files in a zip file per set of optics
def run_cosy_and_save_fort(cosy_command, cosyFile):
    current_directory = os.getcwd()
    new_dir = os.path.join(matrix_directory, cosyFile)
    os.makedirs(new_dir, exist_ok = True)
    os.system("cp " + COSY_directory + f"/{cosyFile}.fox " + new_dir)
    os.system("cp " + COSY_directory + "/cosy " + new_dir)
    os.system("cp " + COSY_directory + "/cosy.bin " + new_dir)
    os.chdir(new_dir)
    os.system(cosy_command)
    os.chdir(current_directory)

# Process fort files from zip and return .pkl
def process_fort_files(pkl_file_name):

    new_dir = os.path.join(matrix_directory, pkl_file_name)
    # Select specific fort files
    fort_file_pattern = re.compile(r'^fort\.(25[0-9]|2[6-9][0-9]|30[0-9]|31[0-9]|32[0-9])$')
    fort_files = [f for f in os.listdir(new_dir) if fort_file_pattern.match(f)]

    # Sort numerically based on the number in 'fort.###'
    fort_files.sort(key=lambda x: int(re.search(r'fort\.(\d+)', x).group(1)))

    processed_sections = []
    for filename in fort_files:
        with open(os.path.join(new_dir, filename), 'r') as f:
            df = read_file(f.read())  # <- You must define/import this
            sections = split_fort(df)  # <- You must define/import this
            processed_sections.extend(process_section(s) for s in sections)  # <- You must define/import this

    pkl_file_path = os.path.join(pkl_directory, f"{pkl_file_name}.pkl")
    with open(pkl_file_path, 'wb') as f:
        pickle.dump({'data_frames': processed_sections}, f)

    return pkl_file_path

# Load processed optics matrices
def load_optics_data(matrix_file, pkl_directory):
    with open(os.path.join(pkl_directory, f"{matrix_file}.pkl"), 'rb') as f:
        processed_data = pickle.load(f)

    df = processed_data['data_frames']
    group_size = 8
    grouped_data_frames = [df[i: i + group_size] for i in range(0, len(df), group_size)]

    return df, grouped_data_frames

###########################################
########## DEFINING THE BEAMLINE ##########
###########################################

# To determine the vacuum chambers and corresponding polygons at given z_values
def find_polygons_at_z(beamline_data, z_values):
    names = []
    polygons = []
    for z in z_values:
        poly = None
        for segment in beamline_data:
            if segment['z_start'] <= z < segment['z_end']:
                poly = segment['polygon']
                name = segment['name']
                break
        names.append(name)
        polygons.append(poly)
    return names, polygons

#########################################
########## DEFINE INITIAL RAYS ##########
#########################################

def generateInitialDistribution(initialDistribution):

    if initialDistribution == None:
        initialDistribution = { 'type': 'stepped_uniform',
                                'x' : {'min': -0.75/1000, 'max': 0.75/1000, 'steps': 3},
                                'ax' : {'min': -25/1000, 'max': 25/1000, 'steps': 7},
                                'y' : {'min': -0.75/1000, 'max': 0.75/1000, 'steps': 3},
                                'ay' : {'min': -25/1000, 'max': 25/1000, 'steps': 3},
                                'dE' : {'min': -3.1/100, 'max': 3.1/100, 'steps': 5},
                                'dZ' : {'min': 0, 'max': 0, 'steps': 1} }

    if initialDistribution['type'] == 'stepped_uniform':
        '''
        Generates rays using uniformly stepped values across each coordinate (x, ax, y, ay, dE, dz) when
        provided a dictionary *_params = {'min': #, 'max': #, 'steps': #}. One can also impose the COSY_condition
                [ax / max(ax)]^2 + [ay / max(ay)]^2 + [dE / max(dE)]^2 < 1
        to only select rays that fill the 3D ellipsoid phase space
        Example:
        initialDistribution = { 'type': 'stepped_uniform',
                                'x' : {'min': -0.75/1000, 'max': 0.75/1000, 'steps': 3},
                                'ax' : {'min': -25/1000, 'max': 25/1000, 'steps': 7},
                                'y' : {'min': -0.75/1000, 'max': 0.75/1000, 'steps': 3},
                                'ay' : {'min': -25/1000, 'max': 25/1000, 'steps': 3},
                                'dE' : {'min': -3.1/100, 'max': 3.1/100, 'steps': 5},
                                'dZ' : {'min': 0, 'max': 0, 'steps': 1} }
        '''
        x_params  = initialDistribution['x']
        ax_params = initialDistribution['ax']
        y_params  = initialDistribution['y']
        ay_params = initialDistribution['ay']
        dE_params = initialDistribution['dE']
        dz_params = initialDistribution['dZ']
        rays = stepped_uniform_rays(x_params, ax_params, y_params, ay_params, dE_params, dz_params, COSY_condition = True)

    if initialDistribution['type'] == 'mixed_rays':
        '''
        The most flexible of all the ray generation functions, mixed_rays generates nRays where each coordinate is
        sampled from independent distributions. Positions (x, y) and angles
        (ax, ay) can be also sampled jointly to generate circular and, in the case of angles, isotropic distributions.
        For target and angles:
            _ : {'option' : 'circle',       'param' : {'center': (#, #), 'radius': #}
            _ : {'option' : 'isotropic',    'param' : {'theta_max': #} --- NOTE: This is specifically for angles!
            _ : {'option' : 'independent',  'param' : { {'X' : ..see below.. }, { 'Y' : ..see below..} }
        For dE, dZ and if target' and angles' independent distributions selected:
            _ : {'option' : 'fixed',    'param' : # }
            _ : {'option' : 'uniform',  'param' : {'min': #, 'max': #}
            _ : {'option' : 'normal',   'param' : {'mu': #, 'sigma': #}
            _ : {'option' : 'skewed',   'param' : {'alpha': #, 'mu': #, 'sigma': #}
        Example:
        initialDistribution = { 'type': 'mixed_rays',
                                'nRays': 50,
                                'target' : {'option': 'circle', 'param': {'center': (0, 0), 'radius': 0.75/1000}}, \
                                'angles' : {'option': 'circle', 'param': {'center': (0, 0), 'radius': 5 / 1000}}, \
                                'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}}, \
                                'dZ' : {'option': 'fixed',  'param': 0}, }
        '''
        # Define ray input parameters for each dimension X, Y, aX, aY, dE/E, charge state
        nRays = initialDistribution['nRays']
        position_option = initialDistribution['target']['option']
        position_params = initialDistribution['target']['param']
        angles_option   = initialDistribution['angles']['option']
        angles_params   = initialDistribution['angles']['param']
        dE_option = initialDistribution['dE']['option']
        dE_params = initialDistribution['dE']['param']
        dz_option = initialDistribution['dZ']['option']
        dz_params = { initialDistribution['dZ']['option'] : initialDistribution['dZ']['param'] }
        # Call the ray function with the parameters
        rays = mixed_rays(position_option, position_params,
                          angles_option, angles_params,
                          dE_option, dE_params,
                          dz_option, dz_params, nRays)

    if initialDistribution['type'] == 'aperture':
        '''
        Generates nRays originating from a circular target and determines which are transmitted through
        a circular downstream aperture. The aperture sizes and positions are defined by the first seven input parameters:
                target_radius, target_offset_x, target_offset_y,
                aperture_radius, aperture_offset_x, aperture_offset_y,
                separation_distance
        The angular distribution is described by the mean angular offsets' theta_mu_x and theta_mu_y for x and y,
        respectively, and a common angular spread theta_sigma. Coordinates dE and dz can be sampled from the "Independent
        Distributions".
        For dE, dZ:
            _ : {'option' : 'fixed',    'param' : # }
            _ : {'option' : 'uniform',  'param' : {'min': #, 'max': #}
            _ : {'option' : 'stepped',  'param' : {'min': #, 'max': #, 'steps': #}
            _ : {'option' : 'normal',   'param' : {'mu': #, 'sigma': #}
            _ : {'option' : 'skewed',   'param' : {'alpha': #, 'mu': #, 'sigma': #}
        Example:
        initialDistribution = { 'type': 'aperture',
                                'nRays': 50,
                                'target_alignment' :   {"X": 0.03055/1000, "Y": -0.01099/1000, "R": 0.75/1000}, \
                                'aperture_alignment' : {"X": -0.448502/1000, "Y": -0.449370/1000, "R": 10.46/1000, 'separation_distance': 0.48619}, \
                                'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0/1000, 'sigma' : 10.24/1000},  \
                                'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}}, \
                                'dZ' : {'option': 'fixed',  'param': 0}, }
        '''
        nRays = initialDistribution['nRays']
        target_radius       = initialDistribution['target_alignment']['R']
        target_offset_x     = initialDistribution['target_alignment']['X']
        target_offset_y     = initialDistribution['target_alignment']['Y']
        aperture_radius     = initialDistribution['aperture_alignment']['R']
        aperture_offset_x   = initialDistribution['aperture_alignment']['X']
        aperture_offset_y   = initialDistribution['aperture_alignment']['Y']
        separation_distance = initialDistribution['aperture_alignment']['separation_distance']
        theta_mu_x          = initialDistribution['angles']['mu_x']
        theta_mu_y          = initialDistribution['angles']['mu_y']
        theta_sigma         = initialDistribution['angles']['sigma']
        dE_option = initialDistribution['dE']['option']
        dE_params = initialDistribution['dE']['param']
        dz_option = initialDistribution['dZ']['option']
        dz_params = { initialDistribution['dZ']['option'] : initialDistribution['dZ']['param'] }

        rays = aperture_rays(
            target_radius = target_radius, target_offset_x = target_offset_x, target_offset_y = target_offset_y,
            aperture_radius = aperture_radius, aperture_offset_x = aperture_offset_x, aperture_offset_y = aperture_offset_y, separation_distance = separation_distance,
            theta_mu_x = theta_mu_x, theta_mu_y = theta_mu_y, theta_sigma = theta_sigma,
            dE_option = dE_option, dE_params = dE_params, dz_option = dz_option, dz_params = dz_params,
            nRays = nRays )

    if initialDistribution['type'] == 'external':
        ray_file = 'target_rays.pkl'
        with open(ray_file, 'rb') as f:
            rays = pickle.load(f)
            # print(rays[0])

    return rays

# Function to check that positions and angles at the target are correct
def checkInitialDistribution(pepperpot = False):

    from matplotlib.patches import Circle
    with open(save_rays_file, 'rb') as f:
	    full_beam = pickle.load(f)
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices = \
        full_beam[0], full_beam[1], full_beam[2], full_beam[3], full_beam[4], full_beam[5], full_beam[6], full_beam[7]
    x  = np.array([ -xx[0]*1000 for i, xx  in enumerate(all_x)]) # in mm
    aX = np.array([ anX[0]*1000 for i, anX in enumerate(all_ax)]) # in mrad
    y  = np.array([  yy[0]*1000 for i, yy in enumerate(all_y)]) # in mm
    aY = np.array([ anY[0]*1000 for i, anY in enumerate(all_ay)]) # in mrad

	# Store list of all rays
    rad = []
    theta = []
    phi = []
	# Fixed seed
    targetX = np.mean(x)
    targetY = np.mean(y)
    for i in range( len(x) ):
        rad.append( 1.0* np.sqrt(x[i]-targetX)**2 + (y[i]-targetY)**2 )
        theta.append( np.sqrt( aX[i]**2 + aY[i]**2 ) )
        phi.append( 180/np.pi*np.arctan2(aY[i], aX[i]) )

    plt.rcParams.update({'font.size': 14})
    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(14, 5))
    axs[0,0].hist(rad, bins = 30)
    axs[0,0].set_xlabel('Target radius [mm]')
    axs[0,0].set_ylabel('Counts')
    axs[0,1].hist(theta, bins = 30)
    axs[0,1].set_xlabel('Theta [mrad]')
    axs[0,1].set_ylabel('Counts')
    axs[0,1].axvline(20.4, color='black', linestyle='--', linewidth=1)
    axs[0,1].axvline(23.0, color='black', linestyle='--', linewidth=1)
    axs[0,2].hist(phi, bins = 30)
    axs[0,2].set_xlabel('Phi [deg]')
    axs[0,2].set_ylabel('Counts')

    axs[1,0].hist2d(x, y, bins=(50, 50), cmap='viridis')
    axs[1,0].set_xlabel('X [mm]')
    axs[1,0].set_ylabel('Y [mm]')
    axs[1,0].set_title('Target')
    target_alignment = {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000}
    circle = Circle((-1000*target_alignment["X"], 1000*target_alignment["Y"]), 1000*target_alignment["R"], \
        fill=False, linewidth=2, edgecolor='white')
    axs[1,0].add_patch(circle)
	#axs[1,0].set_aspect('equal')

    axs[1,1].hist2d(aX, aY, bins=(50, 50), cmap='viridis')
    axs[1,1].set_xlabel('aX [mrad]')
    axs[1,1].set_ylabel('aY [mrad]')
    axs[1,1].set_title('Target')

    if pepperpot:
        separation_distance = 0.48619
        aperture_alignment = {"X": 0.311246954/1000, "Y": 8.5930652/1000, "R": 0.1/1000}

        xp = [x[i] - separation_distance*aX[i]  for i in range(len(x))]
        yp = [y[i] + separation_distance*aY[i]  for i in range(len(y))]
        axs[1,2].hist2d(xp, yp, bins=(50, 50), cmap='viridis')
        axs[1,2].set_xlabel('X [mm]')
        axs[1,2].set_ylabel('Y [mm]')
        axs[1,2].set_title('Pepperpot')
        circle = Circle((-1000*aperture_alignment["X"], 1000*aperture_alignment["Y"]), 1000*aperture_alignment["R"], \
            fill=False, linewidth=2, edgecolor='white')
        axs[1,2].add_patch(circle)
		#axs[1,2].set_aspect('equal')

	# COLUMN_NAMES = ["X", "counts"]
	# df = pd.read_csv("TransmissionData/20mrad_X_1Dgated.csv", skiprows = 1, header = None, sep=',', names = COLUMN_NAMES)
	# plt.plot(df["X"], df["counts"], color='red', linewidth=1)
    plt.tight_layout()
    plt.show()

###########################################
############ FOR RUNNING ECAT #############
###########################################

# Compute the percentage of rays only transmitted up to a given point
def compute_transmission(transmitted_x, n_positions):
    total_rays = len(transmitted_x)

    frequencies = [
        sum(1 for ray in transmitted_x if len(ray) == i + 1)
        for i in range(n_positions)
    ]
    percentages = [freq / total_rays * 100 for freq in frequencies]

    return percentages

# MAIN function to run ecat simulation
def runECAT(initialDistribution = None, run_cosy_flag = False, foxFile = fox_file, tuneChanges = None, save_rays_flag = False, final_index = element_names_stripped.index('DSSD') ):
    '''
    run_cosy_flag = False or True :  Flag to determine if COSY should be run and the matrices processed for the provided fox file
    save_rays_flag = True or False : Can optionally save ray coordinates to a pkl file for further analysis
    final_index  : Run ecat until this beamline element
    tuneChanges = [ ['Q1', 'Roll', -0.08], ['B1 Exit', 'dE', 0.01] ]
    '''

    # LOAD BEAMLINE GEOMETRY
    with open(beamline_geometry, 'rb') as f:
        full_beamline = pickle.load(f)
    # Determines the polygons at the evaluated positions along the beamline
    chamber_names, beampipes = find_polygons_at_z(full_beamline, z_coordinates)

    if tuneChanges != None:
        params, values, dipoleWFchanges = prepareTuneChanges(tuneChanges)
    else:
        params, values, dipoleWFchanges = None, None, None

    # DEFINE OPTICS FILE
    if run_cosy_flag:
        cosyChanges, cosyFile = prepareCOSYchanges(params, values)
        if len(cosyChanges) != 0:
            createCosyFile(cosyChanges, file_name = cosyFile)
            run_cosy_and_save_fort(f"./cosy {cosyFile}.fox", f"{cosyFile}")
            process_fort_files(f"{cosyFile}")
            df, grouped_data_frames = load_optics_data(cosyFile, pkl_directory)
            flagDelete = True
            foxFile = cosyFile
        else:
            run_cosy_and_save_fort(f"./cosy {foxFile}.fox", f"{foxFile}")
            process_fort_files(f"{foxFile}")
            # Unpack df and grouped_data_frames that contain the transformation matrices
            df, grouped_data_frames = load_optics_data(foxFile, pkl_directory)
    else:
        # Unpack df and grouped_data_frames that contain the transformation matrices
        df, grouped_data_frames = load_optics_data(foxFile, pkl_directory)

    # DEFINE INITIAL RAYS
    rays = generateInitialDistribution(initialDistribution)
    print("1. Generated initial rays.")

    # RAY PROPAGATION AND TRANSMISSION CHECKING
    # List to store transmitted coordinates for all rays
    #dipole_WF_Changes = element_names_stripped.index()
    # Vectorized tracer (fast_tracer.py), ray-by-ray identical to the Cython
    # process_all_rays + extract_coordinates incl. dB knobs (test_fast_tracer.py)
    # NOTE: Transmission_indices is a list of indices describing where each ray first failed to transmit
    # That is, for ray i, transmitted_x[i] = all_x[i][0:transmission_indices[i]]
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices = trace_rays(
        rays, grouped_data_frames[:final_index], beampipes[:final_index + 1], dipoleWFchanges)
    print("2. Processed all rays & checked transmission.")
    print("3. Extracted and sorted coordinate information.")

    # Calculate and print transmission
    # percentages = compute_transmission(transmitted_x, len(element_names_stripped))
    # rate = percentages[-1]
    # print("Final Transmission Rate: ", rate, "%")

    # Can optionally save ray coordinates
    if save_rays_flag:
        if run_cosy_flag and len(cosyChanges) != 0 :
            file = f'{results_directory}/{cosyFile}_Rays.pkl'
        else:
            file = f'{results_directory}/{foxFile}_Rays.pkl'
        with open(f"{file}", "wb") as f:
            pickle.dump((all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices), f)
        print("Data is saved")

    return all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, foxFile

###########################################
########## FOR PLOTTING PURPOSES ##########
###########################################

# Plot cross section of rays at a given z-position
def plot_cross_section(pos, beampipes, chamber_names, element_names, all_x, all_y, transmission_indices, z_coordinates, return_plt = True):

    # ---- figure/axes setup ----
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 1, height_ratios=[5, 2], hspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    # For handling the legend/labels
    used_labels = set()
    handles = []
    labels = []

    # Unpack and plot beampipe polygons
    if isinstance(beampipes[pos], list):
        poly_list = beampipes[pos]
    else:
        poly_list = [beampipes[pos]]

    all_x_pts = []
    all_y_pts = []

    for poly in poly_list:
        if poly is not None:
            poly = orient(poly, sign = 1.0)
            x, y = poly.exterior.xy
            x = [xi * 1000 for xi in x]
            y = [yi * 1000 for yi in y]

            all_x_pts.extend(x)
            all_y_pts.extend(y)

            label = chamber_names[pos]

            p, = ax.fill(x, y, alpha = 0.3, edgecolor = 'black',
                    facecolor = (110/255, 75/255, 160/255),
                    label = label)

            handles.append(p)
            labels.append(label)
            used_labels.add(label)

    for ray_x, ray_y, t_idx in zip(all_x, all_y, transmission_indices):
        x_pt = ray_x[pos] * 1000
        y_pt = ray_y[pos] * 1000
        all_x_pts.append(x_pt)
        all_y_pts.append(y_pt)

        if pos < t_idx:
            color, marker, label = 'seagreen', 'o', 'Transmitted'
        elif pos == t_idx:
            color, marker, label = 'mediumblue', 'X', 'First Failed Transmission'
        else:
            color, marker, label = 'black', 'X', 'Previously Failed Transmission'

        if label not in used_labels:
            p, = ax.plot(x_pt, y_pt, color=color, marker=marker,
                         markersize=4, linestyle='None', label=label)
            handles.append(p)
            labels.append(label)
            used_labels.add(label)
        else:
            ax.plot(x_pt, y_pt, color=color, marker=marker,
                    markersize=4, linestyle='None')

    # Set axis limits with margin
    if all_x_pts and all_y_pts:
        x_margin = (max(all_x_pts) - min(all_x_pts)) * 0.1 or 1
        y_margin = (max(all_y_pts) - min(all_y_pts)) * 0.1 or 1
        ax.set_xlim(min(all_x_pts) - x_margin, max(all_x_pts) + x_margin)
        ax.set_ylim(min(all_y_pts) - y_margin, max(all_y_pts) + y_margin)

    ax.set_title(f"z = {z_coordinates[pos]:.3f} | {element_names[pos]}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect('equal', adjustable='box')

    # Place legend in separate axis
    legend_ax.legend(handles, labels, loc='lower center', ncol=1, frameon=False)

    if return_plt:
        return plt.show()
    else:
        return fig

# Histogram to show the percentage of rays stopped at a given position
def transmission_histogram( transmitted_x, element_names, transmission_indices = None, title = None, focal_planes = None,  annotate = True, return_plt = True):
    """
    Plot transmission histogram showing where rays are stopped.
    Parameters:
        transmitted_x: list of lists, where each sublist contains positions a ray passed.
        element_names: full list of beamline segment names.
        transmission_indices: optional list of stopping indices per ray.
        title: title string for the plot.
        focal_planes: list of tuples (start_substring, end_substring, label) to shade regions.
        annotate: if True, adds % labels above bars.
    """
    n_positions = len(element_names)
    percentages = compute_transmission(transmitted_x, n_positions)

    fig, ax = plt.subplots(figsize=(15, 5))
    bars = ax.bar(range(n_positions), percentages, color='skyblue', edgecolor='black')

    ymax = ax.get_ylim()[1]
    offset_frac = 0.02

    if annotate:
        for i, (bar, percent) in enumerate(zip(bars, percentages)):
            if percent > 0:
                ax.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    bar.get_height() + offset_frac * ymax,
                    f"{percent:.2f}%",
                    ha='center', va='bottom',
                    rotation=90,
                )

    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(element_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.5, n_positions - 0.5)
    ax.yaxis.set_major_formatter(PercentFormatter())

    ax.set_xlabel('Final Transmission Position for Rays')
    ax.set_ylabel('Terminated Rays (%)')

    if title is not None:
        ax.set_title(title, fontweight='bold')
    else:
        ax.set_title(f'Transmission Analysis for {len(transmitted_x)} Rays')

    # Handle focal plane highlighting if provided
    if focal_planes:
        fig_height_inches = fig.get_size_inches()[1]
        y_offset_axes = 1.0 - (0.25 / fig_height_inches)
        for start_text, end_text, label in focal_planes:
            try:
                start_idx = next(i for i, name in enumerate(element_names) if start_text in name)
                end_idx = next(i for i, name in enumerate(element_names) if end_text in name)
                ax.axvspan(start_idx - 0.5, end_idx + 0.5, color='grey', alpha=0.2, label=label)
                ax.text(start_idx - 1, y_offset_axes, label,
                        transform=ax.get_xaxis_transform(),
                        verticalalignment='center',
                        horizontalalignment='right',
                        fontweight='bold', color='black')
            except StopIteration:
                print(f"Warning: Could not locate focal plane region for '{label}'.")

    ymax = max(percentages) * 1.25 + 1
    ax.set_ylim(0, ymax)
    fig.tight_layout()

    if return_plt:
        return plt.show()
    else:
        return fig

# Plots x-vs-z and y-vs-z to trace rays along the beamline
def plot_rays(x_list, y_list, z_coordinates, element_names = None, highlight_focal_planes = True, return_plt = True):

    fig, axs = plt.subplots(2, figsize = (12, 8))
    fig.tight_layout(pad=4)

    fp_color = (184 / 255, 140 / 255, 219 / 255)  # consistent purple

    # Plot x trajectories
    for x in x_list:
        axs[0].plot(z_coordinates[:len(x)], x, color='black', linewidth=1)
    axs[0].set_xlabel("z-Position Along Beamline (m)")
    axs[0].set_ylabel("x-Position (m)")
    axs[0].set_xlim(0, z_coordinates[-1])
    axs[0].axhline(y=0, color='grey', linestyle="--", linewidth=1)

    # Plot y trajectories
    for y in y_list:
        axs[1].plot(z_coordinates[:len(y)], y, color='black', linewidth=1)
    axs[1].set_xlabel("z-Position Along Beamline (m)")
    axs[1].set_ylabel("y-Position (m)")
    axs[1].set_xlim(0, z_coordinates[-1])
    axs[1].axhline(y=0, color='grey', linestyle="--", linewidth=1)

    # Optionally highlight focal plane regions
    if highlight_focal_planes and element_names is not None:
        # More general substring search for flexible naming
        focal_planes = {
            "FP 1": ("FP1 Slits", "VD1542"),
            "FP 2": ("FP2 Slits", "VD1638"),
            "FP 3": ("FP3 Slits", "VD1783"),
            "FP 4": ("FP4 Slits", "DSSD"),
        }

        for ax in axs:
            for label, (start_key, end_key) in focal_planes.items():

                # Find first matching elements by substring
                try:
                    start_idx = next(i for i, name in enumerate(element_names) if start_key in name)
                    end_idx = next(i for i, name in enumerate(element_names) if end_key in name)
                except StopIteration:
                    print(f"Warning: Could not find '{start_key}' or '{end_key}' in element_names for {label}")
                    continue

                z_start = z_coordinates[start_idx]
                z_end = z_coordinates[end_idx]

                ax.axvspan(z_start, z_end, color=fp_color, alpha=0.2)

                # Annotate focal plane
                y_offset = 0.95 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
                ax.text(z_start - 0.02 * (z_coordinates[-1]), y_offset, label,
                        verticalalignment='top', horizontalalignment='right',
                        fontweight='bold', color='black', fontsize=10, alpha=0.7)

    plt.suptitle("Ray Tracing Along Beamline")

    if return_plt:
        return plt.show()
    else:
        return fig

def plot_rays_colored(x_list, y_list, z_coordinates, color_dict = None, element_names = None, highlight_focal_planes = True, return_plt = True):
    fig, axs = plt.subplots(2, figsize=(12, 8))
    fig.tight_layout(pad=4)

    fp_color = (184 / 255, 140 / 255, 219 / 255)  # consistent purple

    legend_handles = []
    legend_info = []

    n_rays = len(x_list)

    if color_dict is None:
        ray_colors = ['black'] * n_rays
    else:
        coords = np.asarray(color_dict['color_list'])   # one value per ray
        n_bins = color_dict['n_bins']
        color_key = color_dict['color_key']

        if len(coords) != n_rays:
            raise ValueError(
                "color_dict['color_list'] must have one entry per ray."
            )

        min_val = np.min(coords)
        max_val = np.max(coords)

        if min_val == max_val:
            groups = np.zeros(len(coords), dtype=int)
            bounds = np.array([min_val, max_val])
            cmap = matplotlib.colormaps.get_cmap('viridis').resampled(max(n_bins, 1))
            ray_colors = [cmap(0) for _ in groups]

            label = f"{color_key}: {min_val:.3g}"
            legend_handles = [Line2D([0], [0], color=cmap(0), lw=2, label=label)]
            legend_info = [{
                "bin": 0,
                "color": cmap(0),
                "label": label,
                "lower": min_val,
                "upper": max_val
            }]
        else:
            bounds = np.linspace(min_val, max_val, n_bins + 1)

            # Bin index for each ray: 0, 1, ..., n_bins-1
            groups = np.digitize(coords, bounds[1:-1], right=False)

            cmap = matplotlib.colormaps.get_cmap('viridis').resampled(n_bins)
            ray_colors = [cmap(g) for g in groups]

            # Build legend entries from bin ranges
            for i in range(n_bins):
                lower = bounds[i]
                upper = bounds[i + 1]

                if i < n_bins - 1:
                    label = f"{color_key}: [{lower:.3g}, {upper:.3g})"
                else:
                    label = f"{color_key}: [{lower:.3g}, {upper:.3g}]"

                color = cmap(i)

                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2, label=label)
                )
                legend_info.append({
                    "bin": i,
                    "color": color,
                    "label": label,
                    "lower": lower,
                    "upper": upper
                })

    # Plot x trajectories
    for i, x in enumerate(x_list):
        axs[0].plot(z_coordinates[:len(x)], x, color=ray_colors[i], linewidth=1)

    axs[0].set_xlabel("z-Position Along Beamline (m)")
    axs[0].set_ylabel("x-Position (m)")
    axs[0].set_xlim(0, z_coordinates[-1])
    axs[0].axhline(y=0, color='grey', linestyle="--", linewidth=1)

    # Plot y trajectories
    for i, y in enumerate(y_list):
        axs[1].plot(z_coordinates[:len(y)], y, color=ray_colors[i], linewidth=1)

    axs[1].set_xlabel("z-Position Along Beamline (m)")
    axs[1].set_ylabel("y-Position (m)")
    axs[1].set_xlim(0, z_coordinates[-1])
    axs[1].axhline(y=0, color='grey', linestyle="--", linewidth=1)

    # Optionally highlight focal plane regions
    if highlight_focal_planes and element_names is not None:
        focal_planes = {
            "FP 1": ("FP1 Slits", "VD1542"),
            "FP 2": ("FP2 Slits", "VD1638"),
            "FP 3": ("FP3 Slits", "VD1783"),
            "FP 4": ("FP4 Slits", "DSSD"),
        }

        for ax in axs:
            for label, (start_key, end_key) in focal_planes.items():
                try:
                    start_idx = next(i for i, name in enumerate(element_names) if start_key in name)
                    end_idx = next(i for i, name in enumerate(element_names) if end_key in name)
                except StopIteration:
                    print(f"Warning: Could not find '{start_key}' or '{end_key}' in element_names for {label}")
                    continue

                z_start = z_coordinates[start_idx]
                z_end = z_coordinates[end_idx]

                ax.axvspan(z_start, z_end, color=fp_color, alpha=0.2)

                y0, y1 = ax.get_ylim()
                y_offset = y0 + 0.95 * (y1 - y0)
                ax.text(
                    z_start - 0.02 * z_coordinates[-1],
                    y_offset,
                    label,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontweight='bold',
                    color='black',
                    fontsize=10,
                    alpha=0.7
                )

    # Add color legend if applicable
    # if legend_handles:
    #     axs[0].legend(handles=legend_handles, loc='best', title="Ray color groups")

    fig.suptitle("Ray Tracing Along Beamline")

    for ax in axs:
        ax.set_ylim(-0.1, 0.1)
    if return_plt:
        plt.show()
        return legend_info
    else:
        return fig, legend_info

# Function for plotting beam positions using 2D histograms
def plotInterestingCS(displaySec, section, all_x, all_y, transmission_indices, saveFile = None):

    nuRow = 3
    nuCol = 4
    fig, axs = plt.subplots(nuRow, nuCol, figsize=(17,9))
    plt.suptitle("Viewing with the beam", fontsize=16)

    pat = re.compile(r'(?<!\d)(\d{4})(?!\d)')  # exactly 4 digits, not part of a longer number
    key = [m.group(1) if (m := pat.search(s)) else 0 for s in displaySec]

    for j, sec in enumerate(displaySec):
        pos = next((i for i, x in enumerate(section) if x == sec), None)
        x = np.array([-xx[pos]*1000 for i, xx in enumerate(all_x) if transmission_indices[i] >= pos]) # in mm
        y = np.array([ yy[pos]*1000 for i, yy in enumerate(all_y) if transmission_indices[i] >= pos]) # in mm
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        if sec == 'Target Center':
            axs[int((j)/nuCol), (j)%nuCol].hist2d(x, y, bins=(50, 50), cmap='viridis',range=[[-2, 2], [-2, 2]])
        else:
            FC_flag = False
            VD_flag = False
            if key[j] in faraday_cups:
                FC_flag = True
                FC_Xmin = -faraday_cups[key[j]]['size_mm'][0]/2 - faraday_cups[key[j]]['misalignment'][0]
                FC_Xmax =  faraday_cups[key[j]]['size_mm'][0]/2 - faraday_cups[key[j]]['misalignment'][0]
                FC_Ymin = -faraday_cups[key[j]]['size_mm'][1]/2 - faraday_cups[key[j]]['misalignment'][1]
                FC_Ymax =  faraday_cups[key[j]]['size_mm'][1]/2 - faraday_cups[key[j]]['misalignment'][1]
                xmin = np.min([xmin, FC_Xmin])
                xmax = np.max([xmax, FC_Xmax])
                ymin = np.min([ymin, FC_Ymin])
                ymax = np.max([ymax, FC_Ymax])
            if key[j] in viewers:
                VD_flag = True
                VD_Xmin = -viewers[key[j]]['size_mm'][0]/2 - viewers[key[j]]['misalignment'][0]
                VD_Xmax =  viewers[key[j]]['size_mm'][0]/2 - viewers[key[j]]['misalignment'][0]
                VD_Ymin = -viewers[key[j]]['size_mm'][1]/2 - viewers[key[j]]['misalignment'][1]
                VD_Ymax =  viewers[key[j]]['size_mm'][1]/2 - viewers[key[j]]['misalignment'][1]
                xmin = np.min([xmin, VD_Xmin])
                xmax = np.max([xmax, VD_Xmax])
                ymin = np.min([ymin, VD_Ymin])
                ymax = np.max([ymax, VD_Ymax])

            axs[int((j)/nuCol), (j)%nuCol].hist2d(x, y, bins=(200, 200), cmap='viridis',range=[[1.2*xmin, 1.2*xmax],[1.2*ymin, 1.2*ymax]])
            if FC_flag:
                axs[int((j)/nuCol), (j)%nuCol].plot([FC_Xmin,FC_Xmax],[FC_Ymax,FC_Ymax], c = 'white', linestyle='dashed')
                axs[int((j)/nuCol), (j)%nuCol].plot([FC_Xmin,FC_Xmax],[FC_Ymin,FC_Ymin], c = 'white', linestyle='dashed')
                axs[int((j)/nuCol), (j)%nuCol].plot([FC_Xmin,FC_Xmin],[FC_Ymin,FC_Ymax], c = 'white', linestyle='dashed')
                axs[int((j)/nuCol), (j)%nuCol].plot([FC_Xmax,FC_Xmax],[FC_Ymin,FC_Ymax], c = 'white', linestyle='dashed')
            if VD_flag:
                axs[int((j)/nuCol), (j)%nuCol].plot([VD_Xmin,VD_Xmax],[VD_Ymax,VD_Ymax], c = 'white')
                axs[int((j)/nuCol), (j)%nuCol].plot([VD_Xmin,VD_Xmax],[VD_Ymin,VD_Ymin], c = 'white')
                axs[int((j)/nuCol), (j)%nuCol].plot([VD_Xmin,VD_Xmin],[VD_Ymin,VD_Ymax], c = 'white')
                axs[int((j)/nuCol), (j)%nuCol].plot([VD_Xmax,VD_Xmax],[VD_Ymin,VD_Ymax], c = 'white')
            axs[int((j)/nuCol), (j)%nuCol].text(0.7*xmin, 1.13*ymin, 'X = ' + str(np.round(np.mean(x),1)) + '+-' + str(np.round(np.std(x),2)), fontsize=12, color='yellow')
            axs[int((j)/nuCol), (j)%nuCol].text(0.7*xmin, 1.05*ymax, 'Y = ' + str(np.round(np.mean(y),1)) + '+-' + str(np.round(np.std(y),1)), fontsize=12, color='yellow')
        axs[int((j)/nuCol), (j)%nuCol].set_xlabel('X [mm]')
        axs[int((j)/nuCol), (j)%nuCol].set_ylabel('Y [mm]')
        axs[int((j)/nuCol), (j)%nuCol].set_title( sec )
    plt.tight_layout()
    if saveFile != None:
        plt.savefig(saveFile)
    plt.show()
