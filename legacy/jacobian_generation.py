"""Legacy Jacobian-generation prototype retained for historical reference."""

# Script for generating Jacobians. needs to be run twice for each dataset. Needs segmented mesh, wavelength to simulate, an experimental file to define which channels are used
#a probe file and its coordinate orientation

import pmmc
import numpy as np
import matplotlib.pyplot as plt
import iso2mesh as i2m
import json
from scipy.io import loadmat
import mne
import mne_nirs
import os, sys
import trimesh
from scipy.spatial import Delaunay
from scipy.io import loadmat, savemat
import subprocess

from mmc_nirs.registration import register_probe
from mmc_nirs.utils.jacobian_utils import mmc_to_json, read_cli_output
from mmc_nirs.utils.mesh_utils import find_closest_node

#the mesh file you will need to change
meshfile = loadmat('../mat_files/NewColinMesh.mat')
meshnodes = meshfile['newnodes']
meshelem = meshfile['elem']
scalp_idx = meshelem[:,-1]==5

#ScatterBrains Mesh
# to_ras = make_to_ras_dict()
# mesh_transformation_matrix = to_ras['RIA']
# meshfile = loadmat('../mat_files/scatterBrains-main/Subject03/Subject03_mesh.mat')
# meshnodes = meshfile['node'][:,:3] @ mesh_transformation_matrix
# meshelem = meshfile['elem']
# scalp_idx = meshelem[:,-1]==1

#wavelength you are running simulations for [either 690 or 830]
wavelength = 850

#output file name (you will need to change this)
#output_file_name = 'PainJ830.mat'
#output_file_name = 'SB_03_J830.mat'
output_file_name = 'YY_J850.mat'

#experimental file just for getting channel pairings
#exp_file_name = "./Example_Data_File/FingerTapping.snirf"
exp_file_name = "./Example_Data_File/NIRS-2019-08-10_006.snirf"
exp_raw = mne.io.read_raw_snirf(exp_file_name)


#Loading fNIRS optode montage into 3D coordinates
# sdfile = loadmat('./Example_Data_File/probe.SD')
# src_pts = sdfile['SD'][0][0][1]
# det_pts = sdfile['SD'][0][0][2]
# probeorientation = 'LIA'


#ignore, this is for a different dataset
probefile = loadmat('./Example_Data_File/YYprobe.mat')
#meshnodes = meshfile['newnodes']
#meshelem = meshfile['elem']
src_pts = probefile['sourcepos']
det_pts = probefile['detpos']
probeorientation = 'LIA'

#registering the probe
#TODO return dictionary of registered probe
reg_src_pos, reg_det_pos, src_dir, det_dir, e0_src, e0_det = register_probe(src_pts,det_pts,meshnodes[:,:3], meshelem[:,:4]-1, scalp_idx, probe_orientation=probeorientation)

#moving sources inward by 1 mm
mmc_src_pts = reg_src_pos# + src_dir



#Defining optical properties for each wavelength
#TODO import optical properties in json
#TODO make it so different tissue layers can be permuted according with mesh labels
prop = {}
prop['690'] =   [[0,0,1,1],                 #ambient air
                [0.07,60.0,0.9,1.37],       #white matter
                [0.02,8.8,0.9,1.37],        #gray matter
                [0.0004,0.01,0.9,1.37],     #CSF
                [0.0101,1.0,0.9,1.37],      #skull
                [0.0159,0.8,0.9,1.37]]      #scalp

prop['830'] =   [[0,0,1,1],                 #ambient air
                [0.09,42.9,0.9,1.37],       #white matter
                [0.03,7.0,0.9,1.37],        #gray matter
                [0.0026,0.01,0.9,1.37],     #CSF
                [0.0136,0.86,0.9,1.37],      #skull
                [0.0191,0.66,0.9,1.37]]      #scalp

prop['760'] =   [[0,0,1,1],                 #ambient air
                [0.08,40.0,0.9,1.37],       #white matter
                [0.02,8.8,0.9,1.37],        #gray matter
                [0.0021,0.01,0.9,1.37],     #CSF
                [0.0125,0.93,0.9,1.37],      #skull
                [0.022,0.82,0.9,1.37]]      #scalp

prop['850'] =   [[0,0,1,1],                 #ambient air
                [0.08,34.2,0.88,1.37],       #white matter
                [0.035,7.5,0.9,1.37],        #gray matter
                [0.0026,0.01,0.9,1.37],     #CSF
                [0.0136,0.86,0.9,1.37],      #skull
                [0.022,0.75,0.9,1.37]]      #scalp

# prop['690'] = [
#             [0,0,1,1],              # ambient air
#             [0.0159,0.8,0.9,1.37],  # scalp
#             [0.0101,1.0,0.9,1.37],  # skull
#             [0.0004,0.01,0.9,1.37], # CSF
#             [0.02,8.8,0.9,1.37],    # gray matter
#             [0.07,60.0,0.9,1.37]    # white matter
#         ]

# prop['830'] = [
#         [0,0,1,1],              # ambient air
#         [0.0191,0.66,0.9,1.37], # scalp
#         [0.0136,0.86,0.9,1.37], # skull
#         [0.0026,0.01,0.9,1.37], # CSF
#         [0.03,7.0,0.9,1.37],    # gray matter
#         [0.09,42.9,0.9,1.37]    # white matter
#     ]

# prop['760'] = [
#         [0,0,1,1],              # ambient air
#         [0.022,0.82,0.9,1.37],  # scalp
#         [0.0125,0.93,0.9,1.37], # skull
#         [0.0021,0.01,0.9,1.37], # CSF
#         [0.02,8.8,0.9,1.37],    # gray matter
#         [0.08,40.0,0.9,1.37]    # white matter
#     ]

# prop['850'] = [
#         [0,0,1,1],              # ambient air
#         [0.022,0.75,0.9,1.37],  # scalp
#         [0.0136,0.86,0.9,1.37], # skull
#         [0.0026,0.01,0.9,1.37], # CSF
#         [0.035,7.5,0.9,1.37],   # gray matter
#         [0.08,34.2,0.88,1.37]   # white matter
#     ]


#Defining the generic input config to change as needed
cfg = []
cfg = {
    'nphoton' : 5e9,
    'node' : meshnodes[:,:3].tolist(),
    'elem' : meshelem[:,:4].tolist(),
    'elemprop' : meshelem[:,-1].tolist(),
    'tstart' : 0,
    'tend' : 5e-9,
    'tstep' : 5e-9,
    'srcpos' : mmc_src_pts.tolist(),
    'e0' : e0_src,
    'srcdir' : src_dir.tolist(),
    'prop' : prop,
    'method' : 'elem',
    'issaveexit' : 1,
    'issavedet' : 1,
    'outputtype' : 'flux',
}

#defining the number of sources and detectors
nsrc = src_pts.shape[0]
ndet = det_pts.shape[0]

#finding the closest node to each detector
closest_node_idx = []
for det_idx in range(ndet):
    closest_node, _ = find_closest_node(meshnodes[:,:3],reg_det_pos[det_idx,:])
    closest_node_idx.append(closest_node)

#defining the detector radius (and moving it to the closest node location as a bit of a hack to prevent empty detectors)
detradius = np.ones_like(reg_det_pos[:,0][:,np.newaxis])
#mmc_det_pts = np.hstack([meshnodes[closest_node_idx,:3] - det_dir, detradius])
mmc_det_pts = np.hstack([reg_det_pos, detradius])

#running MMC simulations for sources
srccfg = cfg.copy()
srccfg['prop'] = cfg['prop'][str(wavelength)]
srccfg['detpos'] = mmc_det_pts.tolist()

mea0 = np.zeros([nsrc * ndet, 1])
Green_s = np.zeros([nsrc,len(cfg['node'])])
Green_sd = np.zeros([nsrc * ndet, 1])

for src_idx in range(0,nsrc):

    #specify which source we are running the simulation for
    srccfg['srcpos'] = cfg['srcpos'][src_idx]
    srccfg['e0'] = int(e0_src[src_idx]+1)
    srccfg['srcdir'] = cfg['srcdir'][src_idx]

    #creating the config file to run from command line
    stub = 'singlesource'
    mmc_to_json(srccfg,stub + '.json')

    while not os.path.isfile('singlesource.dat'):
        try:
            src_output = subprocess.run(['../mmc/bin/mmc.exe',
                            '-f' , 'singlesource.json','-d', '1']
                            ,timeout=900,capture_output=True)
        except:
            print(src_output.stderr)
            print('Timed out on source number ' + str(src_idx) +  ', trying again')

    flux, detp = read_cli_output(stub)
    print('succesfully ran source number ' + str(src_idx))
    os.remove(stub + '.dat')
    os.remove(stub + '.mch')

    #calculating Green's function for source and source-detector pairs
    Green_s[src_idx,:] = np.transpose(flux) * srccfg['tstep']

    w0 = pmmc.detweight(detp,np.array(srccfg['prop']))
    for det_idx in range(ndet):
        #indexing over a flattened source-detector matrix
        row = src_idx * ndet + det_idx
        #summing weights of all photons at each detector
        mea0[row] = w0[detp['detid']==det_idx].sum()
        det_elem_idx = meshfile['elem'][(int(e0_det[det_idx])),:4]
        #det_flux = flux[det_elem_idx]
        Green_sd[row] = flux[closest_node_idx[det_idx]] * cfg['tstep'] #det_flux.mean() * cfg['tstep']#



#running mmc simulations for detectors
detcfg = {}
detcfg = cfg.copy()
detcfg['prop'] = cfg['prop'][str(wavelength)]
#del detcfg['detpos']
Green_d = np.zeros([ndet,len(cfg['node'])])


for det_idx in range(ndet):

    #specify which detector we are running the simulation for
    detcfg['srcpos'] = srccfg['detpos'][det_idx][:3]
    detcfg['e0'] = int(e0_det[det_idx]+1)
    detcfg['srcdir'] = det_dir[det_idx,:].tolist()

    #creating the config file to run from command line
    stub = 'singledetector'
    mmc_to_json(detcfg,stub + '.json')
    while not os.path.isfile('singledetector.dat'):
        try:
            det_output = subprocess.run(['../mmc/bin/mmc.exe',
                            '-f' , 'singledetector.json',
                            '-d', '1'],timeout=900,capture_output=True)
        except:
            print('Timed out on detector number ' + str(det_idx) +  ', trying again')

    flux = read_cli_output(stub)
    print('succesfully ran detector number ' + str(det_idx))
    os.remove(stub + '.dat')
    #os.remove(stub + '.mch')

    #calculating Green's function for detectors
    Green_d[det_idx,:] = np.transpose(flux) * detcfg['tstep']

#calculating Jacobian from simulation outputs
J = np.zeros([nsrc * ndet, len(cfg['node'])])
for src_idx in range(nsrc):
    for det_idx in range(ndet):
        row = src_idx*ndet + det_idx
        J[row,:] = Green_s[src_idx,:] * Green_d[det_idx,:]/Green_sd[row]

#Find relevant channels
exp_raw = mne.io.read_raw_snirf(exp_file_name)
ch_names = exp_raw.info.ch_names
ch_names[0]

SDarray = np.zeros([nsrc,ndet])

for chan_idx in ch_names:

    #finding the source and detectors from the mne raw object's channel names
    channel_source = int(chan_idx[chan_idx.find('S')+1:chan_idx.find('_')])
    channel_detector = int(chan_idx[chan_idx.find('D')+1:chan_idx.find(' ')])
    SDarray[channel_source-1,channel_detector-1] = 1
SDvector = SDarray.reshape([1,-1])
#creating indexes for each source-detector pair
channelidx = np.where(SDvector)[1] + 1


#Saving as Dict
J_dict = {
    'Green_d' : Green_d,
    'Green_s' : Green_s,
    'Green_sd' : Green_sd,
    'J' : J,
    'channelidx' : channelidx,
    'mea0' : mea0,
    'sourcepos' : mmc_src_pts,
    'detpos' : mmc_det_pts,
    'detnorms' : det_dir,
    'sourcedir' : src_dir
}

savemat(output_file_name,J_dict)
