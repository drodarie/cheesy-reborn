# -*- coding: utf-8 -*-
"""
Whole brain loading.
"""
# pragma: no cover

__author__ = 'Bobby'


import os
import sys


#~ import hbp_nrp_cle.tf_framework as nrp
import logging
#~ from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import h5py
import nest

logger = logging.getLogger(__name__)


h5_file_path = os.path.join(os.environ.get('HBP'),   'Models/cheesy_reborn_model/H5FILES/ptneu_brain.h5')
sys.path.append( os.path.join(os.environ.get('HBP'), 'Models/cheesy_reborn_model/') )

nest.ResetKernel()
nest.sr("M_WARNING setverbosity")
nest.SetKernelStatus({"local_num_threads": 6})

from loadPtNeuNetwork import *
#~ circuit = load_pointneuron_circuit(h5_file_path, synTypes = [], neuroIDs = np.array([1,2]), returnParas = ["IO_ids", "IO_x", "IO_y"] )
#~ circuit = load_pointneuron_circuit(h5_file_path, synTypes = [], neuroIDs = None, returnParas = ["IO_ids", "IO_x", "IO_y"] )
circuit = load_pointneuron_circuit(h5_file_path, synTypes = ["KERN"], neuroIDs = None, returnParas = ["IO_ids", "IO_x", "IO_y"] )


# create spike detector
circuit["rec"] = nest.Create('spike_detector', params={"withgid": True, "withtime": True, "to_file": False})
nest.Connect(circuit["cells"], circuit["rec"])

#~ nest.SetStatus( circuit["cells"], "Delta_T", 0.1 )
#~ print nest.GetStatus( circuit["cells"], "a" )
#~ print nest.GetStatus( circuit["cells"], "b" )
#~ print nest.GetStatus( circuit["cells"], "Delta_T" )
