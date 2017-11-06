#~ "tsodyks2_synapse"


# loading the h5 format
def load_pointneuron_circuit(h5_filename, synTypes = [], returnParas = [], randomize_neurons = False, neuroIDs = None):
    import h5py
    import numpy as np
    import nest
    circuit = {}
    print("Opening h5 datafile "+'\033[96m'+"\""+h5_filename+"\""+'\033[0m'+" ... ")
    h5file = h5py.File(h5_filename,'r')
    h5fileKeys = h5file.keys()
    # get sub ids
    Nh5 = h5file["x"][:].shape[0]
    if neuroIDs==None:
        N = Nh5
        neuroIDs_ = np.array(range(N))
        translatorDict = np.array(range(N))
    else:
        N = neuroIDs.shape[0]
        print("Sorting new ID list")
        neuroIDs_ = np.sort(neuroIDs)
        print("OK")
        translatorDict = -np.ones(Nh5, np.int32)
        translatorDict[ neuroIDs_ ] = np.array(range(N))
    circuit["neuroIDs"] = neuroIDs_
    circuit["translatorDict"] = translatorDict
    # return custom keys
    for rP in returnParas:
        circuit[rP] = h5file[rP][:][neuroIDs_]
    # subgroup location for synapses
    synapse_dataset_location = h5file["synapse_dataset_location"][:]
    # retrieving neuronal and synaptic model if it exists 
    if "neuronModel" in h5fileKeys: 
        neuronModel = h5file["neuronModel"].value.decode('UTF-8')
    else:
        neuronModel = "aeif_cond_exp"
    circuit["synapseModel"] = h5file["synapseModel"][:][neuroIDs_]
    # creating neurons
    print neuronModel
    cells = nest.Create(neuronModel, N)
    circuit["cells"] = cells
    neuroParams = h5file["neuroParams"].keys()
    for nP in neuroParams:
        print("Setting neural parameter "+nP+" ...")
        nest.SetStatus(cells, str(nP), np.float64(h5file["neuroParams"][nP][:][neuroIDs_]))
    
    if randomize_neurons:
        nest.SetStatus(cells, "V_m", -80.0+20.0*np.random.rand(len(cells)))
    
    #~ Ltype = h5file["cellTypes"][:]
    #~ Lname = h5file["cellTypesToName"][:]
    #~ simulatedTypePrkL = np.where( Lname=="PrkL" )[0][0]
    #~ simulatedTypeDCNe = np.where( Lname=="DCNe" )[0][0]
    #~ circuit["SCPrkL"] = []
    #~ circuit["SCIO"]   = []
    
    totNumSyns = 0
    for sT in synTypes:
        print(sT, "synapse creation:")
        for i,gid in enumerate(neuroIDs_):
            sType = circuit["synapseModel"][i]
            if i%1000 == 0:
                progressbar(float(i)/float(N), 60)
            
            for dir_ in ["IN","OUT"]:
                hasSyns = True
                try:
                    #~ print "syngroup_"+str(synapse_dataset_location[gid]), "syn_"+sT+"_"+dir_+"_T_" +str(gid)
                    synT_old = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_"+dir_+"_T_" +str(gid)][:]
                    synT     = translatorDict[ synT_old ]
                except:
                    hasSyns = False
                if hasSyns:
                    syn  = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_"+dir_+"_"+str(gid)][:,:]
                    # only take synapses that target the current circuit
                    validSynapses = np.where(synT!=-1)[0]
                    synT = synT[ validSynapses ]
                    syn  = syn[:,validSynapses ]
                    totNumSyns += synT.shape[0]
                    
                    #~ print Lname[Ltype[gid]], dir_, synT.shape[0], np.mean(syn[1,:])
                    #~ if Ltype[gid]==simulatedType: print synT, syn[1,:], synT.shape[0]

                    synT = np.float64( np.int64(  synT[:])+1 )
                    syn  = np.float64( syn[:])
                    wasTypeSwitched = 0
                    if type(synT)==np.float64:
                        #~ print "typeswitch"
                        wasTypeSwitched = 1
                        #~ print syn, synT
                        #~ print syn.shape, synT.shape
                        synT = np.array([synT]); 
                        syn  = np.array(syn)
                        #~ print syn, synT
                        #~ print syn.shape, synT.shape

                    if dir_=="IN":
                        #~ if Ltype[gid] == simulatedTypeDCNe:
                        #~ print Ltype[np.int64(synT_old)], syn[1,:]
                        #~ LtypesTMP = Ltype[ neuroIDs_[ np.int64(synT)-1 ] ]
                        #~ print wasTypeSwitched, synT.shape, LtypesTMP, syn[1,:]
                        #~ if np.max(syn[1,LtypesTMP==31])>-0.005+0.001: 
                            #~ print("OH NOES!!!")
                            #~ print syn[1,LtypesTMP==31], LtypesTMP, syn[1,:]
                        #~ circuit["SCPrkL"].append( np.int64(synT)[ Ltype[ neuroIDs_[ np.int64(synT)-1 ] ]==31 ] )
                        #~ circuit["SCIO"  ].append( np.int64(synT)[ Ltype[ neuroIDs_[ np.int64(synT)-1 ] ]==16 ] )
                        #~ circuit["SCIO"  ].append( np.int64(synT)[ syn[1,:]>0.0 ] )
                        
                        #~ syn[ 1,syn[1,:]>0.0 ] = 0.0
                        #~ syn[ 1,syn[1,:]<0.0 ] = 0.0
                        
                        if sType==0:
                            nest.DataConnect(list( np.int64(synT) ), [{
                            'target':  float(i+1)*np.ones(synT.shape[0], np.float64),
                            'delay': syn[0,:], 'weight': syn[1,:], 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]
                            }], "tsodyks2_synapse")
                        else:
                            nest.Connect(list( np.int64(synT) ),list((i+1)*np.ones(synT.shape[0], np.int64)), {'rule': 'one_to_one'}, {"model": "static_synapse", "weight": syn[1,:], "delay": syn[0,:]})
                        
                        #~ nest.Connect(list( np.int64(synT) ),list((i+1)*np.ones(synT.shape[0], np.int64)), {'rule': 'one_to_one'}, {"model": "static_synapse", "weight": syn[1,:], "delay": syn[0,:]})
                        
                        #~ for isynT_,synT_ in enumerate(synT):
                            #~ #if syn[1,isynT_]>0.0: circuit["SCIO"  ].append( int(synT_) )
                            #~ nest.DataConnect( [int(synT_)], [{
                            #~ 'target':  float(i+1)*np.ones(1, np.float64),
                            #~ #'delay': syn[0,isynT_:isynT_+1], 'weight': syn[1,isynT_:isynT_+1], 'U': syn[2,isynT_:isynT_+1], 'tau_rec': syn[3,isynT_:isynT_+1], 'tau_fac': syn[4,isynT_:isynT_+1]
                            #~ 'delay': syn[0,isynT_:isynT_+1], 'weight': syn[1,isynT_:isynT_+1]#, 'U': syn[2,isynT_:isynT_+1], 'tau_rec': syn[3,isynT_:isynT_+1], 'tau_fac': syn[4,isynT_:isynT_+1]
                            #~ #}], synapseModel)
                            #~ }], "static_synapse")
                    elif dir_=="OUT":
                        if sType==0:
                            nest.DataConnect([i+1], [{
                            'target': synT,
                            #~ 'delay': syn[0,:], 'weight': (1.0+10.0*np.float64(sT=="KERN"))*syn[1,:], 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]                        
                            #~ 'delay': syn[0,:], 'weight': syn[1,:], 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]
                            'delay': syn[0,:], 'weight': syn[1,:], 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]
                            #~ 'delay': syn[0,:], 'weight': 0.2*syn[1,:], 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]
                            #~ 'delay': syn[0,:], 'weight': syn[1,:]#, 'U': syn[2,:], 'tau_rec': syn[3,:], 'tau_fac': syn[4,:]
                            }], "tsodyks2_synapse")
                            #~ }], "static_synapse")
                        else:
                            if synT.shape[0]>1: targ_ = list( np.int64(synT) )
                            else:               targ_ = [int(synT[0])]
                            nest.Connect(list((i+1)*np.ones(synT.shape[0], np.int64)), targ_ , {'rule': 'one_to_one'}, {"model": "static_synapse", "weight": syn[1,:], "delay": syn[0,:]})
                            #~ nest.DataConnect([i+1], [{ 'target': synT, 'delay': syn[0,:], 'weight': syn[1,:] }], "static_synapse")
        progressbar(1.0, 60)
    #nest.SetStatus(cells, 'tau_syn_ex', 1.8); nest.SetStatus(cells, 'tau_syn_in', 8.0)
    print("Total number of synapses:", totNumSyns)
    h5file.close()
    return circuit








# colorful text (yay!)
color_dictionary = {}
color_dictionary['white']  = '\033[97m'
color_dictionary['blue']   = '\033[94m'
color_dictionary['green']  = '\033[92m'
color_dictionary['red']    = '\033[91m'
color_dictionary['yellow'] = '\033[93m'
color_dictionary['cyan']   = '\033[96m'
color_dictionary['purple'] = '\033[95m'



# progressbar display
def progressbar(progress, pbnum=40):
    global color_dictionary
    import sys
    loadbar = ""
    for i in range(pbnum):
        if i<=int(progress*float(pbnum)):
            loadbar = loadbar + "="
        else:
            loadbar = loadbar + " "
    #~ sys.stdout.write("\r"+"["+loadbar+"] "+str(int(progress*100.0))+"%")
    if progress < 0.33:
        sys.stdout.write("\r"+"["+color_dictionary['red'   ]+loadbar+'\033[0m'+"]"+" "+str(int(progress*100.0))+"%")
    elif progress < 0.66:
        sys.stdout.write("\r"+"["+color_dictionary['yellow']+loadbar+'\033[0m'+"]"+" "+str(int(progress*100.0))+"%")
    else:
        sys.stdout.write("\r"+"["+color_dictionary['green' ]+loadbar+'\033[0m'+"]"+" "+str(int(progress*100.0))+"%")
    if progress>=1.0:
        print(" "+'\033[0m'+"OK "+'\033[0m')
















'''


# loading the h5 format
def load_pointneuron_circuit(h5_filename, synTypes = [], returnParas = [], randomize_neurons = False, neuroIDs = None):
    import h5py
    import numpy as np
    import nest
    circuit = {}
    print("Opening h5 datafile "+'\033[96m'+"\""+h5_filename+"\""+'\033[0m'+" ... ")
    h5file = h5py.File(h5_filename,'r')
    # return custom keys
    for rP in returnParas:
        circuit[rP] = h5file[rP][:]
    # subgroup location for synapses
    synapse_dataset_location = h5file["synapse_dataset_location"][:]
    Nh5 = h5file["x"][:].shape[0]
    if neuroIDs==None:
        N = Nh5
        neuroIDs_ = np.array(range(N))
        translatorDict = np.array(range(N))
    else:
        N = neuroIDs.shape[0]
        print("Sorting new ID list")
        neuroIDs_ = np.sort(neuroIDs)
        print("OK")
        translatorDict = -np.ones(Nh5, np.int32)
        translatorDict[ neuroIDs_ ] = np.array(range(N))
    # creating neurons
    cells = nest.Create('aeif_cond_exp', N)
    circuit["cells"] = cells
    totNumSyns = 0
    
    for sT in synTypes:
        for i,gid in enumerate(neuroIDs_):
            if i%100 == 0:
                progressbar(float(i)/float(N), 60)


            hasAffSyns = True
            try:
                #~ synT_IN = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_IN_T_" +str(gid)][:]
                synT_IN = translatorDict[ h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_IN_T_" +str(gid)][:] ]
                validSynapses = np.where(synT_IN!=-1)
                syn_IN  = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_IN_"   +str(gid)][:]
                # only take synapses that target the current circuit
                synT_IN = synT_IN[ validSynapses ]
                syn_IN  = syn_IN[  validSynapses ]
                totNumSyns += synT_IN.shape[0]
            except:
                hasAffSyns = False
            if hasAffSyns:
                synT_IN = np.int64(synT_IN)+1
                syn_IN  = np.float64(syn_IN[:])
                if type(synT_IN)==np.float64:
                    synT_IN = np.array([synT_IN])
                    syn_IN  = np.array([syn_IN])
                nest.DataConnect(list( synT_IN ), [{
                'target':  float(i)*np.ones(synT_IN.shape[0], np.float64)+1.0 ,
                'delay':   0.2*np.ones(syn_IN[:].shape[0], np.float64),
                'weight':  syn_IN[:],
                'U':       0.2*np.ones(syn_IN[:].shape[0], np.float64),
                'tau_rec': 0.2*np.ones(syn_IN[:].shape[0], np.float64),
                'tau_fac': 0.2*np.ones(syn_IN[:].shape[0], np.float64)
                }], 'tsodyks2_synapse')



            hasEffSyns = True
            try:
                #~ synT_OUT = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_OUT_T_" +str(gid)][:]
                synT_OUT = translatorDict[ h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_OUT_T_" +str(gid)][:] ]
                validSynapses = np.where(synT_OUT!=-1)            
                syn_OUT  = h5file[ "syngroup_"+str(synapse_dataset_location[gid]) ]["syn_"+sT+"_OUT_"   +str(gid)][:]
                # only take synapses that target the current circuit
                synT_OUT = synT_OUT[ validSynapses ]
                syn_OUT  = syn_OUT[  validSynapses ]
                totNumSyns += synT_OUT.shape[0]
            except:
                hasEffSyns = False
            if hasEffSyns:
                synT_OUT = np.float64( np.int64(synT_OUT)+1 )
                syn_OUT  = np.float64(syn_OUT[:])
                if type(synT_OUT)==np.float64:
                    synT_OUT = np.array([synT_OUT])
                    syn_OUT  = np.array([syn_OUT])
                nest.DataConnect([i+1], [{
                'target':  synT_OUT,
                'delay':   0.2*np.ones(syn_OUT[:].shape[0], np.float64),
                'weight':  syn_OUT[:],
                'U':       0.2*np.ones(syn_OUT[:].shape[0], np.float64),
                'tau_rec': 0.2*np.ones(syn_OUT[:].shape[0], np.float64),
                'tau_fac': 0.2*np.ones(syn_OUT[:].shape[0], np.float64)
                }], 'tsodyks2_synapse')  

    progressbar(1.0, 60)
    print("Total number of synapses:", totNumSyns)
    h5file.close()
    return circuit


'''







'''
# Old stuff


# loading the h5 format
def load_pointneuron_circuit_OLD(h5_filename, id_list = None, load_nest = True, return_synapses = False, randomize_neurons = False, accepted_synapse_categories = None):
    import h5py
    import numpy as np
    print ("Opening h5 datafile "+'\033[96m'+"\""+h5_filename+"\""+'\033[0m'+" ... ")
    h5file = h5py.File(h5_filename+".h5",'r')
    # remapping of gids, in case that id_list is a non-consecutive array
    acceptance_key_           = np.ones(len(h5file["x"].value), dtype=np.int8)
    translation_to_selection_ = (-1)*np.ones(len(h5file["x"].value), dtype=np.int64)
    id_list_ = range(1,len(h5file["x"].value)+1)
    if id_list!=None:
        id_list_ = id_list
        acceptance_key_ = np.zeros(len(h5file["x"].value), dtype=np.int8)
        acceptance_key_[np.array(id_list)-1] = 1
    accepted_ids_ = np.nonzero(acceptance_key_==1)[0]
    # circuit dictionary to return (only neural properties here)
    circuit = {
    "x"         : np.float32(h5file["x"].value[accepted_ids_]),
    "y"         : np.float32(h5file["y"].value[accepted_ids_]),
    "z"         : np.float32(h5file["z"].value[accepted_ids_]),
    "a"         : np.float32(h5file["a"].value[accepted_ids_]),
    "b"         : np.float32(h5file["b"].value[accepted_ids_]),
    "V_th"      : np.float32(h5file["V_th"].value[   accepted_ids_]),
    "Delta_T"   : np.float32(h5file["Delta_T"].value[accepted_ids_]),
    "C_m"       : np.float32(h5file["C_m"].value[    accepted_ids_]),
    "g_L"       : np.float32(h5file["g_L"].value[    accepted_ids_]),
    "V_reset"   : np.float32(h5file["V_reset"].value[accepted_ids_]),
    "tau_w"     : np.float32(h5file["tau_w"].value[  accepted_ids_]),
    "t_ref"     : np.float32(h5file["t_ref"].value[  accepted_ids_]),
    "V_peak"    : np.float32(h5file["V_peak"].value[ accepted_ids_]),
    "E_L"       : np.float32(h5file["E_L"].value[    accepted_ids_]),
    "E_ex"      : np.float32(h5file["E_ex"].value[   accepted_ids_]),
    "E_in"      : np.float32(h5file["E_in"].value[   accepted_ids_])
    }
    # number of neurons to be created
    N = accepted_ids_.shape[0]
    id_tmp = 0
    for iN in range(len(h5file["x"].value)):
        if acceptance_key_[iN]==1:
            translation_to_selection_[iN] = id_tmp
            id_tmp += 1
    # if loading into nest should be done here (uses less RAM as variables do not need to be returned)
    if load_nest:
        import nest
        # creating neurons
        print(N)
        cells = nest.Create('aeif_cond_exp', N)
        #~ cells = nest.Create('iaf_cond_exp', N)
        #~ cells = nest.Create('aeif_cond_alpha', N)
        #~ cells = nest.Create('aeif_cond_alpha_multisynapse', N)
        circuit["cells"] = cells
    print("Loading synapses ...")
    # dataset IDs of synapses (for multiple datasets created during parallel generation) (if of -1 means the synapse is in the main dataset)
    synapse_dataset_location = (-1)*np.ones(N, dtype=np.int16)
    try:
        synapse_dataset_location = h5file["synapse_dataset_location"].value[accepted_ids_]
        print("Dataset has synapse dataset locations ... ")
    except:
        print("Dataset has NO synapse dataset locations ... ")
    synapse_h5files = {}
    synapse_h5files["-1"] = h5file
    for synum in np.unique(synapse_dataset_location):
        if synum!=-1:
            synapse_h5files[ str(synum) ] = h5py.File(h5_filename+"_synapses_"+str(synum)+".h5",'r')
    # loop through all neurons and create their efferent synapses while loading (less RAM usage)
    for i,gid_ in enumerate(id_list_):
        #~ if h5file["layer"].value[gid_-1]==5:
        if i%100 == 0:
            progressbar(float(i)/float(N), 60)
        hasSyns = True
        try:
            r_syns    = synapse_h5files[str(synapse_dataset_location[i])]["syn_" +str(i)].value
            r_synsT   = synapse_h5files[str(synapse_dataset_location[i])]["synT_"+str(i)].value
            r_synsExp = synapse_h5files[str(synapse_dataset_location[i])]["synExp_"+str(i)].value
            #~ printCsX(r_syns.shape, "red")
            if accepted_synapse_categories!=None:
                r_syns_tmp = np.zeros((r_syns.shape[0], 0), dtype=r_syns.dtype)
                #~ accepted_ids_tmp = np.zeros((0), dtype = np.int64)
                for asc in accepted_synapse_categories:
                    #~ accepted_ids_tmp = np.concatenate( ( accepted_ids_tmp, np.nonzero( r_synsExp==asc )[0] ) )
                    accepted_ids_tmp = np.nonzero( r_synsExp==asc[0] )[0]
                    r_syns_selected = r_syns[ :, accepted_ids_tmp ]
                    # applied scaling factors for weight and delay and min/max distances (might become obsolete at some point)
                    r_syns_selected[ 1,: ] *= asc[1][0]
                    r_syns_selected[ 2,: ] *= asc[1][1]
                    tids_tmp = np.int64(r_syns_selected[ 0,: ])
                    accepted_ids_distance_range = np.ones(tids_tmp.shape, dtype=np.int64)
                    x_tmp = h5file["x"].value[i]; y_tmp = h5file["y"].value[i]; z_tmp = h5file["z"].value[i]
                    if asc[2]!=None:
                        accepted_ids_distance_range *= (np.linalg.norm(np.vstack(( h5file["x"].value[tids_tmp-1] - x_tmp, h5file["y"].value[tids_tmp-1] - y_tmp, h5file["z"].value[tids_tmp-1] - z_tmp)), axis=0) > asc[2] )
                    if asc[3]!=None:
                        accepted_ids_distance_range *= (np.linalg.norm(np.vstack(( h5file["x"].value[tids_tmp-1] - x_tmp, h5file["y"].value[tids_tmp-1] - y_tmp, h5file["z"].value[tids_tmp-1] - z_tmp)), axis=0) < asc[3] )
                    r_syns_selected = r_syns_selected[:, np.nonzero(accepted_ids_distance_range)[0] ]
                    r_syns_tmp = np.concatenate((r_syns_tmp, r_syns_selected), axis=1)
                r_syns = np.copy(r_syns_tmp)
            #~ printCsX(r_syns.shape, "green")
            #~ print(r_syns[2,:])
        except:
            hasSyns = False
        if hasSyns:
            #~ print(r_syns.shape[1], "synapses ... ")
            #~ ids_that_are_in = np.nonzero( (r_syns[0,:] >= id_min) * (r_syns[0,:] <= id_max) )[0]
            ids_that_are_in = np.nonzero(acceptance_key_[np.int64(r_syns[0,:])-1]==1)[0]                    
            # !!!!!!
            #ids_that_are_in = ids_that_are_in[ np.nonzero(h5file["layer"].value[np.int64(r_syns[0, ids_that_are_in])-1]==6) ]
            #~ if ids_that_are_in.shape[0]>1:
            if ids_that_are_in.shape[0]>1:
                if load_nest:
                    #~ print(i)
                    #~ if np.float64(r_syns[2, ids_that_are_in])[0]<0.0:
                    # create synapses in NEST
                    nest.DataConnect([i+1], [{
                    'target':  np.float64( translation_to_selection_[np.int64(r_synsT[ids_that_are_in])-1]+1 ),
                    'delay':   np.float64(r_syns[0, ids_that_are_in]),
                    'weight':  np.float64(r_syns[1, ids_that_are_in]),
                    'U':       np.float64(r_syns[2, ids_that_are_in]),
                    'tau_rec': np.float64(r_syns[3, ids_that_are_in]),
                    'tau_fac': np.float64(r_syns[4, ids_that_are_in])
                    }], 'tsodyks2_synapse')
                if return_synapses:
                    #~ circuit["syn_"+str(gid_)] = {
                    # OR/AND add them to the dictionary to return them
                    circuit["syn_"+str(i+1)] = {
                    'target':  np.float32( translation_to_selection_[np.int64(r_synsT[ids_that_are_in])-1]+1 ),
                    'delay':   np.float32(r_syns[0, ids_that_are_in]),
                    'weight':  np.float32(r_syns[1, ids_that_are_in]),
                    'U':       np.float32(r_syns[2, ids_that_are_in]),
                    'tau_rec': np.float32(r_syns[3, ids_that_are_in]),
                    'tau_fac': np.float32(r_syns[4, ids_that_are_in])
                    }
                    try:
                        circuit["syntype_"+str(i+1)] = synapse_h5files[str(synapse_dataset_location[i])]["syntype_"+str(gid_)].value[ids_that_are_in]
                    except:
                        tmp1 = 0
    progressbar(1.0, 60)
    # checking for other neural properties which might or might not exist depending on the circuit creation
    try:   circuit["gid"] = np.int64(h5file["gid"].value[accepted_ids_])
    except:print("No GID properties ... ")
    try:   circuit["layer"] = h5file["layer"].value[accepted_ids_]
    except:print("No layer properties ... ")
    try:   circuit["excitatory"] = np.int16(h5file["excitatory"].value[accepted_ids_])
    except:print("No E/I properties ... ")
    try:   circuit["mtype"] = h5file["mtype"].value[accepted_ids_]
    except:print("No M-type properties ... ")
    try:   circuit["lmtype_number"] = h5file["lmtype_number"].value[accepted_ids_]
    except:print("No LM-type properties ... ")
    try:   circuit["mtype_number"] = h5file["mtype_number"].value[accepted_ids_]
    except:print("No M-type number properties ... ")
    try:   circuit["Larea"] = np.int16(h5file["Larea"].value[accepted_ids_])
    except:print("No Area properties ... ")
    try:
        circuit["vtx"] = h5file["vtx"].value[accepted_ids_]
        circuit["vty"] = h5file["vty"].value[accepted_ids_]
        circuit["mapnumber"] = h5file["mapnumber"].value[accepted_ids_]
    except:print("No mapping properties ... ")
    try:   circuit["isExc"] = h5file["isExc"].value[accepted_ids_]
    except:print("No E/I (int16) properties ... ")
    try:   circuit["isColumn"] = h5file["isColumn"].value[accepted_ids_]
    except:print("No isColumn properties ... ")
    for mapname in ["SSmap_left", "SSmap_right", "whiskermap_left", "whiskermap_right"]:
        try:circuit[mapname] = h5file[mapname].value[accepted_ids_]
        except:print("No " + mapname)
    try:
        circuit["colorx"] = h5file["colorx"].value[accepted_ids_]
        circuit["colory"] = h5file["colory"].value[accepted_ids_]
        circuit["colorz"] = h5file["colorz"].value[accepted_ids_]
    except:print("No color properties ... ")
    try:
        circuit["depth"] = h5file["depth"].value[accepted_ids_]
    except:print("No depth properties ... ")
    # some nest property setting
    if load_nest:
        print("Setting neural parameters")
        nest.SetStatus(cells, 'tau_syn_ex', 1.8)
        nest.SetStatus(cells, 'tau_syn_in', 8.0)
        # AdEx parameters are set
        nest.SetStatus(cells, 'a',       circuit["a"])
        nest.SetStatus(cells, 'b',       circuit["b"])
        nest.SetStatus(cells, 'V_th',    circuit["V_th"])
        nest.SetStatus(cells, 'Delta_T', circuit["Delta_T"])
        nest.SetStatus(cells, 'C_m',     circuit["C_m"])
        nest.SetStatus(cells, 'g_L',     circuit["g_L"])
        nest.SetStatus(cells, 'V_reset', circuit["V_reset"])
        nest.SetStatus(cells, 'tau_w',   circuit["tau_w"])
        nest.SetStatus(cells, 't_ref',   circuit["t_ref"])
        nest.SetStatus(cells, 'V_peak',  circuit["V_peak"])
        nest.SetStatus(cells, 'E_L',     circuit["E_L"])
        nest.SetStatus(cells, 'E_ex',    circuit["E_ex"])
        nest.SetStatus(cells, 'E_in',    circuit["E_in"])
    if randomize_neurons:
        print("Randomizing neurons")
        VV = nest.GetStatus(cells, "V_m")
        nest.SetStatus(cells, "V_m", list( np.copy(np.array(VV)*(0.0) - 150.0 + 100.0*np.random.rand(np.array(VV).shape[0])) ) )
        ww = nest.GetStatus(cells, "w")
        nest.SetStatus(cells, "w", list(np.array(ww)*1.0) )
        VV = None
        ww = None   
    h5file.close()
    return circuit

'''
