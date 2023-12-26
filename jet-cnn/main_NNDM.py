import nndm_library as nndm
import pylhe
import numpy as np

###  Step one: read a file given a certain mk1 delta and eps


# supstep 1: extract the possible mk1, delta and eps to make it real
def solution_substep1():
    # first let us read the parameters of the signals
    scan_dir  = "Data/Signal/maddump_scan3/"
    files_lhe = nndm.ReadLhe(scan_dir, var_of_interest=['e', 'px', 'py', 'pz'], read_data=False)
    param_dict = files_lhe.extract_params_from_path()
    print(param_dict)


# substep 2: give the input file_path where files should be read and the parameters to be find in the name of the file
def solution_substep2():
    # Check that given the mk1, delta and eps the correct reading is done  
    file_path = "Data/Signal/maddump_scan3/"
    dir_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08}
    read_names_constrained(file_path, dir_constrained, var_of_interest=['e', 'px', 'py', 'pz'])

    files = read_names_constrained(file_path, dir_constrained, var_of_interest=['e', 'px', 'py', 'pz'])    

    print(files)

# substep 3: balance the events that are eta, pion, mk1 and mk2

def read_names_constrained(file_path, dir_constrained, var_of_interest=['e', 'px', 'py', 'pz']):
    files_lhe = nndm.ReadLhe(file_path, var_of_interest=var_of_interest, read_data=False)
    param_dict = files_lhe.extract_params_from_path()
    
    percentage_tolerance = 1e-3
    conditions_met = [np.abs(np.array(param_dict[key]) - dir_constrained[key]) / np.array(param_dict[key]) < percentage_tolerance 
                      for key in dir_constrained.keys()]
    
    # extract the bool array that shows in which instances the conditions are satisiferized
    filter_array = None

    for filter in conditions_met:
        if filter_array is None:
            filter_array = filter
        else:
            filter_array = filter_array * filter

    values = np.where(filter_array)[0]
    
    files = [files_lhe.files_dir[int(v)] for v in list(values)]
    
    return files

