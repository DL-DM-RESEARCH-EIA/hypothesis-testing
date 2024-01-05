import nndm_library as nndm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os

def measure_execution_time(func, *args, **kwargs):
    """
    Measures the execution time of a given function.

    :param func: The function to measure.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function execution.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time taken to run {func.__name__}: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    return result

def read_lhe_file(file_path):
    """Reads an LHE file and extracts parameters."""
    files_lhe = nndm.ReadLhe(file_path, read_data=False)
    return files_lhe, files_lhe.extract_params_from_path()

def read_other_file(file_path, ext):
    """Reads an LHE file and extracts parameters."""
    files_other = nndm.ReadFileBase(file_path, ext=ext, read_data=False)
    return files_other, files_other.extract_params_from_path()

def check_conditions(param_dict, dict_constrained, percentage_tolerance=1e-3):
    """Checks if parameters meet the specified conditions."""
    conditions_met = []
    for key in dict_constrained.keys():
        condition = np.abs(np.array(param_dict[key]) - dict_constrained[key]) / np.array(param_dict[key]) < percentage_tolerance
        conditions_met.append(condition)
    return conditions_met

def apply_filter(conditions_met):
    """Applies a filter to determine which files meet the conditions."""
    filter_array = None
    for condition in conditions_met:
        if filter_array is None:
            filter_array = condition
        else:
            filter_array = filter_array * condition
    return np.where(filter_array)[0]

def read_names_constrained(file_path, dict_constrained, ext=".lhe"):
    """Main function that uses helper functions to read files and apply constraints."""
        
    if ext == ".lhe":
        files_lhe, param_dict = read_lhe_file(file_path)
    else:
        files_lhe, param_dict = read_other_file(file_path, ext=ext)
    
    if len(param_dict) == 0:
        print(f"Please check that the file_path={file_path} or dict_constrained={dict_constrained} has sense in the context")
        exit(0)

    conditions_met = check_conditions(param_dict, dict_constrained)
    values = apply_filter(conditions_met)
    files = [files_lhe.files_dir[int(v)] for v in list(values)]
    return files

def find_weights_for_given_params(file_path, csv_file_path, dict_constrained, var_of_interest):
    files = read_names_constrained(file_path, dict_constrained)    
    files_lhe = nndm.ReadLhe(files, read_data=False)

    params_file = files_lhe.extract_params_from_path()

    # This is to create the expected values in the traning dataset

    columns2 = ['mChi2', 'mEta2', 'alphaD', 'eps', 'Lp', 'Lc',
                'Le', 'Lhp', 'Lec', 'Lcp', 'Lhc', 'Lpe', 'Lhe',
                'gx', 'g1x', 'g1x_c', 'thetaP', 'mzp', 'm_eta',
                'm_chi', 'm_hh2', 'm_hh1','vp', 'sg_pion', 'sg_eta',
                'sg_omega', 'signif', 'delta', 'ntt',
                'Nt_pions', 'Nt_etas', 'Nt_omegas', 'N_bkg',
                'kf_pion', 'kf_eta','kf_omega', 'eps-_t','chi2_pm',
                'Nets_pm', 'effic_pion', 'effic_eta','eps_likelihood',
                'eps_t_y', 'kl', 'ntt_01','ntt_02','ntt_1','ntt_2',
                'Gamma_T0', 'Br_chi', 'Br_eta', 'epss2', 'mk']


    provided_values = {
        'mk': params_file['mk1'][0],
        'eps': params_file['eps2'][0] ** (0.5),
    }

    df_proportions_info = pd.read_csv(csv_file_path, names=columns2) # input

    precision = 0.5 / 100
    for key, value in provided_values.items():
        lower_bound = value * (1 - precision)
        upper_bound = value * (1 + precision)
        df_proportions_info = df_proportions_info[(df_proportions_info[key] >= lower_bound) & (df_proportions_info[key] <= upper_bound)]

    sg_eta = df_proportions_info['sg_eta'].mean()
    sg_pion = df_proportions_info['sg_pion'].mean()
    Br_eta = sg_eta / (sg_pion + sg_eta)
    Br_pion = sg_pion / (sg_pion + sg_eta)

    Br_mk1 = df_proportions_info['Br_chi'].max()
    Br_mk2 = df_proportions_info['Br_chi'].min()

    # Go over param files for the correct weights

    # Update each entry with the correct weight
    for i in range(len(params_file['particle_type'])):
        particle_type = params_file['particle_type'][i]
        flag = params_file['flag'][i]

        if particle_type == 111 and flag == True:
            weight = Br_mk1 * Br_pion
        elif particle_type == 111 and flag == False:
            weight = Br_mk2 * Br_pion
        elif particle_type == 221 and flag == False:
            weight = Br_mk2 * Br_eta
        elif particle_type == 221 and flag == True:
            weight = Br_mk1 * Br_eta

        # Add the calculated weight to the params_files
        if 'weight' in params_file:
            params_file['weight'].append(weight)
        else:
            params_file['weight'] = [weight]


    return params_file


def add_weights_to_given_singnal_df(file_path, csv_file_path, dict_constrained, var_of_interest, n_samples=None):
    params_file = find_weights_for_given_params(file_path, csv_file_path, dict_constrained, var_of_interest)

    # Here we really read the data 
    files = read_names_constrained(file_path, dict_constrained)     
    files_lhe = nndm.ReadLhe(files, var_of_interest=var_of_interest, n_samples=n_samples)
    
    files_lhe.data['weight'] = files_lhe.data['path'].apply(lambda x: params_file['weight'][x])

    return files_lhe.data


# Function to filter the data based on mass
def filter_mass(t_list, x_list, y_list, z_list, mass_list, target_mass, precision):
    lower_bound = target_mass * (1 - precision)
    upper_bound = target_mass * (1 + precision)
    for t, x, y, z, mass in zip(t_list, x_list, y_list, z_list, mass_list):
        if lower_bound <= mass <= upper_bound and mass != 0:
            return t, x, y, z, mass
    return None, None, None, None, None  # Return None if no matching mass found


def limit_datafame_to_given_mass_events(df, target_mass, precision):
    # Apply the filter function to each row
    df[['out.t', 'out.x', 'out.y', 'out.z', 'out._mass']] = df.apply(lambda row: filter_mass(
        row['out.t'], row['out.x'], row['out.y'], row['out.z'], row['out._mass'], 
        target_mass, precision), axis=1, result_type='expand')

    # Remove rows with None values after filtering
    df = df.dropna()

    return df

def add_back_relative_weights(df, mode_relative_weights):
    total_weight = sum(mode_relative_weights.values())
    normalized_weights = {key: val / total_weight for key, val in mode_relative_weights.items()}

    file = nndm.ReadRoot("Data/Background", output_base_tree="treeout", 
                    pattern_output="first", output_base_middle_branch = "/e/out",
                    leafs = ["out.t", "out.x", "out.y", "out.z", "out._mass"], recursive=True, read_data=False)    

    filename_map = file.files_dir

    # Map the path to filename and then to the corresponding normalized weight
    df['weight'] = df['path'].map(filename_map).map(normalized_weights)

    # 4 significant figures
    df['weight'] = df['weight'].apply(lambda x: '{:.4g}'.format(x))

    return df

def read_background_df(n_samples=None):
    file = nndm.ReadRoot("Data/Background", output_base_tree="treeout", 
            pattern_output="first", output_base_middle_branch = "/e/out",
            leafs = ["out.t", "out.x", "out.y", "out.z", "out._mass"], recursive=True, n_samples=n_samples)


    # Target mass and precision
    target_mass = 0.51099
    precision = 0.1 / 100  # 0.1%

    file.data = limit_datafame_to_given_mass_events(file.data, target_mass, precision)

    mode_relative_weights = {"Data/Background/v_e_scattering/onnumulepton10125.root": 63.5, 
                            "Data/Background/v_e_scattering/onnuelepton10125.root" : 6.0, 
                            "Data/Background/v_e_scattering/onantinumulepton10125.root" : 45.0, 
                            "Data/Background/v_e_scattering/onantinuelepton10125.root" : 2.0, 
                            "Data/Background/CCQE/antinuonnue.root" : 7.0, 
                            "Data/Background/CCQE/nuonnue.root" :1.5}

    file.data = add_back_relative_weights(file.data, mode_relative_weights)

    return file.data


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
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08}

    files = read_names_constrained(file_path, dict_constrained)    

    print(files)

# substep 3: create weights for the  eta and pion casees when mk1 or either mk2 are the ones in simulation
def solutions_substep3():
    file_path = "Data/Signal/maddump_scan3/" # input 
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    var_of_interest=['e', 'px', 'py', 'pz'] # input
    csv_file_path = "Data/Signal/data_scan/run_3/out_scan_parcial.csv"

    params_file = find_weights_for_given_params(file_path, csv_file_path, dict_constrained, var_of_interest)

    print(params_file)


# substep 4: add to the read data frame of the signal distribution the given weights
def solutions_substep4():
    file_path = "Data/Signal/maddump_scan3/" # input 
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    var_of_interest=['e', 'px', 'py', 'pz'] # input
    csv_file_path = "Data/Signal/data_scan/run_3/out_scan_parcial.csv"

    df_with_weight = add_weights_to_given_singnal_df(file_path, csv_file_path, dict_constrained, var_of_interest)

    print(df_with_weight)


# substeb 5: read the background and limit the data to the electrons that are scattered over
def solution_substep5():
    data = {
        'out.t': [
            [4214.118570980115, 154.60560786841492],
            [1821.025872419834, 2966.345802134001],
            [3998.2211594766304, 378.32737450792126],
            [1075.3575082818018, 1490.883050632886],
            [2468.113680886015, 1384.1976539701386]
        ],
        'out.x': [
            [12.302239749652465, -12.302239749652465],
            [-6.595450880957304, 6.595450880957304],
            [-0.56910228514902, 0.56910228514902],
            [-1.4657242362880825, 1.4657242362880825],
            [12.32459404681569, -12.32459404681569]
        ],
        'out.y': [  # Made-up data for out.y
            [3.5, -3.5],
            [1.2, -1.2],
            [2.1, -2.1],
            [0.8, -0.8],
            [4.0, -4.0]
        ],
        'out.z': [  # Made-up data for out.z
            [5.5, 5.6],
            [2.3, 2.4],
            [3.7, 3.8],
            [1.4, 1.5],
            [6.2, 6.3]
        ],
        'out._mass': [
            [0.0, 0.5109990000000001],
            [0.0, 0.5109990000000001],
            [0.0, 0.5109990000000001],
            [0.0, 0.5109990000000001],
            [0.0, 0.5109990000000001]
        ],
        'path': [0, 0, 0, 0, 0]
    }

    # Creating the DataFrame
    df = pd.DataFrame(data)

    # Target mass and precision
    target_mass = 0.51099
    precision = 0.1 / 100  # 0.1%

    new_df = limit_datafame_to_given_mass_events(df, target_mass, precision)

    print(new_df)

# substep 6: add the correct weights at the different files
def substep_6():
    data = {
        'out.t': [154.605608, 2966.345802, 878.327375, 378.327375, 1490.883051, 1384.197654],
        'out.x': [-12.302240, 6.595451, 0.880902, 0.569102, 1.465724, -12.324594],
        'out.y': [-3.5, -1.2, -2.1, 0.9, -0.8, -4.0],
        'out.z': [5.6, 2.4, 3.8, 4.3, 1.5, 6.3],
        'out._mass': [0.510999, 0.510999, 0.510999, 0.510999, 0.510999, 0.510999],
        'path' : [0, 1, 2, 3, 4, 5]
    }

    df = pd.DataFrame(data)

    mode_relative_weights = {"Data/Background/v_e_scattering/onnumulepton10125.root": 63.5, 
                            "Data/Background/v_e_scattering/onnuelepton10125.root" : 6.0, 
                            "Data/Background/v_e_scattering/onantinumulepton10125.root" : 45.0, 
                            "Data/Background/v_e_scattering/onantinuelepton10125.root" : 2.0, 
                            "Data/Background/CCQE/antinuonnue.root" : 7.0, 
                            "Data/Background/CCQE/nuonnue.root" :1.5}


    df_new = add_back_relative_weights(df, mode_relative_weights)


    print(df_new)

# substep 7: read background and add weights    
def substep_7():
    file = nndm.ReadRoot("Data/Background", output_base_tree="treeout", 
                pattern_output="first", output_base_middle_branch = "/e/out",
                leafs = ["out.t", "out.x", "out.y", "out.z", "out._mass"], recursive=True)


    # Target mass and precision
    target_mass = 0.51099
    precision = 0.1 / 100  # 0.1%

    file.data = limit_datafame_to_given_mass_events(file.data, target_mass, precision)

    mode_relative_weights = {"Data/Background/v_e_scattering/onnumulepton10125.root": 63.5, 
                            "Data/Background/v_e_scattering/onnuelepton10125.root" : 6.0, 
                            "Data/Background/v_e_scattering/onantinumulepton10125.root" : 45.0, 
                            "Data/Background/v_e_scattering/onantinuelepton10125.root" : 2.0, 
                            "Data/Background/CCQE/antinuonnue.root" : 7.0, 
                            "Data/Background/CCQE/nuonnue.root" :1.5}

    file.data = add_back_relative_weights(file.data, mode_relative_weights)

    print(file.data)

# substep 8: read the background and signal and separate data in traning and test
def substep_8():
    file_path = "Data/Signal/maddump_scan3/" # input 
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    var_of_interest=['e', 'px', 'py', 'pz'] # input
    csv_file_path = "Data/Signal/data_scan/run_3/out_scan_parcial.csv"

    df_signal = add_weights_to_given_singnal_df(file_path, csv_file_path, dict_constrained, var_of_interest)
    df_back = read_background_df()

    print(df_back, df_signal)

# substep 9: given the input of two dataframs that emulated the distribution of signal and background create test and train distributions 
#   approximations of the distributions for each dataframe (signal, background) and save them

#   A consistent approach would be to increase the dataset size until a convergence is reached. This removes undesired behaviour 
#     coming from not having enough size information for each dataset. Despite this, our purposes are to make the comparison
#     of two methods, we will take traning conditions to be a possible realization for the mean time and come later to asses how not
#     having more care in thos affect performance. In then end, we assume thatif the test and training converge to similar points in 
#     certain parameters, then they are statistically equivalent in the precision of certain information.
    
# To avoid overfitting we divide the dataframes from the begginig since doing an aggresive bootstraping would cause overfitting problems.
#   So, before all we do from each a train and test dataframe and for a first approximation we will use under sampling. This divided dataframes
#  represent the approimation of the resultant signal+background distributions.

def create_test_filename(base, params):
    return f"{base}_{'_'.join(f'{key}_{value}' for key, value in params.items())}.pkl"

def save_dataframe(df, base_filename, params, target_dir, params_name=True):
    if params_name:
        filename = create_test_filename(base_filename, params)
    else:
        filename = base_filename + ".pkl"

    df.to_pickle(os.path.join(target_dir, filename))

def sub_step9():
    file_path = "Data/Signal/maddump_scan3/" # input 
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    var_of_interest=['e', 'px', 'py', 'pz'] # input
    csv_file_path = "Data/Signal/data_scan/run_3/out_scan_parcial.csv"

    n_samples = None

    df_signal = add_weights_to_given_singnal_df(file_path, csv_file_path, dict_constrained, var_of_interest, n_samples=n_samples)
    df_back = read_background_df(n_samples)

    df_back.rename(columns={'out.t': 'e', 'out.x': 'px', 'out.y': 'py', 'out.z': 'pz', 'path': 'path', 'weight': 'weight'}, inplace=True)

    # Remove the 'out._mass' column
    df_back.drop(columns='out._mass', inplace=True)

    # Divide the 'e' column by 1000
    df_back['e'] = df_back['e'] / 1000

    # Convert 'weight' column in df_back
    df_back['weight'] = pd.to_numeric(df_back['weight'], errors='coerce')

    # Convert 'weight' column in df_signal
    df_signal['weight'] = pd.to_numeric(df_signal['weight'], errors='coerce')

    # Split df_back
    df_back_train, df_back_test = train_test_split(df_back, test_size=0.2, random_state=42)

    # Split df_signal
    df_signal_train, df_signal_test = train_test_split(df_signal, test_size=0.2, random_state=42)


    # Directory paths
    base_dir = "Data"
    target_dir = os.path.join(base_dir, "MLTrainData/DataframeDistributions")

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Assuming df_back_train, df_back_test, df_signal_train, df_signal_test are defined
    # Save the dataframes as pickle files in the new directory
    save_dataframe(df_back_train, "df_back_train_PDF", dict_constrained, target_dir, params_name=False)
    save_dataframe(df_back_test, "df_back_test_PDF", dict_constrained, target_dir, params_name=False)
    save_dataframe(df_signal_train, "df_signal_train_PDF", dict_constrained, target_dir)
    save_dataframe(df_signal_test, "df_signal_test_PDF", dict_constrained, target_dir)


# substep 10: based in the approximated distributions in each dataframe create a function to create the dataframe for the training
#   and anther one for the test. In short: make the combined df to sample in trainings having the same proportions than in the initial
#   dataframes that approximate the distributions.

def find_string_with_substring(strings, substring):
    matching_strings = [s for s in strings if substring in s]
    return matching_strings[0] if matching_strings else None

def sub_step10():
    # Load the dataframes from pickle files in the 'Data' directory

    file_path = 'Data/MLTrainData/DataframeDistributions/'
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    files = read_names_constrained(file_path, dict_constrained, ext=".pkl")
    df_singal_train_path = find_string_with_substring(files, 'test')
    df_signal_train = pd.read_pickle(df_singal_train_path)
    df_back_train = pd.read_pickle(file_path + "df_back_train_PDF.pkl")

    # Divide them and construct thethe train ones.

    def bootstrap_df(df, n_samples):
        return df.sample(n=n_samples, replace=True, random_state=42)

    # Determine which dataframe to bootstrap
    if len(df_back_train) < len(df_signal_train):
        df_back_train_bootstrapped = bootstrap_df(df_back_train, len(df_signal_train))
        df_signal_train_final = df_signal_train
        df_back_train_final = df_back_train_bootstrapped
    elif len(df_signal_train) < len(df_back_train):
        df_signal_train_bootstrapped = bootstrap_df(df_signal_train, len(df_back_train))
        df_back_train_final = df_back_train
        df_signal_train_final = df_signal_train_bootstrapped
    else:
        # If they are equal, no bootstrapping needed
        df_back_train_final = df_back_train
        df_signal_train_final = df_signal_train


    df_signal_train_final['label'] = 1
    df_back_train_final['label'] = 0

    df_train_combined = pd.concat([df_back_train_final, df_signal_train_final], ignore_index=True)    

    # Create a combined column for stratification
    df_train_combined['stratify_col'] = df_train_combined['label'].astype(str) + "_" + df_train_combined['path'].astype(str)

    print("\nInitial proportions")
    print("\ndf_combined[X_train['label'] == 0]['path'] proportions")
    print(df_train_combined[df_train_combined['label'] == 0]['path'].value_counts(normalize=True))  # Background
    print("\ndf_train_combined[df_train_combined['label'] == 1]['path'] proportions")
    print(df_train_combined[df_train_combined['label'] == 1]['path'].value_counts(normalize=True))  # Signal

    print("\ndf_combined['label'] proportions")
    print(df_train_combined['label'].value_counts(normalize=True))  # Background

    # Now use train_test_split with this stratification
    X_train, X_val = train_test_split(
        df_train_combined, 
        test_size=0.2,  # Adjust as needed
        stratify=df_train_combined['stratify_col'], 
        random_state=42
    )

    X_train = X_train.drop(columns=['stratify_col'])
    X_val = X_val.drop(columns=['stratify_col'])

    print("\nX_train[path] proportions")
    print(X_train['path'].value_counts(normalize=True))

    print("\nX_val[path] proportions")
    print(X_val['path'].value_counts(normalize=True))

    # Check class balance
    print("\nX_train[label] proportions")
    print(X_train['label'].value_counts(normalize=True))
    print("\nX_train[label] proportions")
    print(X_val['label'].value_counts(normalize=True))

    # Check `path` balance within each class
    print("\nX_train[X_train['label'] == 0]['path'] proportions")
    print(X_train[X_train['label'] == 0]['path'].value_counts(normalize=True))  # Background
    print("\nX_train[X_train['label'] == 1]['path'] proportions")
    print(X_train[X_train['label'] == 1]['path'].value_counts(normalize=True))  # Signal

    base_dir = "Data"
    target_dir = os.path.join(base_dir, "MLTrainData/TrainDataDist/")

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Assuming df_back_train, df_back_test, df_signal_train, df_signal_test are defined
    # Save the dataframes as pickle files in the new directory
    save_dataframe(X_train, "df_train", dict_constrained, target_dir)
    save_dataframe(X_val, "df_val", dict_constrained, target_dir)


# Create a function that creates a training dataset of a given size and the corresponding val dataframe, separation 
#  in principle is of the same size as in the division before
def create_dataset_dataframe_balanced(path, dict_constrained, n_data, dataset_type='df_train'):
    """
    Creates a dataset dataframe of a given size for training or validation.

    :param path: The file path to the dataset.
    :param dict_constrained: Dictionary with constraints to select the correct file.
    :param n_data: The number of data points to sample.
    :param dataset_type: The type of dataset to create ('df_train' or 'df_val').
    :return: A dataframe sampled according to weights.
    """
    files = read_names_constrained(path, dict_constrained, ext=".pkl")
    df_path = find_string_with_substring(files, dataset_type)
    df = pd.read_pickle(df_path)

    df_signal = df[df['label'] == 1]
    df_background = df[df['label'] == 0]

    n_samples_per_class = int(n_data / 2)

    df_signal_sampled = df_signal.sample(n=n_samples_per_class, weights='weight', random_state=42, replace=True)
    df_background_sampled = df_background.sample(n=n_samples_per_class, weights='weight', random_state=42, replace=True)

    sampled_df = pd.concat([df_signal_sampled, df_background_sampled])
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=['path', 'weight'], errors='ignore')

    return sampled_df

def sub_step11():
    file_path = 'Data/MLTrainData/TrainDataDist/'
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    n_data = 1e6
    df_train  =  create_dataset_dataframe_balanced(file_path, dict_constrained, n_data)
    df_val = create_dataset_dataframe_balanced(file_path, dict_constrained, int(n_data * 0.2), dataset_type='df_val')
    # finally save it here

    base_dir = "Data"
    target_dir = os.path.join(base_dir, "MLTrainData/TrainData/")

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Assuming df_back_train, df_back_test, df_signal_train, df_signal_test are defined
    # Save the dataframes as pickle files in the new directory
    save_dataframe(df_train, "train", dict_constrained, target_dir)
    save_dataframe(df_val, "val", dict_constrained, target_dir)


def mean_number_of_s_and_b(sample_file_info, mk1, eps2):
    columns2 = ['mChi2', 'mEta2', 'alphaD', 'eps', 'Lp', 'Lc',
                'Le', 'Lhp', 'Lec', 'Lcp', 'Lhc', 'Lpe', 'Lhe',
                'gx', 'g1x', 'g1x_c', 'thetaP', 'mzp', 'm_eta',
                'm_chi', 'm_hh2', 'm_hh1','vp', 'sg_pion', 'sg_eta',
                'sg_omega', 'signif', 'delta', 'ntt',
                'Nt_pions', 'Nt_etas', 'Nt_omegas', 'N_bkg',
                'kf_pion', 'kf_eta','kf_omega', 'eps-_t','chi2_pm',
                'Nets_pm', 'effic_pion', 'effic_eta','eps_likelihood',
                'eps_t_y', 'kl', 'ntt_01','ntt_02','ntt_1','ntt_2',
                'Gamma_T0', 'Br_chi', 'Br_eta', 'epss2', 'mk']

    df_nsignal = pd.read_csv(sample_file_info, names=columns2)

    n_signal = df_nsignal.loc[abs(df_nsignal['mk'] - mk1) < 1e-3]
    n_signal = n_signal[abs(n_signal.eps - eps2 ** 0.5) < 1e-6]
    n_signal = int(n_signal['ntt_1'].max() + n_signal['ntt_2'].max())
    
    # Coefficients for the quadratic equation
    Z = 1.64
    a = Z**2 * 0.1**2
    b = Z**2
    c = -n_signal**2
    # Use np.roots to solve for Nt
    coefficients = [a, b, c]
    n_total = np.roots(coefficients).max()

    n_background = int(n_total - n_signal)
    return n_signal, n_background

def create_test_dataframe(path, dict_constrained):
    df_test_path_back = "hypothesis-testing/jet-cnn/Data/MLTrainData/DataframeDistributions/df_back_test_PDF.pkl"
    df_background = pd.read_pickle(df_test_path_back)
    print(df_background)

    files = read_names_constrained(path, dict_constrained, ext=".pkl")
    df_test_path = find_string_with_substring(files, 'df_signal_test')
    print(df_test_path)
    df_signal = pd.read_pickle(df_test_path)

    # Find the data nearst to the eps and background
    sample_file_info = 'hypothesis-testing/jet-cnn/Data/Signal/data_scan/run_3/out_scan_parcial.csv'
    n_s, n_b = mean_number_of_s_and_b(sample_file_info, dict_constrained['mk1'], dict_constrained['eps2'])

    # Sample from each class with weights
    df_signal_sampled = df_signal.sample(n=n_s, weights='weight', random_state=42, replace=True)
    df_background_sampled = df_background.sample(n=n_b, weights='weight', random_state=42, replace=True)

    df_signal_sampled['label'] = 1
    df_background_sampled['label'] = 0
    # Combine the sampled data
    sampled_df = pd.concat([df_signal_sampled, df_background_sampled])

    # Shuffle the combined dataframe if necessary
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(sampled_df['label'].value_counts())

    # Drop the 'path' and 'weight' columns if they exist
    sampled_df = sampled_df.drop(columns=['path', 'weight', 'label'], errors='ignore')

    return sampled_df


def sub_step12():
    file_path = 'Data/MLTrainData/DataframeDistributions/'
    dict_constrained = {"mk1": 0.02514990210703923, "delta": 1.05,  "eps2": 1.4550810518824755e-08} # input
    df_test  =  create_test_dataframe(file_path, dict_constrained)


    base_dir = "Data"
    target_dir = os.path.join(base_dir, "MLTrainData/TestData/")

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Assuming df_back_train, df_back_test, df_signal_train, df_signal_test are defined
    # Save the dataframes as pickle files in the new directory
    save_dataframe(df_test, "test", dict_constrained, target_dir)


# sub_step12()
