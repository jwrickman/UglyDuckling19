from dataset_prep.dataset_utils import create_dataset
from sklearn.model_selection import train_test_split
import os


def collect_filenames(input_path):
    """Collect all valid '.record' files from input_path directory.

    Parameters:
        input_path (string): a string path to directory containing '.records'

    Returns:
        filenames (string[]): a list of valid '.records' filenames
    """
    filenames = [input_path + name for name in os.listdir(input_path)
                 if name[-6:] == 'record']
    return filenames


def split_cases(split_filenames, split, balance=True, seed=None):
    """Split benign and malignant examples into train and validation sets

    Splits benign and malignant files into train and val sets, and if balance
    is true, rebalances those sets so that the number of benign cases is equal
    to the number of malignant cases.

    Parameters:
        benign (string[]): A list of '.record' filenames of benign
            examples.
        malignant (string[]): A list of '.record' filenames of
            malignant examples.
        split (float): Percentage of data to be training data. Remainder will
            be validation data.
        balance (bool): Bool that represents whether to rebalance data so that
            the number of benign cases equals the number of malignant.
        seed (int): (Optional.) A seed to pass to sklearn train_test_split,
            to allow for reproducing train test splits; default behavior is to
            allow for train_test_split to generate random seed to split data.

    Returns:
        train_filenames (string[]): A list of filenames for training.
        val_filenames (string[]): A list of filenames for validation.
    """
    for key in split_filenames:
        split_filenames[key] = train_test_split(
            split_filenames[key],
            test_size=(1-split),
            random_state=seed
        )

    if balance:
        train_split_filenames = {k: v[0] for k, v in split_filenames.items()}
        train_filenames = balance_cases(train_split_filenames)
        val_split_filenames = {k: v[1] for k, v in split_filenames.items()}
        val_filenames = balance_cases(val_split_filenames)
    else:
        train_filenames = [example
                           for class_list in split_filenames.values()
                           for example in class_list[0]]

        val_filenames = [example
                         for class_list in split_filenames.values()
                         for example in class_list[1]]

    return train_filenames, val_filenames


def balance_cases(split_filenames):
    """ Creates a numerically balanced list of files by class

    Parameters:
        split_filenames (dict of str: list): A dictionary where each key is a
            is a unique class, and each value is a list of filenames of that
            class.
    Returns:
        filename_set (string[]): A list of '.record' filenames numerically
        balanced by class.
    """
    largest = 0
    for key, value in split_filenames.items():
        largest = max(largest, len(value))

    filename_set = []
    for key, value in split_filenames.items():
        factor = largest // len(value)
        remainder = largest // len(value)
        filename_set += value * factor + value[:remainder]

    return filename_set


def separate_by_diagnosis(filenames, flags, flag_pos):
    """ Split list of '.record' filenames into a dict of lists of diff classes.

    Splits list of filenames into a dictionary of flag: class_filename_list.

    Parameters:
        filenames (string[]): A list of '.record' filenames, with class flag
            in position [-10].
        flags (string[]): A list of single character strings, corresponding
            with the different classes present in the dataset.
        flag_pos (int): The position of the flag in the filename (0 index).

    Returns:
        split_filenames (dict of str: list): A dictionary where each key is a
            flag from flags, and each value is a list of filenames of class
            flag.
    """
    split_filenames = {}
    for flag in flags:
        split_filenames[flag] = []

    for name in filenames:
        try:
            split_filenames[name[flag_pos]].append(name)
        except KeyError:
            print(name + " does not have a valid class flag at " +
                  str(flag_pos))
        except IndexError:
            print("Index out of range")
            print(name)
    return split_filenames


def separate_by_patient(filenames, pos=[64, 67]):
    """ Split list of '.record' filenames into a dict of lists of diff patients

    Splits list of filenames into a dictionary of patient_id: patient_files.
    Splits based on patient id (pid) in filename between pos[0] and pos[2].

    Parameters:
        filenames (string[]): A list of '.record' filenames, with patient pid
            between pos[0] and pos[1].
        pos (int[2]): A length 2 list with the 0th index corresponding to
            the beginning index of the pid in the filename, and the 1st index
            corresponding to the ending index of the pid in the filename.
    Returns:
        patient_dict (dict of int: string[]): A dictinary mapping pids to
            corresponding '.record' filenames.
    """
    patient_dict = {}
    for name in filenames:
        pid = ''.join(ch for ch in name[pos[0]:pos[1]] if ch.isdigit())
        if pid in patient_dict:
            patient_dict[pid].append(name)
        else:
            patient_dict[pid] = [name]
    return patient_dict


def split_patient_dict(patients, split, seed=None):
    """Splits a dict containing filenames split by patient into train and val.

    Parameters:
        patients (dict of str: string[]): A dictinary mapping pids to
            corresponding '.record' filenames.
        split (float): Percentage of data to be training data. Remainder will
            be validation data.
        seed (int): (Optional.) A seed to pass to sklearn train_test_split,
            to allow for reproducing train test splits; default behavior is to
            allow for train_test_split to generate random seed to split data.

    Returns:
        train_patients (dict of str: string[]): A dictinary mapping pids to
            corresponding '.record' filenames.
        val_patients (dict of str: string[]): A dictinary mapping pids to
            corresponding '.record' filenames.
    """
    def split_by_key(pids):
        return {pid: patients[pid] for pid in pids}
    train_pids, val_pids = train_test_split(
        [pid for pid in patients],
        test_size=(1-split),
        random_state=seed
    )
    train_patients = split_by_key(train_pids)
    val_patients = split_by_key(val_pids)
    return train_patients, val_patients


def make_patient_dataset(patients, aug):
    """Make a dataset batched by patient.

    Parameters:
        patients (dict of str: str[]): A dictionary mapping patient ids to
            '.record' filenames.
        aug (bool): A bool that represents whether to preform data-augmenation
            on the data.

    Returns:
        dataset (tf.data.Dataset): A tf Dataset, batched by patient.
    """
    dataset = None
    for patient in patients.values():
        patient = create_dataset(patient, len(patient), aug, False)
        if dataset is None:
            dataset = patient
        else:
            dataset = dataset.concatenate(patient)
    return dataset


def get_dataset(input_path,
                split,
                epochs,
                include_z_array,
                sort_by_patient,
                flags,
                flag_pos_1,
                flag_pos_2=None,
                batch_size=None,
                seed=None,
                aug_train=True,
                aug_val=False,
                rebalance=True):
    """Gets a tf.data.Dataset from '.record' files in input_path directory.

    Creates a tf.data.Dataset from '.record' files in the input_path directory.
    The dataset can be data-augmented, batched by patient, and class rebalanced
    If split into training and validation sets, the validation set is
    concatenated onto the end of the training set.

    Parameters:
        input_path (str): String path to data directory.
        split (float): Percentage of data to be training data. Remainder will
            be validation data.
        epochs (int): Number of times to iterate through the data.
        include_z_array (bool): A bool that represents whether data has a
            z-score encoded layer for differential recognition.
        sort_by_patient (bool): A bool that represents whether to sort and
            batch the data by patient.
        flags (str[]): A list of flags to look for in filenames.
        flag_pos_1 (int): The index of the first (or only) character of the
            class flag.
        flag_pos_2 (int): The index of the last character of the class flag.
        batch_size (int): Number of examples to run in parralel through model.
        seed (int): (Optional.) A seed to pass to sklearn train_test_split,
            to allow for reproducing train test splits; default behavior is to
            allow for train_test_split to generate random seed to split data.
        aug_train (bool): A bool that represents whether to perform data
            augmentation on the training set.
        aug_val (bool): A bool that represents whether to perform data
            augmentation on the validation set.
        rebalance (bool): A bool that represents whether to rebalance the data
            so that every classified-class has an equal number of training
            examples.

    Returns:
        dataset (tf.data.Dataset): A dataset created from '.record's files in
            the input_path directory.
        dataset_length(int[]): A length 2 list, where the first position
            corresponds to the number of examples in the training set,
            and the second position corresponds to the number of examples in
            the validation set.
    """
    filenames = collect_filenames(input_path)
    print("There are {} examples in {}".format(len(filenames), input_path))
    dataset_length = [0, 0]

    if sort_by_patient:
        patients = separate_by_patient(filenames, [flag_pos_1, flag_pos_2])
        for patient in patients.values():
            dataset_length[0] += len(patient)
        dataset = make_patient_dataset(patients, aug_train)
    else:
        classes = separate_by_diagnosis(
            filenames,
            flags,
            flag_pos_1
        )
        train_set, val_set = split_cases(classes, split, rebalance, seed)

        dataset_length[0] = len(train_set)
        dataset_length[1] = len(val_set)

        train_dataset = create_dataset(
            train_set,
            batch_size,
            aug_train,
            include_z_array,
            True
        )

        val_dataset = create_dataset(
            val_set,
            batch_size,
            aug_val,
            include_z_array,
            True
        )
        dataset = train_dataset.concatenate(val_dataset)

    return dataset.prefetch(1).repeat(epochs), dataset_length
