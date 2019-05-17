import os
import sys


def check_data_path(file_name):
    data_path = os.path.join(os.getcwd(), file_name)
    if not os.path.exists(data_path):
        print("Error: file hasn't included. Please find the dataset and retry")
        sys.exit()
    return data_path


def create_result_preprocessing():
    preprocessing_result_path = os.path.join(os.getcwd(), "preprocessing_data")
    if not os.path.isdir(preprocessing_result_path):
        print("Creating ...")
        os.makedirs(preprocessing_result_path)
    return preprocessing_result_path


def write_file(path, data):
    with open(path, "w+") as f:
        for i in sorted(data):
            f.write(i+"\n")
        f.close()


def file_to_set(file_name):
    results = set()
    with open(file_name, "r+") as f:
        for line in f:
            results.add(line.replace('\n', ''))
    return results
