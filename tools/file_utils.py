import os


def make_dir(file_path):
    if os.path.exists(file_path):
        return file_path
    os.makedirs(file_path)
    return file_path


def is_dir_empty(file_dir):
    if os.path.isfile(file_dir):
        raise Exception('{} is not a directory'.format(file_dir))
    files = os.listdir(file_dir)
    return len(files) == 0


def check_is_csv(file_path):
    return file_path.lower().endswith('.csv')
