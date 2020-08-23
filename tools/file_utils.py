import os


def make_dir(file_path: str):
    os.makedirs(file_path, exist_ok=True)
