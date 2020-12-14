import os
from datetime import datetime
from pathlib import Path

project_root = Path.cwd()


def _get_result_path():
    print()
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    result_root = project_root / 'results' / dt_string
    return result_root


result_root_path = _get_result_path()
visualization_path = result_root_path / 'visualization'
data_path = project_root / 'data'


def create_folder():
    folders = [visualization_path, data_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__':
    print(project_root)
