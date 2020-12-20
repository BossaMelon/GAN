import getpass
import os
from datetime import datetime
from pathlib import Path

current_username = getpass.getuser()

if current_username == 'wyh':
    # project_root = Path.cwd()
    resource_root = Path('/Users/wyh/Documents/Project/cousera/pytorch_implementation/GAN/resources')
    data_path = resource_root / 'data'
    model_path = resource_root / 'trained_models'


elif current_username == 'wangyueh':
    resource_root = Path('/home/wangyueh/projects/GAN/resources')
    data_path = '/phys/ssd/wangyueh/GAN/data'
    model_path = '/phys/ssd/wangyueh/GAN/trained_models'

else:
    raise Exception('no valid data path')


def _get_result_path():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    result_root = resource_root / 'results' / dt_string
    return result_root


result_root_path = _get_result_path()
visualization_path = result_root_path / 'visualization'


def create_folder():
    folders = [visualization_path, data_path, model_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


create_folder()

if __name__ == '__main__':
    print(resource_root)
