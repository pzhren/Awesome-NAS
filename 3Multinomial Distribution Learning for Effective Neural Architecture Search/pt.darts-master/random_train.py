import os
import subprocess

def find_training_ge(search_sub_folders, search_folder, train_folder):
    trained_ge = []
    for search_sub_folder in search_sub_folders:
        for i in range(16):
            search_path = os.path.join(search_folder, search_sub_folder, str(i))
            train_path = os.path.join(train_folder, search_sub_folder, str(i))
            file = open(os.path.join(search_path, '{0}.log'.format(search_sub_folder)))
            lines = file.readlines()
            line_str = lines[-1]
            ge_str = line_str.split(':')[-1]
            if os.path.exists(train_path):
                # get ge in search_path
                if 'Finalbestge' in line_str.replace(" ", ""):
                    trained_ge.append(ge_str)
                continue
            else:
                if ge_str in trained_ge:
                    continue
                else:
                    return train_path, ge_str


if __name__ == '__main__':
    search_sub_folders = ['random_50', 'random_50_layer_8']
    search_folder = '/userhome/project/pt.darts/experiment/random_search'
    train_folder = '/userhome/project/pt.darts/experiment/random_train'
    find_sub_folder, find_ge = find_training_ge(search_sub_folders, search_folder, train_folder)
    command = 'python /userhome/project/pt.darts/augment.py --save_path {0} --name {1} --genotype "{2}" --save_dir {3}'.\
        format(find_sub_folder.split('/')[-1], find_sub_folder.split('/')[-2], find_ge, train_folder)
    print("training start")
    subprocess.call(command, shell=True)