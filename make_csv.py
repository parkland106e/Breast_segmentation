import pandas as pd
import glob
import os

def check_path(root_dir):


    check_list = []
    case_list = [path.split('\\')[-1] for path in glob.glob(root_dir+'/*')]
    #print(case_list)

    for case in case_list:
        path = root_dir+'/'+case+'/Resample_Breast Mask'
        #print(path)
        if not os.path.exists(path):
            check_list.append(case)


    return check_list

if __name__ == '__main__':

    train_root = 'E:/dataset/CR_Resection/CR_Resection/Train'
    test_root = 'E:/dataset/CR_Resection/CR_Resection/Test'
    tuning_root = 'E:/dataset/CR_Resection/CR_Resection/Tuning'

    train_no_breast = check_path(train_root)
    test_no_breast = check_path(test_root)
    tuning_no_breast = check_path(tuning_root)

    train_table = pd.DataFrame({'ID': train_no_breast})
    train_table.to_csv('E:/dataset/CR_Resection/mr_resection_meta/train_no_breast.csv')

    test_table = pd.DataFrame({'ID': test_no_breast})
    test_table.to_csv('E:/dataset/CR_Resection/mr_resection_meta/test_no_breast.csv')

    tuning_table = pd.DataFrame({'ID': tuning_no_breast})
    tuning_table.to_csv('E:/dataset/CR_Resection/mr_resection_meta/tuning_no_breast.csv')