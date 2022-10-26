import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
import torch.nn as nn
from dataset import *
import numpy as np

SEED = 3141592653

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.000025,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def making_axial_cube(preds, size, file_length):
    cube = np.zeros(size)
    for i in range(file_length):
        temp = preds[i].squeeze(dim=0).squeeze(dim=0)
        if temp.shape != (512, 512):
            print(temp.shape, i)
        cube[i, :, :] = temp.cpu().detach().numpy()

    return cube


def get_dice_score(output, target, epsilon=1e-9):
    SPATIAL_DIMENSIONS = 2, 3, 4

    intersection = (output * target).sum(dim=SPATIAL_DIMENSIONS)
    union = output.sum(dim=SPATIAL_DIMENSIONS) + target.sum(dim=SPATIAL_DIMENSIONS)

    dice_score = 2 * (intersection + epsilon) / (union + epsilon)

    return dice_score


def making_cube(axis):
    print('-----------------------------------------------------')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    print('학습을 진행하는 기기:', device)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())
    print('-----------------------------------------------------')

    SEED = 3141592653
    torch.manual_seed(SEED)

    if axis == 0:
        test_dataset = MRI_T1_whole_seg_Dataset('D:/seg_complete/test_seg_100_axial_whole_breast.csv',
                                                'D:/seg_complete/seg_complete_100', transform=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
        model_name = 'C:/Users/seohyegyo/Desktop/Breast_tumor_1000_100/saved_models/batch16_epoch150_Axial.pth'

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b2", encoder_weights='imagenet', in_channels=1, classes=1,
                             activation="sigmoid")
    model = model.to(device)
    model.load_state_dict(torch.load(model_name), strict=False)

    n_test = test_loader.__len__()
    print(axis)
    print(f'n_test_len:{n_test}')
    cnt = 0

    preds = []
    masks = []
    cube_list = []

    if axis == 0:
        model.eval()
        with torch.no_grad():
            for test_imgs, test_true_masks, id in test_loader:
                test_imgs = test_imgs.to(device, dtype=torch.float32)
                test_true_masks = test_true_masks.to(device, dtype=torch.float32)
                cnt += 1

                test_mask_pred = model(test_imgs)
                file_length = len(os.listdir('D:/seg_complete/seg_complete_100/' + id + '/img'))
                shape = (file_length, 512, 512)
                id = str(id).split("'")[1]

                # threshold값
                test_pred = (test_mask_pred > 0.5).float()

                preds.append(test_pred)
                masks.append(test_true_masks)

                # 환자 한명분이 끝나면 axial cube만들기
                if cnt % file_length == 0:
                    print(axis, id, cnt)
                    pred_for_dice = making_axial_cube(preds, shape, file_length)
                    mask_for_dice = making_axial_cube(masks, shape, file_length)
                    cube_list.append(pred_for_dice)

                    preds = []
                    masks = []
                    cnt = 0
                    tot = 0

    return cube_list


def flip_array(x, direction):
    output = x.copy()
    direction = np.array(direction)
    direction = np.flip(direction[direction != 0])
    for i, is_fliped in enumerate(direction):
        if is_fliped == -1:
            output = np.flip(output, i)
    return output


def save_concat_image(load_path, axial_cube, save_path, p):
    p = p
    binary_list = load_path

    for i in range(len(binary_list)):

        axial_image = axial_cube[i, :, :]
        save_file_name = binary_list[i].split('\\')[-1]

        mask_info = pydicom.dcmread((binary_list[i]))

        image = axial_image
        if mask_info.BitsAllocated == 16:
            image = (image * 255).astype(np.uint16)
        else:
            image = (image * 255).astype(np.uint8)

        mask_info.PixelData = image.tobytes()
        mask_info.save_as(save_path + '/' + save_file_name)

def save_image(default_dir,id_csv,model_path,save_path=False):

    dataset = Sub1_3D_dataset(default_dir,id_csv)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b2", encoder_weights='imagenet', in_channels=1, classes=1,
                             activation="sigmoid").to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    with torch.no_grad():

        for batch_index, (mri_images, id) in enumerate(loader):
            print(id)
            for i in range(len(mri_images)):
                single_output = model(mri_images[i].to(device,dtype=torch.float32))
                print(single_output.shape,type(single_output))
                single_output = single_output.detach().cpu().numpy()
                print(single_output.shape, type(single_output))



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(device)
    logging.info(f'Using device {device}')

    model_path = './batch16_epoch150_Axial.pth'

    save_image('E:/dataset/CR_Resection/CR_Resection/Train','E:/dataset/CR_Resection/mr_resection_meta/train_no_breast.csv',model_path)

    '''
    axial_cube = making_cube(0)
    # test하기 원하는 환자 id적은 후 
    patient_name = ['4475916', '3199796', '3338587', '3697083', '5887889', '8598800', '5984427', '8905112', '9998510',
                    '1329307', '1359460', '4879237', '9699790']

    # file_path = sorted([path for path in glob.glob('F:/CR_resection_p1/*/mri_t1_post_sub_1_breast/*.dcm')])

    # test하기 원하는 환자 명수만큼 range바꿈
    for i in range(0, 13):
        load_path = sorted(
            [path for path in glob.glob('F:/CR_resection_p1/' + patient_name[i] + '/mri_t1_post_sub_1_breast/*.dcm')])
        # test dicom 저장하고 싶은 곳 주소 적기
        save_path = 'F:/seg1000transfer_test_image/' + patient_name[i]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_concat_image(load_path, axial_cube[i], save_path, i)
    '''