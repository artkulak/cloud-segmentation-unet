import zipfile as zf
import numpy as np
import os
import cv2
import scipy.misc

PATH = 'D:/landsat8/'


def extract_images_random(bands, mask, number, size = (250, 250)):
    minibands = np.empty((number, size[0], size[1], bands.shape[2]))
    masks = np.empty((number,size[0], size[1], 1))
    for k in range(number):
        while True:
            # check for correctness
            i = np.random.randint(0, bands.shape[0] - size[0])
            j = np.random.randint(0,bands.shape[1] - size[1])
            minibands[k, :, :, :] = bands[i: i + size[0], j : j + size[1], :]/255
            masks[k] = mask[i: i + size[0], j : j + size[1], :]/255
            if len(np.unique(masks[k])) > 1:
                break
    return minibands, masks


if __name__ == '__main__':
    nums = 10
    size = (256, 256)
    band_n = 7  # change band numb
    total = nums * 2
    minibands = np.empty((total, size[0], size[1], band_n))
    target = np.empty((total, size[0], size[1], 1))
    k = 0
    index = 0
    for value in os.listdir(PATH):
        if value == '8' or value == '7':
            path = PATH + value + '/' + value + '/'
            mask = cv2.imread(path + 'QB.tif')
            mask = mask[:, :, 1].reshape(mask.shape[0], mask.shape[1], 1)
            bnds = np.empty((mask.shape[0], mask.shape[1], band_n))
            for value in os.listdir(path):
                if value.split('.')[0][0] == 'B':
                    n = int(value.split('.')[0][1:])
                    if n < 8:
                        bnd = cv2.imread(path + value)[:, :, 0]
                        bnds[:, :, n - 1] = bnd
                        k += 1
            k = 0
            mnbands, msk = extract_images_random(bnds, mask, nums, size=size)
            minibands[index * nums: (index + 1) * nums, :, :, :] = mnbands
            target[index * nums: (index + 1) * nums, :, :, :] = msk
            index += 1

    save_path = PATH = 'D:/landsat8/' + 'valdata/'
    for i in range(total):
        scipy.misc.imsave(save_path + f'QB.tif', target[i].reshape((256, 256)))
        for k in range(band_n):
            scipy.misc.imsave(save_path + f'B{k + 1}.tif', minibands[i, :, :, k])

        with zf.ZipFile(save_path + f'val_data_{i}.zip', 'w') as zp:
            for file in os.listdir(save_path):
                if file.split('.')[1] == 'tif':
                    zp.write(save_path + file)
