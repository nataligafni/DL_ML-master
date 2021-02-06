import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import os
import glob
import skimage
import skimage.transform
import cv2 as cv
from skimage import io
# #ploting images
# path=r'C:\Users\MorBen-Nun\Documents\University\Deep_learning_medical\Data\segmentation2_100'
# total_sices=[]
# folders=os.listdir(path)
# i=1
# l=1
# file_name=glob.glob(path+'\*.nii')
# for k in file_name:
#     path1=os.path.join(path,k)
#     # name ='coronacases_001.nii'
#     # full_path = path1 + '\\' + name
#     img = nib.load(path1)
#     a = np.array(img.dataobj)
#     image1 = a[:, :,6]
#     plt.subplot(1,3,i)  # Address proper subplot in 2x2 array
#     plt.imshow(image1,cmap='gray')
#     i=i+1
#     l=l+1
# plt.show()


name='tr_im_20.nii'
path=r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\Segmentation\Image'

name='tr_mask_merged_13.nii'
path=r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\Segmentation\Mask'

full_path = path + '\\' + name
img = nib.load(full_path)
pixels = np.array(img.dataobj)
pixels =np.rot90(pixels,3)

X_data_resized = skimage.transform.resize(pixels,(256,256))
plt.imshow(X_data_resized,cmap='gray')
plt.show()

print(np.where(a==1))

path = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\CT_NonCOVID'
pixels = cv.imread(path + '\\' + '0.jpg')


pixels = pixels.astype('float32')
# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
# global standardization of pixels
pixels = (pixels - mean) / std


a = pixels.astype('float32')
plt.imshow(pixels)
plt.show()


# #checking the new array
# path=r'C:\Users\MorBen-Nun\Documents\University\Deep_learning_medical\Data\segmentation2_100'
# folder='tr_mask_merged.nii'
# path1=os.path.join(path,folder)
# img = nib.load(path1)
# a = np.array(img.dataobj)
# for i in range(0,99):
#     image1 = a[:, :,99]
#     plt.imshow(image1,cmap='gray')
#     plt.show()
# for i in range(1,2):
#     print(i)
# def Exctracting_information_image(file_name):
#     ind = [y.start() for y in re.finditer('_', file_name)]
#     ind2 = [p.start() for p in re.finditer('.bmp', file_name)]
#     Short_name = file_name[0:ind[len(ind) - 1]]
#     return Short_name

# saving image

Main_Path=r'C:\Users\MorBen-Nun\Documents\University\Deep_learning_medical\Data\segmentation2_100'
file_names=glob.glob(Main_Path+'\*.nii')
Path=r'C:\Users\MorBen-Nun\Documents\University\Deep_learning_medical\Data\Segmentation'
if not os.path.exists(Path):
    os.makedir(Path)

for i in file_names:
    if i=='tr_im.nii':
        full_path =Path+ '\\Image'
    else:
        full_path = Path + '\\Mask'
    if not os.path.exists(full_path ):
          os.makedirs(full_path)

    path1=os.path.join(Main_Path,i)
# name ='coronacases_001.nii'
# full_path = path1 + '\\' + name
    img = nib.load(path1)
    a = np.array(img.dataobj)
    for k in range(0,a.shape[2]):
        image1 = a[:, :,k]
        ni_img=nib.Nifti1Image(image1,img.affine)
        img_name=i[:-4]+ '_' +str(k)+'.nii'
        nib.save(ni_img,img_name)


#          plt.subplot(1,3,i)  # Address proper subplot in 2x2 array
#      plt.title(k)
#     plt.imshow(image1,cmap='gray')
#     i=i+1
# plt.show()
# # plt.subplot()
# #     total_sices.append(img.shape[2])
# #
# #     image1 = a[:, :, 0]
# #     plt.imshow(image1, cmap='gray')
# #     plt.show()
# #
    
    
    
    
#
# for i in range(1,9):
#     name=str(i)+'.nii'
#     full_path=path+ '\\' + name
#     img=nib.load(full_path)
#     a=np.array(img.dataobj)
#     total_sices.append(img.shape[2])
# print(sum(total_sices))
#
# # for i in range(0,45):
# #     image1=a[:,:,i]
# #     label=np.unique(image1)
# #     if len(label)==1:
# #          print('Healty')
# #          print(i)
# # #image1=img.get_fdata()[0]
#
# image1=a[:,:,0]
# plt.imshow(image1,cmap='gray')
# plt.show()
#
