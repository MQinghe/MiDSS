from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import matplotlib.pyplot as plt

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/Fundus',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(i), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')

            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((256, 256), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((256, 256), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            self.image_pool.append(
                Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)
            self.img_domain_code_pool.append(self.image_list[index]['domain_code'])



    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(self.splitid) + ')'

class ProstateSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/ProstateSlice',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4,5,6],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                print('target rgb')
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'Prostate(phase=' + self.phase+str(self.splitid) + ')'

class MNMSSegmentation(Dataset):
    """
    MNMS segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/MNMS/mnms',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'vendorA', 2:'vendorB', 3:'vendorC', 4:'vendorD'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((288, 288), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288, 288), Image.NEAREST)
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).resize((288, 288), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288,288), Image.NEAREST)
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'MNMS(phase=' + self.phase+str(self.splitid) + ')'

if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        # tr.RandomHorizontalFlip(),
        # tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(#split='train1',
    splitid=[1,2],lb_domain=2,lb_ratio=0.2,
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)
    print(len(dataloader))
    for ii, sample in enumerate(dataloader):
        # print(sample)
        exit(0)
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)
