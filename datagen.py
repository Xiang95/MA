import os
import kitti_util
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

root = '/home/xiang/MA/dataset/kitti'

transform = transforms.Compose([
    transforms.ToPILImage(),  # only works on PILImage
    transforms.CenterCrop((370, 1224)),
])


class KittiDataset(Dataset):
    def __init__(self, dataset_path=root, split='training'):
        self.samples = []
        self.split = split
        self.data_path = dataset_path
        # self.kitti_data = KittiDataset(data_path=dataset_path, json_path=dataset_path + 'data', verbose=True)

        self.data_path_image = os.path.join(self.data_path, 'image_2', split, 'image_2')
        self.data_path_pc = os.path.join(self.data_path, 'velodyne', split, 'velodyne')
        self.data_path_calib =os.path.join(self.data_path, 'calib', split, 'calib')

        self.sample_size = len(self.data_path_image)
        self.transform = transforms.Compose([

            transforms.ToTensor(),
        ])

    def __len__(self):
        # total number of samples
        return self.sample_size

    def __getitem__(self, item):
        sample_idx = item
        # sample = self.kitti_data.sample[sample_idx]
        velo_pc = kitti_util.load_velo_scan(os.path.join(self.data_path_pc,
                                                         sorted(os.listdir(self.data_path_pc))[item]))
        # img are numpy array so transform must add ToPIL
        img = kitti_util.load_image(os.path.join(self.data_path_image, sorted(os.listdir(self.data_path_image))[item]))
        calibration_path = os.path.join(self.data_path_calib, sorted(os.listdir(self.data_path_calib))[item])
        calibration = kitti_util.Calibration(calibration_path)
        img = transform(img)
        sample = {'image': img, 'velo_pc': velo_pc, 'calibration': calibration}

        return sample


sample = KittiDataset()

dataloader = DataLoader(sample, batch_size=4, shuffle=False, num_workers=4)

# print(sample[-1]['velo_pc'].shape)