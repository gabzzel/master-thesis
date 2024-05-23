import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class S3DISDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='trainval_fullarea',
                 num_point=4096,
                 test_area=5,
                 block_size=1.0,
                 sample_rate=1.0,
                 transform=None,
                 include_colors=True,
                 includes_normals=False,
                 apply_coordinate_normalization=True,
                 use_original_indexer=True,
                 apply_z_centering=False):
        """
        The dataset loader specifically made for the S3DIS dataset.

        Parameters:
            split: Whether to train or to test/evaluate
            data_root: The path to the directory where all the .npy data files are stored
            num_point: The number of points per batch entry
            test_area: Which area to use for testing/evaluation
            block_size: The size of the blocks used to divide the area into batches (each block will be divided into batches)
            sample_rate: ???
            include_colors: Whether to include colors during data feeding
            includes_normals: Whether the data includes normals and that data needs to be read
            apply_coordinate_normalization: Whether to add a data augmentation step that adds coordinates normalized
                to the room min and max
            apply_z_centering: If True, uses the centroid of a block of chosen points to normalizes the Z coordinate
                of the chosen points as well as the X and Y. If False, only normalized the X and Y this way.
                Warning! Setting this to True can interfere with other data augmentation!
            use_original_indexer: Whether to use the original __get_item__ method or the new one.
        """

        super().__init__()

        self.use_original_indexer = use_original_indexer
        self.apply_coordinate_normalization = apply_coordinate_normalization
        self.includes_normals = includes_normals
        self.include_colors = include_colors
        self.apply_z_centering = apply_z_centering

        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7

            # If we have normals, we take until 9 (xyz, rgb, normals), else until 6 (xyz, rgb)
            point_data_max_index = 6 if not includes_normals else 9
            points = room_data[:, :point_data_max_index]
            labels = room_data[:, point_data_max_index]
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp

            self.room_points.append(points)
            self.room_labels.append(labels)

            coord_min = np.amin(points[:, :3], axis=0)
            coord_max = np.amax(points[:, :3], axis=0)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)

            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        if self.use_original_indexer:
            return self.get_item_original(idx=idx)

        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                            points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6 (coords + colors) or numpoint * 9 (coords, colors and normals)

        point_feature_number = 0
        if self.include_colors:
            point_feature_number += 3
        if self.includes_normals:
            point_feature_number += 3
        if self.apply_coordinate_normalization:
            point_feature_number += 3

        current_points = np.zeros((self.num_point, 3 + point_feature_number))

        # The point feature is always the point (coordinate) normalized by the chosen center
        current_points[:, 0] = selected_points[:, 0] - center[0]
        current_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 2] = selected_points[:, 2] - center[2] if self.apply_z_centering else selected_points[:, 2]

        occupied_indices = 3

        # Add the colors and normalize them
        if self.include_colors:
            current_points[:, occupied_indices:occupied_indices + 3] = selected_points[:, 3:6] / 255.0
            occupied_indices += 3

        # Add the data augmentation of adding points that are normalized by dividing by the room max coordinates
        # TODO fix this normalization so it uses the minimum coords as well, instead of only the max
        if self.apply_coordinate_normalization:
            for i in range(3):
                current_points[:, occupied_indices + i] = selected_points[:, i] / self.room_coord_max[room_idx][i]
            occupied_indices += 3

        # Include the normals
        if self.includes_normals:
            current_points[:, occupied_indices:occupied_indices+3] = selected_points[:, 6:9]
            occupied_indices += 3

        if self.includes_normals:
            current_points[:, 0:9] = selected_points
        else:
            current_points[:, 0:6] = selected_points

        current_labels = labels[selected_point_idxs]

        # Do not apply transform is we have normals.
        if self.transform is not None and not self.includes_normals:
            current_points, current_labels = self.transform(current_points, current_labels)

        # Current points explanation
        # 0 = x coordinate of selected points, normalized by subtracting the centroid of the selected points
        # 1 = y coordinate of selected points, normalized by subtracting the centroid of all selected points
        # 2 = z coordinate of selected points, normalized by subtracting the centroid of all selected points
        # 3, 4, 5 = normalized colors of selected points
        # 6, 7, 8 = xyz coordinates of selected points normalized by dividing by the room size (i.e. max coords)
        # 9, 10, 11 = normals xyz
        return current_points, current_labels

    def get_item_original(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) == -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) != -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                                points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


def run_as_main():
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    point_data = S3DISDataset(split='train', data_root=data_root, num_point=4096, test_area=5, block_size=1.0,
                              sample_rate=0.01, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()


if __name__ == '__main__':
    run_as_main()
