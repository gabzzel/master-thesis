import numpy as np
import tqdm


def convert_to_batches(points: np.ndarray,
                       colors: np.ndarray = None,
                       normals: np.ndarray = None,
                       block_size: float = 0.5,
                       stride: float = 0.1,
                       padding: float = 0.001,
                       point_amount: int = 4096,
                       batch_size: int = 32):
    """
    Convert point cloud data into batches that can be fed into the pointnet++ network.

    Parameters:
        points (np.ndarray): point cloud coords to be converted to batches. Must be of shape (n_points, 3).
        colors (np.ndarray): colors of points. Must be of shape (n_points, 3).
        normals (np.ndarray): normals of points. Must be of shape (n_points, 3).
        block_size (float): the size of the 'blocks' that will be used to group points in the batches. Only applies
            to x and y dimensions. For example, of value of 1.0 will group points in 'pillars' of 1x1 meters.
        padding (float): the amount of padding to be added to each block to avoid missing points on the edge.
        stride (float): the stride size of the blocks, convolution style. This makes sure the blocks overlap and
            points are in multiple blocks / batches.
        point_amount (int): the minimum amount of points in per batch entry. If there are more points in the block than
            the points amount, the batch element will just be larger. If there are fewer points in the block than the
            point amount, random choice with replacement will be used to fill the gaps.
        batch_size: The size of each batch.
    """

    minimum_coordinates: np.ndarray = np.amin(points, axis=0)
    maximum_coordinates: np.ndarray = np.amax(points, axis=0)
    grid_x = int(np.ceil(float(maximum_coordinates[0] - minimum_coordinates[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(maximum_coordinates[1] - minimum_coordinates[1] - block_size) / stride) + 1)

    blocks_data = None
    indices = None

    print(f"Dividing point cloud into batches using (XY) blocks of size {block_size} and stride {stride}")
    pbar = tqdm.tqdm(total=grid_x * grid_y, desc="Dividing point cloud into blocks", unit="block")

    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            pbar.update(1)
            s_x: float = minimum_coordinates[0] + index_x * stride
            e_x: float = min(s_x + block_size, maximum_coordinates[0])
            s_x = e_x - block_size
            s_y: float = minimum_coordinates[1] + index_y * stride
            e_y: float = min(s_y + block_size, maximum_coordinates[1])
            s_y = e_y - block_size

            point_idxs: np.ndarray = np.where((points[:, 0] >= s_x - padding) &
                                              (points[:, 0] <= e_x + padding) &
                                              (points[:, 1] >= s_y - padding) &
                                              (points[:, 1] <= e_y + padding))[0]

            if point_idxs.size == 0:
                continue

            # The amount of arrays that we need to accommodate this block, i.e. the amount of arrays we need
            # to fit all the points in the block such that each array is of size 'point_amount x C'
            num_batch: int = int(np.ceil(point_idxs.size / point_amount))

            # The amount of total points we need in order to neatly fill each array in the batch
            point_size: int = int(num_batch * point_amount)

            # Whether we need to reuse points in order to neatly fill each array
            replace: bool = point_size - point_idxs.size > point_idxs.size

            # The repeated indices such that we neatly fill the arrays and every array is of size 'point amount x C'
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))

            # Shuffle the indices and create the data
            np.random.shuffle(point_idxs)

            block_data = np.zeros((len(point_idxs), 3))
            # First the "centered" points
            block_data[:, 0] = points[point_idxs, 0] - (s_x + block_size / 2.0)
            block_data[:, 1] = points[point_idxs, 1] - (s_y + block_size / 2.0)
            block_data[:, 2] = points[point_idxs, 2]

            # Add the colors
            occupied_indices = 3
            if colors is not None:
                relevant_color_data = colors[point_idxs]
                block_data = np.hstack((block_data, relevant_color_data))
                occupied_indices += 3

            # Add the normalized points
            for i in range(3):
                normalized_points = points[point_idxs, i] / maximum_coordinates[i]
                block_data = np.hstack((block_data, normalized_points[:, np.newaxis]))
            occupied_indices += 3

            if normals is not None:
                block_data = np.hstack((block_data, normals[point_idxs]))
                occupied_indices += 3

            blocks_data = np.vstack((blocks_data, block_data)) if blocks_data is not None else block_data
            indices = np.hstack((indices, point_idxs)) if indices is not None else point_idxs

    # All the blocks, neatly in a multiple of point amount
    blocks_data = blocks_data.reshape((-1, point_amount, blocks_data.shape[1]))
    indices = indices.reshape((-1, point_amount))

    return blocks_data, indices

    number_of_batches = int(np.ceil(blocks_data.shape[0] / block_size))
    batches = []
    batches_indices = []
    for i in tqdm.trange(number_of_batches, desc="Generating batches out of the blocks..."):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, blocks_data.shape[0])
        batches.append(blocks_data[start_index:end_index])
        batches_indices.append(indices[start_index:end_index])

    return batches, batches_indices
