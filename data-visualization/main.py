from typing import List
import matplotlib.pyplot as plt
import numpy as np
import open3d


def plot_aspect_ratios():
    data_paths: List[str] = [
        "E:\\thesis-results\\office\\chosen\\alpha-shapes.npz",
        "E:\\thesis-results\\office\\chosen\\bpa.npz",
        #"E:\\thesis-results\\chosen\\delaunay.npz",
        "E:\\thesis-results\\office\\chosen\\spsr.npz"
    ]
    arrays = []
    for data_path in data_paths:
        data_raw = np.load(data_path)
        arrays.append(data_raw['aspect_ratios'])
    means = [np.mean(a) for a in arrays]
    labels = ["Alpha Shapes", "BPA", "SPSR"]
    # bins = 100
    bins = np.arange(start=0, stop=1000, step=0.1).tolist()
    #plt.hist(arrays, histtype='step', log=True, label=labels, bins=bins)
    # plt.ylim(ymin=1)
    fig, ax = plt.subplots()
    ax.set_xlabel("Aspect ratio")
    ax.set_ylabel("Number of triangles")
    # ax.set_title("Aspect ratio distribution of Training Complex dataset reconstructions")
    ax.set_xscale('log')
    ax.set_xlim(xmax=1000)
    n, bins, patches = ax.hist(arrays, histtype='step', log=True, bins=bins)
    #bin_indexes_with_most_entries = np.argmax(n, axis=1)
    plt.legend(labels=labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_size_inches(10, 5)
    fig.dpi = 300
    # fig = plt.figure(figsize=(10, 5), dpi=300)
    # fig.savefig("aspect-ratios-training-complex.png", dpi=300)
    fig.show()


def show_meshes():
    paths = [
        "E:\\thesis-results\\delaunay-1\\1711352017.4267113_run_0\\enfsi-2023_reduced_cloud_mesh.ply",
        # "E:\\thesis-results\\delaunay-1\\1711358023.1019151_run_2\\enfsi-2023_reduced_cloud_mesh.ply"
    ]

    meshes = [open3d.io.read_triangle_mesh(path) for path in paths]
    open3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)


def color_points_by_distances():
    pcd_path = "E:\\etvr_datasets\\ruimte_ETVR_downsampled.ply"
    print(f"Loading point cloud {pcd_path}")
    pcd = open3d.io.read_point_cloud(pcd_path)

    print("Loading distances...")
    results_path = "E:\\thesis-results\\office\\bpa-1\\1712736109.978807_run_0\\raw_results.npz"
    distances: np.ndarray = np.load(results_path)["distances"]

    print("Colouring points by distances...")
    max_distance = distances.max()
    distances = distances / max_distance
    colors = np.zeros(shape=(len(distances), 3))
    colors[:, 0] = distances / max_distance
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([pcd])


def plot_computation_times():
    # Sample data
    datasets = ['Training Complex', 'Office', 'Zuidberg']
    algorithms = ['AS', 'BPA', 'SPSR']

    # index 0,1,2 are Alpha Shapes, BPA and SPSR on Training complex
    # index 3,4,5 are algorithms on office
    # etc
    labels = ['TC - A', 'TC - BPA', 'TC - SPSR',
              'Office - A', 'Office - BPA', 'Office - SPSR',
              'Zuidberg - A', 'Zuidberg - BPA', 'Zuidberg - SPSR']
    loading_times = np.array([3145, 75420, 75170, 654849, 0, 1125012, 0, 0, 0])
    reconstruction_times = np.array([59283, 28946, 66288, 8499369, 0, 235038, 0, 0, 0])
    cleaning_times = np.array([2820, 5145, 6423, 308765, 0, 26101, 0, 0, 0])

    bar_width = 0.5  # Adjust bar width
    bar_spacing = 0.05  # Adjust spacing between bars
    dataset_width = 2 * bar_spacing + 3 * bar_width  # How wide the dataset groups are.
    dataset_margin = 0.5  # The margin between datasets in the plot, between each other and between them and the edges
    total_width = 3 * dataset_width + 4 * dataset_margin  # The total width of the plot.

    fig, ax = plt.subplots()
    bar_colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, dataset in enumerate(datasets):
        dataset_index = i*3

        # The start x position of the first bar in this dataset
        dataset_offset = (i + 1) * dataset_margin + i * dataset_width

        for j, algorithm in enumerate(algorithms):
            index = dataset_index + j

            x_position = dataset_offset + j * (bar_width + bar_spacing)
            loading_time = loading_times[index]
            ax.bar(x_position, loading_time, width=bar_width, color=bar_colors[0], align='edge')

            reconstruction_time = reconstruction_times[index]
            ax.bar(x_position, reconstruction_time, width=bar_width, bottom=loading_time, color=bar_colors[1], align='edge')

            cleaning_time = cleaning_times[index]
            p = ax.bar(x_position, cleaning_time, width=bar_width, bottom=loading_time+reconstruction_time, color=bar_colors[2], align='edge')
            ax.bar_label(p, label_type='edge', labels=[algorithm])

    x_ticks = np.zeros(shape=(3,))
    for i in range(3):
        x_ticks[i] = (i + 1) * dataset_margin + 0.5 * dataset_width + i * dataset_width

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['Training Complex', 'Office', 'Zuidberg'])

    # ax.set_xlabel('Algorithms - Datasets')
    ax.set_ylim(10**2, 10**8)
    ax.set_ylabel('Time (ms)')
    ax.set_yscale('log')
    ax.set_title('Computation Times for Different Algorithms on Different Datasets')
    ax.legend(["Loading", "Reconstruction", "Cleaning"])

    plt.tight_layout()
    plt.show()


def plot_computation_times_separate():
    # Sample data
    datasets = ['Training Complex', 'Office', 'Zuidberg']
    algorithms = ['AS', 'BPA', 'SPSR']

    # index 0,1,2 are Alpha Shapes, BPA and SPSR on Training complex
    # index 3,4,5 are algorithms on office
    # etc
    labels = ['TC - A', 'TC - BPA', 'TC - SPSR',
              'Office - A', 'Office - BPA', 'Office - SPSR',
              'Zuidberg - A', 'Zuidberg - BPA', 'Zuidberg - SPSR']
    loading_times = np.array([3145, 75420, 75170, 60211, 1284031, 1125012, 2516, 61108, 61895])
    loading_time_errors = [84, 2460, 2144, 2728, 207702, 108638, 46, 1772, 901]
    reconstruction_times = np.array([59283, 28946, 66288, 941427, 812001, 235038, 47131, 46572, 84872])
    reconstruction_errors = [447, 1873, 1120, 90653, 176905, 6259, 389, 3292, 1953]
    cleaning_times = np.array([2820, 5145, 6423, 35769, 46146, 26101, 2523, 3481, 10744])
    cleaning_time_errors = [56, 77, 53, 675, 774, 3548, 41, 28, 121]

    bar_width = 0.5  # Adjust bar width
    bar_spacing = 0.05  # Adjust spacing between bars

    fig, ax = plt.subplots()
    bar_colors = ['tab:blue', 'tab:orange', 'tab:green']

    dataset_index = 1
    capsize = 5

    for j, algorithm in enumerate(algorithms):
        x_position = j * (bar_width + bar_spacing) + bar_spacing
        index = dataset_index * 3 + j
        loading_time = loading_times[index]
        loading_time_error = loading_time_errors[index]

        p = ax.bar(x_position, loading_time, width=bar_width, color=bar_colors[0], align='edge',
                   yerr=loading_time_error, capsize=capsize)
        # ax.bar_label(p, label_type='center', labels=[str(loading_time)])

        reconstruction_time = reconstruction_times[index]
        reconstruction_error = reconstruction_errors[index]
        p = ax.bar(x_position, reconstruction_time, width=bar_width, bottom=loading_time, color=bar_colors[1],
               align='edge', yerr=reconstruction_error, capsize=capsize)
        # ax.bar_label(p, label_type='center', labels=[str(reconstruction_time)])

        cleaning_time = cleaning_times[index]
        cleaning_time_error = cleaning_time_errors[index]
        p = ax.bar(x_position, cleaning_time, width=bar_width, bottom=loading_time + reconstruction_time,
                   color=bar_colors[2], align='edge', yerr=cleaning_time_error, capsize=capsize)
        # ax.bar_label(p, label_type='center', labels=[str(cleaning_time)])

    x_ticks = np.zeros(shape=(3,))
    x_ticks[0] = bar_spacing + 0.5 * bar_width
    x_ticks[1] = 2 * bar_spacing + 1.5 * bar_width
    x_ticks[2] = 3 * bar_spacing + 2.5 * bar_width
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['Alpha Shapes', 'BPA', 'SPSR'])

    # ax.set_xlabel('Algorithms - Datasets')
    # ax.set_ylim(10 ** 2, 10 ** 8)
    ax.set_ylabel('Time (ms)')
    # ax.set_yscale('log')
    # ax.set_title('Computation Times for Different Algorithms on Different Datasets')

    ax.legend(["Loading", "Reconstruction", "Cleaning"])

    fig.set_size_inches(5, 5)
    fig.dpi = 300
    plt.tight_layout()
    plt.show()


def print_octree_sizes(margin: float = 1.1):

    paths = [
        "E:\\etvr_datasets\\enfsi-2023_reduced.pcd",
        # "E:\\etvr_datasets\\ruimte_ETVR.ply",
        "E:\\etvr_datasets\\Zuidberg.ply"
    ]

    max_differences = []

    for path in paths:
        tc_pcd = open3d.io.read_point_cloud(path)
        tc_points = np.asarray(tc_pcd.points)
        tc_differences = np.absolute(np.amax(tc_points) - np.amin(tc_points))
        max_diff = np.amax(tc_differences) * margin
        max_differences.append(max_diff)

    for i in range(9, 15):
        for j, path in enumerate(paths):
            print(f"Octree cell size is {max_differences[j] / (2**i)} at depth {i} for {path}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #color_points_by_distances()
    # plot_aspect_ratios()
    # show_meshes()
    # plot_computation_times()
    # print_octree_sizes(1.1)
    plot_computation_times_separate()
