from typing import List
import matplotlib.pyplot as plt
import numpy as np
import open3d


def plot_results():
    data_paths: List[str] = [
        "E:\\thesis-results\\chosen\\alpha-shapes.npz",
        "E:\\thesis-results\\chosen\\bpa.npz",
        "E:\\thesis-results\\chosen\\delaunay.npz",
        "E:\\thesis-results\\chosen\\spsr.npz"
    ]
    arrays = []
    for data_path in data_paths:
        data_raw = np.load(data_path)
        arrays.append(data_raw['aspect_ratios'])
    labels = ["Alpha Shapes", "BPA", "Delaunay", "SPSR"]
    # bins = 100
    bins = np.arange(start=0, stop=1000, step=0.1).tolist()
    #plt.hist(arrays, histtype='step', log=True, label=labels, bins=bins)
    # plt.ylim(ymin=1)
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of triangles")
    ax.set_ylabel("Aspect ratio")
    # ax.set_title("Aspect ratio distribution of Training Complex dataset reconstructions")
    ax.set_xscale('log')
    ax.set_xlim(xmin=1, xmax=1200)
    n, bins, patches = ax.hist(arrays, histtype='step', log=True, bins=bins)
    plt.legend(labels=labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_size_inches(10, 5)
    fig.dpi = 300
    # fig = plt.figure(figsize=(10, 5), dpi=300)
    fig.savefig("aspect-ratios-training-complex.png", dpi=300)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    color_points_by_distances()
    #plot_results()
    # show_meshes()