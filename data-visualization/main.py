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
    plt.hist(arrays, histtype='step', log=True, label=labels, bins=bins)
    # plt.ylim(ymin=1)
    plt.ylabel('Number of triangles')
    plt.xlabel('Aspect Ratio')
    plt.xscale('log')
    plt.xlim(xmin=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(labels)
    plt.show()

def show_meshes():
    paths = [
        "E:\\thesis-results\\delaunay-1\\1711352017.4267113_run_0\\enfsi-2023_reduced_cloud_mesh.ply",
        # "E:\\thesis-results\\delaunay-1\\1711358023.1019151_run_2\\enfsi-2023_reduced_cloud_mesh.ply"
    ]

    meshes = [open3d.io.read_triangle_mesh(path) for path in paths]
    open3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # plot_results()
    show_meshes()