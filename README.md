# Manual

This manual contains information on how to use the executables in this project. There are zipped folders in the releases containing the `.exe` files for `pointcloud2mesh` (i.e. creating a mesh/3D model out of a point cloud) and `pointcloudclassification` for segmenting and classifying a point cloud.

## Reconstruction (pointcloud2mesh)
You can use the `pointcloud2mesh-singular.exe` (download [here](https://github.com/gabzzel/master-thesis/releases/download/v0.1/pointcloud2mesh-singular.exe)) to create 3D meshes out of point clouds. If this executable does not work, try the `main.exe` in the 'foldered' version (download [here](https://github.com/gabzzel/master-thesis/releases/download/v0.1/pointcloud2mesh-foldered.zip)).

To use this software, you *must* execute the .exe using the commandline. Opening by double clicking will not do anything.

### Command line arguments

Keep in mind that required arguments need to be specified *before* optional argument. Also, the required arguments have to specified *in order* and *without argument name*! See the examples below for clarification.

If in the list below "(Required)" is not specified, you can assume the argument is optional and the desired value should be prefaced / prefixed by the argument name. Again, see the examples for more info.

1. `point_cloud_path` (Required) = The first argument is the path to the point cloud file. This can be a `.xyz`, `.pcd` or `.ply` file. 
2. `result_path` (Required) = The path where the resulting 3D model / mesh will be stored. This must be a directory, not a file!
3. `-segments_path` = The path where the 'segments' are located. This must be a `.npy` file that contains a cluster / segment index per point. If not specified, the program will assume no segments and create a single output.
4. `-classifications_path` = The path to an `.npy` file that contains a classification per point. Ignored if not specified. Only used to name individual meshes if multiple meshes are created depending on the clusters. Keep in mind the program expects the class of all points within a cluster to be the same. 
5. `-down_sample_method` or `-dsm` = The downsampling method to use. If not specified, no downsampling of the point cloud will occur. Possible options are `none`, `voxel` and `random`. See [here](https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html) for an explanation on voxel downsampling. Random downsampling removes a certain percentage of points at random. Recommended is to use voxel downsampling. Doing no downsampling can result is very long computation times. 
6. `-down_sample_params` or `-dsp` = The parameter of the voxel downsampling method used. 
    - When using random downsampling, this value represents the ratio of the resulting (downsampled) point cloud to the original point cloud (in terms of point count), which means the value must be between 0 and 1. 
    - When using voxel downsampling, this value represents the size of the voxels used (in all dimensions). When voxel downsampling, the recommended value is approximately `0.01`. The higher the value, the larger the voxels and the lower the quality (but higher the speed.)
7. `-normal_estimation_neighbours` or `-nen` = The number of neighbours used during normal estimation. Defaults to 10. Setting this too low will result in normals not being estimated correctly, which impacts the quality of the final mesh. Setting this too high will impact computation time. Anything between 5 and 30 should be fine.
8. `-normal_estimation_radius` or `-ner` = The radius around a target point within which neighbours will be consired during normal estimation. Which value this should be depends on the density of the point cloud (which may be impacted by downsampling!). Setting this too low will result in neighbours not being considered, negatively impacting the quality of the resulting mesh. Setting this too high increases the probability of far away neighbours being considered, also lowering the probability of the resulting mesh. The recommended value is around 3.5 times the voxel size (when using voxel downsampling.) Using 0.1 if you don't know.
9. `-skip_normalizing_normals` or `-snn` = Skip normalizing the normals. This prevents the normals from being forced to length 1. This is not recommended. No value has to follow this argument.
10. `-orient_normals` or `-on` = How many neighbours to use (for each point) when orienting the normals. Set to 0 to not orient normals. Defaults to 10. Orienting the normals is recommended, as doing so significantly improves mesh quality. However, orienting normals takes a long time, so be prepared.
11. `-verbose` or `-v` = Including this argument (no value necessary) makes sure results and progress are printed to the console.
12. `-surface_reconstruction_algorithm` or `-sra` = Which algorithm to use for reconstruction (i.e. creation of the mesh). Must be either:
    - `ball_pivoting` (alias `bpa`) for [Ball Pivoting Algoritm](http://mesh.brown.edu/taubin/pdfs/bernardini-etal-tvcg99.pdf),
    - `poisson` (alias `spsr`) for [Screened Poisson Surface Reconstruction](https://www.3dvar.com/Kazhdan2013Screened.pdf) or 
    -  `alpha_shapes` (alias `alpha`).

    Defaults to `spsr`.
13. `-alpha` = The value for alpha when using Alpha Shapes. Ignored otherwise. 
14. `-ball_pivoting_radii` or `-bpar` = The radii to use for the Ball Pivoting Algorithm. It is recommended to specify 3 to 5 radii at regular intervals lower than the mean nearest neighbour distance and voxel size. For example: I got good results when using `0.001 0.009 0.024 0.048 0.1` when using voxel size of 0.01. If the program seems stuck for a long time on a certain radius, try little permutations.
15. `-poisson_density_quantile` or `-pdq` = When using SPSR, buldging can occur. This can be mitigated (or at least reduced) by cleaning the mesh based on the local density around certain points. Point with low density have lower support and can sometimes be removed without penalty. Use this parameter to remove a portion of the points (between 0 and 1) depending on their support. Recommended is to set this between 0 and 0.1.
16. `-poisson_octree_max_depth` or `-pomd` = The maximum depth of the internal octree used when executing SPSR. A higher value results in more detailed meshes, but a higher execution time as well. Values between 8 and 15 are recommended, based on the scale of the point cloud. Use 11 if not sure. 
17. `-poisson_octree_min_width` or `-pomw` = The minimum size of the cells (in all dimensions) of the internal octree of SPSR. Use either this or the max depth parameter, not both. A larger value results in lower resolution octree and thus a lower quality reconstruction, but higher computation speed. Recommended to keep this higher than the voxel size when voxel downsampling.
18. `-mesh_clean_methods` or `-mcm` = The mesh cleaning methods to use after creating the mesh. Options are `simple`, `edge_length` and/or `aspect_ratio`. Set to `all` to use all, which is recommended. Simple cleaning removes invalid triangles and lonely points. Edge cleaning removes edges that are very large and aspect ratio removes triangles that have high aspect ratios. See the next parameters for more info.
19. `-edge_length_clean_portion` or `-elcp` = a value between 0 and 1 that determines the amount of edges that need to be *retained* when removing the larges edges. Use a high value, between 0.9 and 1. Setting this to 1 ignores edge cleaning.
20. `-aspect_ratio_clean_portion` or `-arcp` = a value between 0 and 1 that determines the amount of triangles that are *retained* when removing the triangles with the largest aspect ratios. The aspect ratio is calculated by dividing the largest edge by the shortest edge for each triangle. Use a high value, between 0.9 and 1. 
21. `-mesh_quality_metrics` or `-mqm` = a list of values that determines which metrics will be calculated and printed and/or saved to disk, measured on the resulting mesh. Can be `edge_lengths`, `aspect_ratios`, `connectivity`, `discrete_curvature` and/or `normal_deviations`. Use `all` to use all. Warning, calculating discrete curvatures is very costly.
22. `-mesh_to_cloud_metrics` or `-m2cm` = Which metrics to calculate that say something about how well the reconstruction / mesh approximates the original point cloud. Options are `chamfer`, `hausdorff` and `distances`. The `distances` metric calculates the root mean squared error, where the error is the distance from a vertex in the mesh to the nearest point in the original point cloud.
23. `-result_path` = The path (must be a directory / folder!) where the results will be stored, including the final mesh.
24. `-mesh_output_format` = The file format that needs to be used when savig the final mesh(es). Must be either `.gltf`, `.obj` or `.ply`. Defaults to `.gltf`.
25. `-processes` = How many workers / threads to use for certain functions or methods (for example SPSR). Set to `-1` to automatically determine. Defaults to `-1`.
26. `-draw` = including this argument will draw the resulting mesh to the screen once done.

### Examples

Load a pointcloud called `point_cloud.pcd` in the documents folder, convert to a mesh using SPSR (with octree dpeth 10) and store the mesh at `D:\result_dir` in `.ply` format.
```
pointcloud2mesh.exe C:\Users\...\Documents\point_cloud.pcd D:\result_dir -sra spsr -verbose -poisson_octree_max_depth 10 -store_mesh -mesh_output_format .ply
```


## Classification (pointcloudclassification)
This section describes how to use the point cloud classifcation and segmentation tool.

Like the `pointcloud2mesh`, the `pointcloudclassification` has command line arguments and can ONLY be used using the command line.

### Command line arguments
1. `point_cloud_path` (Required) = The path to the point cloud to classify / segment. Must be a `.ply`, `.pcd` or `.xyz` file.
2. `method` (Required) = The classification / segment method to use. Choices are `hdbscan` and `pointnetv2`. 
3. `-result_path` = The directory where the resulting segments and classification will be saved in `.npy` format.
4. `-downsampling_method` = which downsampling method to use before classification / segmentation. Choices are `voxel`, `random` or `none`. Defaults to `none`.
5. `-downsampling_param` = The parameter passed to the downsampling. For voxel downsampling, this is the voxel size. For random downsampling, this is the ratio of retained points.
6. `-normal_estimation_radius` = See the same parameter for `pointcloud2mesh`.
7. `-normal_estimation_neighbours` = See the same parameter for `pointcloud2mesh`
8. `-normal_orientation_neighbours` = See the same parameter for `pointcloud2mesh`
9. `-hdbscan_min_cluster_size` = The minimum amount of points for HDBSCAN to consider it a cluster. Recommended to set to at least 100, but depends on point cloud density. Defaults to 125. Setting this to a higher value produces less, bigger clusters. Setting this lower will product more, smaller clusters.
10. `-hdbscan_min_samples` = The minimum amount of points another point needs to have within a radius to be considered a 'core' point by HDBSCAN. Recommended to be set a bit higher than `-hdbscan_min_cluster_size`. Defaults to 200.
11. `-include_colors` = Whether to include color data during classification or segmentation. Not recommended for HDBSCAN, but is recommended for PointNetV2. Defaults to False.
12. `-include_normals` = Whether to include normal data during classification or segmentation. In contrast to color data, recommended to set to true when using HDBSCAN, and false for PointNetV2.
13. `-verbose` = Whether to print progress and results to the console.
14. `-pointnetv2_checkpoint_path` = The path to the `.pth` file that contains the trained PointNetV2 model.
15. `-do_segmentation` = Whether to do basic radius-based segmentation after PointNetV2 classification. Ignored when using HBDSCAN. Recommended to be set to True. Can take a while. Defaults to False (because of time reasons).
16. `-segmentation_max_distance` = The maximum distance points can be from each other in order to be included in the same cluster during segmentation after classification. Ignored when using HDBSCAN or if `-do_segmentation` is set to False. Recommended value is below 0.5, defaults is 0.2 (which is already pretty large). 
17. `-hdbscan_noise_nearest_neighbours` = When performing HDBSCAN, point can be left unassigned (i.e. noise). These points can be reassigned to a nearby cluster, depending on the majority class of nearby points. The amount of neighbouring points considered during this process is determined with this parameter. Defaults to 3. Recommended to be a small odd number (<10).
