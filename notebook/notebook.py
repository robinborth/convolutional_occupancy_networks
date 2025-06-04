# %%
from pathlib import Path

import numpy as np
import open3d as o3d

data_dir = Path(
    "/home/borth/convolutional_occupancy_networks/data/ShapeNet/03001627/1a8bbf2994788e2743e99e0cae970928"
)
point_cloud = np.load(data_dir / "pointcloud.npz")
print(point_cloud.files)
pcd_points = point_cloud["points"]
pcd_normals = point_cloud["normals"]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
o3d.visualization.draw_plotly([pcd])

# %%

from pathlib import Path

import numpy as np
import open3d as o3d

data_dir = Path(
    "/home/borth/convolutional_occupancy_networks/data/ShapeNet/03001627/1a8bbf2994788e2743e99e0cae970928"
)
point_cloud = np.load(data_dir / "points.npz")
pcd_points = point_cloud["points"]
pcd_occupancies = np.unpackbits(point_cloud["occupancies"])[: pcd_points.shape[0]]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)
o3d.visualization.draw_plotly([pcd])
# %%
import open3d as o3d
import open3d.core as o3c

vbg = o3d.t.geometry.VoxelBlockGrid(
    attr_names=("tsdf", "weight"),
    attr_dtypes=(o3c.float32, o3c.float32),
    attr_channels=((1), (1)),
    voxel_size=3.0 / 512,
    block_resolution=16,
    block_count=50000,
    device=o3c.Device("CUDA:0"),
)
