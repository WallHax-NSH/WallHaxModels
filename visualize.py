import open3d as o3d
import numpy as np
import json

# Load point cloud
pcd = o3d.io.read_point_cloud("data.ply")

# Load detections
with open("3detr/detections.json") as f:
    det = json.load(f)["det"]

# Create Open3D geometries for boxes
box_geometries = []
for d in det:
    box = np.array(d["box"])
    # Create lines for the 12 edges of the box
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # red boxes
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(box),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    box_geometries.append(line_set)

# Visualize
o3d.visualization.draw_geometries([pcd] + box_geometries)