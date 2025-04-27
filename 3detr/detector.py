import numpy as np
import torch
import pathlib, sys, inspect, json
from datetime import datetime
from .datasets.sunrgbd import SunrgbdDatasetConfig

SUNRGBD_CLASSES = [
    "bed","table","sofa","chair","toilet","desk",
    "dresser","night_stand","bookshelf","bathtub",
    "monitor","lamp","bag","bin","box","door",
    "shelf","picture"
]

class DummyCfg:
    def __init__(self, num_class, enc_type=None):
        self.use_color = False
        self.enc_dim = 256
        self.enc_nhead = 8
        self.enc_ffn_dim = 128
        self.enc_dropout = 0.1
        self.enc_activation = "relu"
        self.enc_nlayers = 3
        self.preenc_npoints = 2048
        self.dec_dim = 256
        self.dec_nhead = 8
        self.dec_ffn_dim = 256
        self.dec_dropout = 0.1
        self.dec_nlayers = 8
        self.mlp_dropout = 0.3
        self.nqueries = 128
        self.num_class = num_class
        if enc_type is not None:
            self.enc_type = enc_type

def load_points(path):
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype("f4")[:, :3]
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(p))
    return np.asarray(pc.points, dtype="f4")

def run_detection(cloud_path, ckpt_path=None, thr=0.3, masked=False):
    xyz = load_points(cloud_path)
    N = 20_000
    xyz = xyz[np.random.choice(len(xyz), N, replace=True)].astype("f4")

    from .models.model_3detr import build_3detr

    # Always use the hardcoded checkpoint
    ckpt_path = str(pathlib.Path(__file__).resolve().parent / "sunrgbd_masked_ep1080.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    names = ckpt.get("class_names", SUNRGBD_CLASSES)

    sig = inspect.signature(build_3detr)
    enc_type = "masked" if masked else "vanilla"
    dummy_args = DummyCfg(len(names), enc_type)
    dummy_cfg = SunrgbdDatasetConfig()
    model, _ = build_3detr(dummy_args, dummy_cfg)
    model.load_state_dict(ckpt["model"]); model.eval().cuda()

    pts = torch.from_numpy(xyz).unsqueeze(0).cuda()
    pc_min = torch.from_numpy(xyz.min(0)).float().unsqueeze(0).cuda()
    pc_max = torch.from_numpy(xyz.max(0)).float().unsqueeze(0).cuda()
    inputs = {
        "point_clouds": pts,
        "point_cloud_dims_min": pc_min,
        "point_cloud_dims_max": pc_max
    }
    with torch.no_grad():
        out = model(inputs)

    prob  = out["outputs"]["sem_cls_logits"][0].softmax(-1)
    keep  = prob.max(-1).values > thr
    labels= prob[keep].argmax(-1).cpu().numpy()
    scores= prob[keep].max(-1).values.cpu().numpy()
    boxes = out["outputs"]["box_corners"][0][keep].cpu().numpy()

    dets  = [dict(cls=names[int(l)], score=float(s),
                  box=[list(map(float, corner)) for corner in b])
             for b,l,s in zip(boxes, labels, scores)]

    return {
        "ts": str(datetime.now()),
        "src": cloud_path,
        "det": dets
    }
