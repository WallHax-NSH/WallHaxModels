#!/usr/bin/env python
import argparse, json, pathlib, sys, inspect, numpy as np, torch
from datetime import datetime
from datasets.sunrgbd import SunrgbdDatasetConfig

# ---------- fallback SUN RGB-D class list (18) ----------
SUNRGBD_CLASSES = [
    "bed","table","sofa","chair","toilet","desk",
    "dresser","night_stand","bookshelf","bathtub",
    "monitor","lamp","bag","bin","box","door",
    "shelf","picture"
]

# ---------- helpers ----------
def load_points(path):
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype("f4")[:, :3]
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(p))
    return np.asarray(pc.points, dtype="f4")

def cxcyczwhd_to_xyz_minmax(b):
    c, d = b[:, :3], b[:, 3:6] / 2
    return torch.cat([c - d, c + d], -1)

# ---------- tiny dummy dataset_config for all API versions ----------
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud"); ap.add_argument("ckpt")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()

    xyz = load_points(args.cloud)
    N = 20_000
    xyz = xyz[np.random.choice(len(xyz), N, replace=True)].astype("f4")

    REPO = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(REPO))
    from models.model_3detr import build_3detr

    ckpt = torch.load(args.ckpt, map_location="cpu")
    names = ckpt.get("class_names", SUNRGBD_CLASSES)

    # -------- build model regardless of signature ----------
    sig = inspect.signature(build_3detr)
    enc_type = "masked" if args.masked else "vanilla"


    enc_type = "masked" if args.masked else "vanilla"
    dummy_args = DummyCfg(len(names), enc_type)
    dummy_cfg = SunrgbdDatasetConfig()
    model, _ = build_3detr(dummy_args, dummy_cfg)
   
    model.load_state_dict(ckpt["model"]); model.eval().cuda()

    pts = torch.from_numpy(xyz).unsqueeze(0).cuda()  # (1, N, 3)
    pc_min = torch.from_numpy(xyz.min(0)).float().unsqueeze(0).cuda()  # shape (1, 3)
    pc_max = torch.from_numpy(xyz.max(0)).float().unsqueeze(0).cuda()  # shape (1, 3)
    inputs = {
        "point_clouds": pts,
        "point_cloud_dims_min": pc_min,
        "point_cloud_dims_max": pc_max
    }
    with torch.no_grad():
        out = model(inputs)

    prob  = out["outputs"]["sem_cls_logits"][0].softmax(-1)
    keep  = prob.max(-1).values > args.thr
    labels= prob[keep].argmax(-1).cpu().numpy()
    scores= prob[keep].max(-1).values.cpu().numpy()
    boxes = out["outputs"]["box_corners"][0][keep].cpu().numpy()

    dets  = [dict(cls=names[int(l)], score=float(s),
                  box=[list(map(float, corner)) for corner in b])
             for b,l,s in zip(boxes, labels, scores)]

    print(f"Detected {len(dets)} objects > {args.thr}")
    for d in dets[:10]:
        print(f"{d['cls']:<10} {d['score']:.2f}  {np.round(d['box'],2)}")

    json.dump({"ts": str(datetime.now()), "src": args.cloud, "det": dets},
              open("detections.json","w"), indent=2)
    print("Saved detections.json")

if __name__ == "__main__":
    main()
