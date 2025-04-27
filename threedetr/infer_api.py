import numpy as np
import torch
from datetime import datetime
from .run import load_points, DummyCfg, SUNRGBD_CLASSES
from .datasets.sunrgbd import SunrgbdDatasetConfig

def run_detection(cloud_path, ckpt_path, masked=False, thr=0.3):
    xyz = load_points(cloud_path)
    N = 20_000
    xyz = xyz[np.random.choice(len(xyz), N, replace=True)].astype("f4")

    from .models.model_3detr import build_3detr

    ckpt = torch.load(ckpt_path, map_location="cpu")
    names = ckpt.get("class_names", SUNRGBD_CLASSES)

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

    return {"ts": str(datetime.now()), "src": cloud_path, "det": dets} 