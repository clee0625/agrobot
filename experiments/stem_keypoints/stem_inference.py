"""Stem junction inference. BGR image in, junctions with cut point and angles out. No ROS."""
import math
import numpy as np
import torch
import torchvision

KEYPOINTS = ("Mtop", "JP", "Mbut", "Pu")

def _angle_deg(v):
    """Direction in the image: 0 = right, 90 = up, -90 = down."""
    return math.degrees(math.atan2(-float(v[1]), float(v[0])))

class StemDetector:
    def __init__(self, weights_path, device=None, score_thresh=0.7, cut_offset=0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_thresh, self.cut_offset = score_thresh, cut_offset
        m = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, num_keypoints=len(KEYPOINTS))
        m.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = m.to(self.device).eval()

    @torch.no_grad()
    def detect(self, bgr):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([x])[0]
        results = []
        for box, score, kp in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["keypoints"].cpu().numpy()):
            if score < self.score_thresh:
                continue
            mtop, jp, mbut, pu = kp[:, :2]
            cut = jp + self.cut_offset * (pu - jp)
            results.append({
                "score": float(score),
                "box": [float(v) for v in box],
                "keypoints": {n: [float(p[0]), float(p[1])] for n, p in zip(KEYPOINTS, kp[:, :2])},
                "cut_point": [float(cut[0]), float(cut[1])],
                "cluster_stem_angle_deg": _angle_deg(pu - jp),
                "main_stem_angle_deg": _angle_deg(mtop - mbut),
            })
        return results
