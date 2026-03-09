"""
Clean Action Recognition Pipeline

- Stream 1: Person actions (normal / fall / fighting)
    * YOLOv11 person detector
    * Simple IOU + centroid tracker
    * SlowFast + Pose classifier (trained in SlowFast_Pose_Retrain.ipynb)
    * Temporal smoothing + short persistence

- Stream 2: Fire detection
    * YOLO fire detector (best_model_fire.pt)
    * Optional MobileNetV3 CNN verifier (fire_red_cnn.pth)

The main entrypoint kept for app.py compatibility:
    process_video_with_tracking(video_path, slowfast_checkpoint, yolo_model, out_path, ...)
"""

from collections import defaultdict, deque
from pathlib import Path
import logging
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ----------------------------- CONFIG ---------------------------------

ACTION_CLASSES = ["normal", "fall", "fighting"]  # index 0,1,2
CLIP_LEN = 64
MIN_FRAMES_FOR_ACTION = 8  # minimum frames to start action classification

# Simple, easy-to-understand thresholds
ACTION_THRESHOLDS = {
    "fall": 0.50,       # require strong evidence
    "fighting": 0.65,   # STRICT - avoid false positives on bystanders
    "normal": 0.0,
}

# How long to keep an abnormal label after last strong evidence (in seconds)
ACTION_PERSISTENCE_SECONDS = 5.0

# Temporal smoothing window for classifier probabilities
SMOOTH_WINDOW = 10  # frames per track

# Fire detection config
FIRE_DETECTION_CONFIG = {
    "conf": 0.45,          # increased to reduce false positives on skin
    "iou": 0.30,
    "max_det": 20,
    "imgsz": 640,
    "classes": [0],        # fire class id in best_model_fire.pt
    "min_area": 800,       # increased to avoid detecting faces/hands as fire
}


# ----------------------------- MODELS ---------------------------------

class PoseEncoder(nn.Module):
    """Same Pose encoder as used in SlowFast_Pose_Retrain.ipynb."""

    def __init__(self, input_dim=51, hidden_dim=256, output_dim=512, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            256,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, pose):
        # pose: (B, T, 17, 3)
        B, T, _, _ = pose.shape
        pose_flat = pose.reshape(B, T, -1)        # (B, T, 51)
        x = pose_flat.permute(0, 2, 1)            # (B, 51, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)                          # (B, 256, T//2)
        x = x.permute(0, 2, 1)                    # (B, T//2, 256)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=1)
        return self.fc(h)


class SlowFastPoseModel(nn.Module):
    """SlowFast + Pose fusion for 3-class person action."""
        
    def __init__(self, num_classes=3, pose_feature_dim=512):
        super().__init__()
        try:
            from pytorchvideo.models.hub import slowfast_r101
            
            self.slowfast = slowfast_r101(pretrained=True)
            if hasattr(self.slowfast, "blocks"):
                self.slowfast_feat_dim = self.slowfast.blocks[-1].proj.in_features
                self.slowfast.blocks[-1].proj = nn.Identity()
            else:
                self.slowfast_feat_dim = 2304
            logger.info(f"Loaded SlowFast R101, feat_dim={self.slowfast_feat_dim}")
        except Exception as e:
            logger.warning(f"Could not load pretrained SlowFast, fallback avg-pool: {e}")
            self.slowfast = None
            self.slowfast_feat_dim = 2304
        
        self.pose_encoder = PoseEncoder(input_dim=51, hidden_dim=256, output_dim=pose_feature_dim)
        fusion_dim = self.slowfast_feat_dim + pose_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, video_inputs, pose_inputs):
        if self.slowfast is not None:
            rgb_features = self.slowfast(video_inputs)
        else:
            # video_inputs is [slow, fast]
            x = video_inputs[0]
            rgb_features = x.mean(dim=(2, 3, 4))
        pose_features = self.pose_encoder(pose_inputs)
        fused = self.fusion(torch.cat([rgb_features, pose_features], dim=1))
        return self.classifier(fused)


def pack_pathway_input(frames: torch.Tensor):
    """
    Pack frames to SlowFast two-pathway input.
    frames: (1, C, T, H, W)
    """
    from math import ceil

    assert frames.dim() == 5, "frames must be (B, C, T, H, W)"
    B, C, T, H, W = frames.shape
    alpha_slow = 8
    alpha_fast = 2

    idx_slow = torch.linspace(0, T - 1, ceil(T / alpha_slow), device=frames.device).long()
    idx_fast = torch.linspace(0, T - 1, ceil(T / alpha_fast), device=frames.device).long()
    slow = frames.index_select(2, idx_slow)
    fast = frames.index_select(2, idx_fast)
    return [slow, fast]


def load_yolo_model(model_path: str) -> YOLO:
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    logger.info(f"Loading YOLO model: {model_path}")
    return YOLO(model_path)


def load_action_model(checkpoint_path: str) -> SlowFastPoseModel:
    logger.info(f"Loading SlowFast+Pose checkpoint: {checkpoint_path}")
    model = SlowFastPoseModel(num_classes=len(ACTION_CLASSES), pose_feature_dim=512).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_pose_model():
    """YOLOv11 pose model for keypoints."""
    try:
        logger.info("Loading YOLOv11 pose model (yolo11m-pose.pt)")
        model = YOLO("yolo11m-pose.pt")
        return model
    except Exception as e:
        logger.warning(f"Pose model not available, using zero-poses: {e}")
        return None


def extract_pose(pose_model, frame_bgr, bbox):
    """Return (17,3) tensor in [0,1] (x,y,conf)."""
    if pose_model is None:
        return torch.zeros(17, 3, device=DEVICE)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return torch.zeros(17, 3, device=DEVICE)
    crop = frame_bgr[y1:y2, x1:x2]
    try:
        res = pose_model(crop, verbose=False)
        if len(res) == 0 or res[0].keypoints is None:
            return torch.zeros(17, 3, device=DEVICE)
        kpts = res[0].keypoints.data[0]  # (17,3)
        ch, cw = crop.shape[:2]
        kpts_norm = kpts.clone()
        kpts_norm[:, 0] /= max(cw, 1)
        kpts_norm[:, 1] /= max(ch, 1)
        return kpts_norm.to(DEVICE)
    except Exception as e:
        logger.debug(f"Pose error: {e}")
        return torch.zeros(17, 3, device=DEVICE)


# -------------------------- FIRE MODELS -------------------------------


class FireVerificationCNN(nn.Module):
    """MobileNetV3-Small head for 2-class fire / non-fire."""

    def __init__(self, num_classes=2):
        super().__init__()
        try:
            from torchvision import models

            base = models.mobilenet_v3_small(weights=None)
            in_feats = base.classifier[0].in_features
            base.classifier = nn.Sequential(
                nn.Linear(in_feats, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1024, num_classes),
            )
            self.model = base
        except Exception as e:
            logger.warning(f"torchvision not available for Fire CNN: {e}")
            self.model = None
    
    def forward(self, x):
        if self.model is None:
            return torch.zeros(x.size(0), 2, device=x.device)
        return self.model(x)


def load_fire_verification_cnn(checkpoint_path: str):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        logger.warning(f"Fire CNN checkpoint not found: {checkpoint_path}")
        return None
    model = FireVerificationCNN(num_classes=2).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("Loaded Fire verification CNN")
    return model


def load_fire_model(fire_model_path: str):
    fire_model_path = str(fire_model_path)
    if not Path(fire_model_path).exists():
        logger.warning(f"Fire YOLO model missing: {fire_model_path}")
        return None
    logger.info(f"Loading fire YOLO model: {fire_model_path}")
    return YOLO(fire_model_path)


def verify_fire_with_cnn(cnn_model, frame_bgr, bbox, conf_threshold=0.5):
    if cnn_model is None:
        return True, 1.0
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224, 224))
    img = torch.from_numpy(crop_resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = ((img - mean) / std).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(img)
        probs = torch.softmax(logits, dim=1)[0]
        fire_conf = float(probs[1].item())
    return fire_conf >= conf_threshold, fire_conf


def get_posture_type(bbox, frame_height):
    """
    Determine if person is lying down or standing/sitting.
    Returns: 'lying', 'standing', or 'unknown'
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect = width / max(height, 1)
    near_ground = (bbox[3] / frame_height) > 0.75
    
    # Person is lying if width > height significantly AND close to ground
    if aspect > 1.3 and near_ground:
        return 'lying'
    # Person is standing/sitting if height > width
    elif aspect < 0.8:
        return 'standing'
    return 'unknown'


def has_close_proximity(tid, all_tracks, bbox, threshold_ratio=0.25):
    """
    Check if this person has another person very close by (potential physical interaction).
    Returns True if there's another person within threshold_ratio * diagonal distance.
    STRICT: Only people in very close physical contact (25% of diagonal).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    proximity_dist = diag * threshold_ratio
    
    for other_tid, other_bbox, _ in all_tracks:
        if other_tid == tid:
            continue
        ox1, oy1, ox2, oy2 = other_bbox
        ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
        if dist < proximity_dist:
            return True
    return False


def has_significant_motion(track_state, min_motion_ratio=0.12):
    """
    Check if person has significant motion in recent frames.
    Returns True if bbox center moved more than min_motion_ratio * frame_height.
    STRICT: Require 12% movement to avoid triggering on small sways.
    """
    history = track_state.get("bbox_history", deque(maxlen=20))
    if len(history) < 10:
        return False  # Not enough history, don't assume motion
    
    # Calculate center positions
    centers = []
    for bbox in history:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        centers.append((cx, cy))
    
    # Check max movement
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    
    # Get approximate frame dimensions from bbox
    first_bbox = history[0]
    frame_h = first_bbox[3] * 2  # rough estimate
    
    motion_ratio = max(x_range, y_range) / max(frame_h, 1)
    return motion_ratio >= min_motion_ratio


def detect_instant_fall(track_state, bbox, frame_height):
    """
    Lightweight heuristic to trigger FALL immediately (before SlowFast) when:
    - person is lying horizontally near ground
    - OR person suddenly drops + tilts horizontal
    Returns (trigger: bool, confidence: float)
    """
    history = track_state.setdefault("bbox_history", deque(maxlen=20))
    posture_history = track_state.setdefault("posture_history", deque(maxlen=20))
    
    history.append(bbox)
    current_posture = get_posture_type(bbox, frame_height)
    posture_history.append(current_posture)
    
    if len(history) < 3 or frame_height <= 0:
        return False, 0.0

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect = width / max(height, 1)
    
    # Key conditions
    near_ground = (bbox[3] / frame_height) > 0.72
    very_near_ground = (bbox[3] / frame_height) > 0.85
    lying_horizontal = aspect > 1.3  # width > height = lying down
    clearly_horizontal = aspect > 1.8
    
    # Check for vertical drop (optional boost)
    prev_bbox = history[0]
    prev_center = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_center = (bbox[1] + bbox[3]) / 2
    drop_norm = (curr_center - prev_center) / max(frame_height, 1)
    had_drop = drop_norm > 0.08
    
    # Check for posture change (was standing, now lying)
    posture_change = False
    if len(posture_history) >= 8:
        old_postures = list(posture_history)[:4]
        recent_postures = list(posture_history)[-4:]
        was_standing = old_postures.count('standing') >= 2
        now_lying = recent_postures.count('lying') >= 2
        posture_change = was_standing and now_lying

    # FALL detection paths:
    
    # Path A: Clearly horizontal + very near ground = definitely fallen
    if clearly_horizontal and very_near_ground:
        return True, 0.90
    
    # Path B: Lying horizontal + near ground + (had drop OR posture changed)
    if lying_horizontal and near_ground and (had_drop or posture_change):
        conf = min(0.88, 0.70 + drop_norm * 2)
        return True, conf
    
    # Path C: Currently lying posture (from get_posture_type)
    if current_posture == 'lying' and near_ground:
        # Person is in lying position near ground
        return True, 0.75
    
    return False, 0.0


def detect_instant_fighting(track_state, bbox, pose, all_tracks, frame_height):
    """
    STRICT Fighting detection - only for actual violent confrontations.
    Requires: Multiple people + Physical contact/overlap + Aggressive motion
    NOT triggered by: Single person, walking, normal gestures
    """
    history = track_state.setdefault("fight_bbox_history", deque(maxlen=20))
    pose_history = track_state.setdefault("pose_history", deque(maxlen=20))
    
    history.append(bbox)
    pose_history.append(pose.cpu().numpy() if torch.is_tensor(pose) else pose)
    
    if len(history) < 8:  # Need more history for confidence
        return False, 0.0
    
    # CRITICAL: if person is lying down, they are FALLEN not fighting
    current_posture = get_posture_type(bbox, frame_height)
    if current_posture == 'lying':
        return False, 0.0
    
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_area = bbox_w * bbox_h
    
    # 1. MUST have other people nearby (fighting requires 2+ people)
    nearby_count = 0
    overlap_count = 0
    close_proximity_count = 0  # very close (touching distance)
    proximity_threshold = max(bbox_w, bbox_h) * 2.0
    close_threshold = max(bbox_w, bbox_h) * 1.2  # very close
    
    for other_tid, other_track in all_tracks.items():
        if other_tid == track_state.get("_tid"):
            continue
        other_bbox = other_track.get("bbox", [])
        if len(other_bbox) < 4:
            continue
        
        other_cx = (other_bbox[0] + other_bbox[2]) / 2
        other_cy = (other_bbox[1] + other_bbox[3]) / 2
        dist = ((cx - other_cx) ** 2 + (cy - other_cy) ** 2) ** 0.5
        
        if dist < close_threshold:
            close_proximity_count += 1
        if dist < proximity_threshold:
            nearby_count += 1
        
        # Physical overlap (bodies touching)
        x1 = max(bbox[0], other_bbox[0])
        y1 = max(bbox[1], other_bbox[1])
        x2 = min(bbox[2], other_bbox[2])
        y2 = min(bbox[3], other_bbox[3])
        if x2 > x1 and y2 > y1:
            inter_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = inter_area / max(bbox_area, 1)
            if overlap_ratio > 0.10:  # 10% overlap = significant contact
                overlap_count += 1
    
    # If alone or no one close, definitely NOT fighting
    if nearby_count == 0:
        return False, 0.0
    
    # 2. Check for AGGRESSIVE motion (not normal walking)
    centers = [((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in history]
    motion_scores = []
    motion_directions = []  # track direction for walking detection
    
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        motion_scores.append(abs(dx) + abs(dy))
        
        # Calculate movement direction (angle)
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            angle = np.arctan2(dy, dx)
            motion_directions.append(angle)
    
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    motion_normalized = avg_motion / max(frame_height, 1)
    
    # Check motion consistency - fighting has JERKY motion, walking is smooth
    motion_variance = np.std(motion_scores) if len(motion_scores) > 2 else 0
    motion_consistency = motion_variance / max(avg_motion, 1e-6)
    
    # CRITICAL: Detect walking pattern
    # Walking has: consistent direction + smooth speed + moderate motion
    is_consistent_direction = False
    if len(motion_directions) >= 5:
        # Check if direction is consistent (walking straight)
        direction_std = np.std(motion_directions)
        is_consistent_direction = direction_std < 0.5  # radians, ~30 degrees variation
    
    is_smooth_motion = motion_consistency < 0.35  # low variance = smooth
    is_walking_speed = 0.008 < motion_normalized < 0.030  # typical walking speed
    
    # If pattern matches walking, NOT fighting
    if is_consistent_direction and is_smooth_motion and is_walking_speed:
        return False, 0.0
    
    is_jerky_motion = motion_consistency > 0.6  # increased threshold for jerky
    is_rapid = motion_normalized > 0.035  # increased threshold
    is_moderate = motion_normalized > 0.015
    
    # 3. Check arm movement - DISTINGUISH walking arm swing vs fighting punches
    arm_motion = 0.0
    arm_motions = []
    left_arm_motions = []   # Left arm (indices 7, 9)
    right_arm_motions = []  # Right arm (indices 8, 10)
    
    if len(pose_history) >= 8:
        poses = list(pose_history)
        for i in range(1, len(poses)):
            curr_pose = poses[i]
            prev_pose = poses[i-1]
            
            # Track left and right arms separately
            left_motion = 0
            right_motion = 0
            
            # Left arm: left elbow (7) and left wrist (9)
            for kpt_idx in [7, 9]:
                if curr_pose[kpt_idx, 2] > 0.3 and prev_pose[kpt_idx, 2] > 0.3:
                    dx = abs(curr_pose[kpt_idx, 0] - prev_pose[kpt_idx, 0])
                    dy = abs(curr_pose[kpt_idx, 1] - prev_pose[kpt_idx, 1])
                    left_motion += dx + dy
            
            # Right arm: right elbow (8) and right wrist (10)
            for kpt_idx in [8, 10]:
                if curr_pose[kpt_idx, 2] > 0.3 and prev_pose[kpt_idx, 2] > 0.3:
                    dx = abs(curr_pose[kpt_idx, 0] - prev_pose[kpt_idx, 0])
                    dy = abs(curr_pose[kpt_idx, 1] - prev_pose[kpt_idx, 1])
                    right_motion += dx + dy
            
            left_arm_motions.append(left_motion)
            right_arm_motions.append(right_motion)
            arm_motions.append(left_motion + right_motion)
        
        arm_motion = np.mean(arm_motions) if arm_motions else 0
        
        # CRITICAL: Check if arms move in alternating pattern (walking)
        # Walking: left-right-left-right alternating swing
        # Fighting: both arms move together or randomly
        is_alternating = False
        if len(left_arm_motions) >= 6 and len(right_arm_motions) >= 6:
            # Calculate correlation between left and right arm motion
            # Walking: negative correlation (when left moves, right doesn't)
            # Fighting: positive or no correlation
            try:
                correlation = np.corrcoef(left_arm_motions, right_arm_motions)[0, 1]
                is_alternating = correlation < -0.2  # negative correlation = alternating
            except:
                is_alternating = False
        
        # Check arm motion variance - fighting has erratic arm movement
        arm_variance = np.std(arm_motions) / max(np.mean(arm_motions), 1e-6)
    else:
        is_alternating = False
        arm_variance = 0
    
    # If alternating arm pattern (walking swing), NOT fighting
    if is_alternating and is_consistent_direction:
        return False, 0.0
    
    # Fighting: high arm motion WITHOUT alternating pattern
    has_aggressive_arms = arm_motion > 0.18 and not is_alternating  # increased threshold
    has_active_arms = arm_motion > 0.10 and arm_variance > 0.4
    
    # 4. Check bbox instability - fighting causes erratic bbox changes
    sizes = [(h[2] - h[0]) * (h[3] - h[1]) for h in history]
    aspects = [(h[2] - h[0]) / max(h[3] - h[1], 1) for h in history]
    
    size_variation = np.std(sizes) / max(np.mean(sizes), 1) if len(sizes) > 3 else 0
    aspect_variation = np.std(aspects) if len(aspects) > 3 else 0
    
    very_erratic_size = size_variation > 0.20
    erratic_size = size_variation > 0.12
    erratic_shape = aspect_variation > 0.10
    
    # ULTRA STRICT Decision Logic - Reject walking at all costs
    
    # Additional safety check: if motion is too smooth or directional, NOT fighting
    if is_smooth_motion or (is_consistent_direction and not very_erratic_size):
        return False, 0.0
    
    # Path A: Physical contact + aggressive motion
    # (bodies overlapping + rapid jerky motion + aggressive arm movement)
    # MUST have all conditions
    if overlap_count >= 1 and is_jerky_motion and is_rapid and has_aggressive_arms:
        # Double check not walking
        if not is_walking_speed:
            conf = min(0.85, 0.65 + overlap_count * 0.1)
            return True, conf
    
    # Path B: Very close proximity + VERY aggressive signals
    # (within touching distance + multiple strong signals)
    if close_proximity_count >= 1 and not is_walking_speed:
        strong_signals = sum([
            has_aggressive_arms,
            is_rapid and is_jerky_motion,
            very_erratic_size,
            erratic_shape
        ])
        # Increased requirement: need ALL 4 signals
        if strong_signals == 4:
            conf = min(0.80, 0.60 + strong_signals * 0.08)
            return True, conf
    
    # Path C: REMOVED - too risky for false positives
    # Group proximity alone is not enough, must have physical contact
    
    # Default: NOT fighting
    return False, 0.0


# --------------------------- TRACKING --------------------------------


class PersonTracker:
    """IOU-based tracker with appearance templates for robust ID preservation."""

    def __init__(self, iou_thresh=0.15, max_missed=15):
        self.tracks = {}  # id -> dict(bbox, appearance_template, last_seen, probs_history, action_state)
        self.next_id = 0
        self.iou_thresh = iou_thresh  # Lower threshold for better separation handling
        self.max_missed = max_missed
        self.frame_idx = 0

    @staticmethod
    def iou(b1, b2):
        x1, y1, x2, y2 = b1
        a1, b1y, a2, b2y = b2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1y)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2y)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2y - b1y)
        return inter / max(area1 + area2 - inter, 1e-6)
    
    @staticmethod
    def center_distance(b1, b2):
        """Calculate normalized center distance between two boxes."""
        cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
        cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    @staticmethod
    def size_similarity(b1, b2):
        """Calculate size similarity (0-1, higher is more similar)."""
        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
        area1, area2 = w1 * h1, w2 * h2
        ratio = min(area1, area2) / max(area1, area2, 1e-6)
        return ratio
    
    @staticmethod
    def extract_appearance_features(frame, bbox):
        """
        Extract appearance features (color histogram) from a bbox.
        Returns normalized histogram or None if error.
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            
            # Use HSV color space (better for clothing color)
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Multi-channel histogram: H (32 bins) + S (32 bins) for better discrimination
            hist_h = cv2.calcHist([crop_hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([crop_hsv], [1], None, [32], [0, 256])
            
            # Normalize
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            
            # Concatenate H and S histograms
            hist = np.vstack([hist_h, hist_s])
            return hist
        except:
            return None
    
    @staticmethod
    def compare_appearance(hist1, hist2):
        """
        Compare two appearance histograms.
        Returns similarity score (0-1, higher = more similar).
        """
        if hist1 is None or hist2 is None:
            return 0.5  # neutral
        try:
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0, min(1, similarity))  # clip to [0, 1]
        except:
            return 0.5

    def update(self, detections, frame=None):
        """
        detections: Nx5 array [x1,y1,x2,y2,conf] for persons.
        frame: BGR frame for appearance matching (optional but recommended)
        Returns list of (track_id, bbox, conf).
        """
        self.frame_idx += 1
        dets = detections.tolist() if isinstance(detections, np.ndarray) else detections
        unmatched = set(range(len(dets)))
        # mark tracks as missed
        for tid, t in list(self.tracks.items()):
            t["missed"] = t.get("missed", 0) + 1

        # IMPROVED matching: Compare with stored appearance TEMPLATE
        # This prevents ID swaps even when bboxes merge then separate
        for tid, t in list(self.tracks.items()):
            best_score, best_j = 0.0, -1
            
            for j in unmatched:
                iou = self.iou(t["bbox"], dets[j][:4])
                # Allow low IOU for people who separate after overlap
                if iou < self.iou_thresh:
                    # Still consider if very close (adjacent people)
                    center_dist = self.center_distance(t["bbox"], dets[j][:4])
                    diag = ((t["bbox"][2] - t["bbox"][0]) ** 2 + (t["bbox"][3] - t["bbox"][1]) ** 2) ** 0.5
                    if center_dist > diag * 2.0:  # too far away
                        continue
                
                # Calculate center distance (normalized by diagonal)
                diag = ((t["bbox"][2] - t["bbox"][0]) ** 2 + (t["bbox"][3] - t["bbox"][1]) ** 2) ** 0.5
                center_dist = self.center_distance(t["bbox"], dets[j][:4])
                center_score = 1.0 - min(center_dist / max(diag * 1.5, 1), 1.0)  # allow farther distance
                
                # Calculate size similarity
                size_sim = self.size_similarity(t["bbox"], dets[j][:4])
                
                # Compare with stored appearance TEMPLATE (not current bbox)
                appearance_sim = 0.3  # low default (penalize if no template)
                if frame is not None and "appearance_template" in t and t["appearance_template"] is not None:
                    # Extract current appearance
                    current_appearance = self.extract_appearance_features(frame, dets[j][:4])
                    appearance_sim = self.compare_appearance(t["appearance_template"], current_appearance)
                
                # HEAVY weight on appearance template matching
                # This is KEY to prevent ID swaps
                if "appearance_template" in t and t["appearance_template"] is not None:
                    # With template: 20% IOU + 15% center + 15% size + 50% appearance
                    score = 0.20 * max(iou, 0) + 0.15 * max(center_score, 0) + 0.15 * size_sim + 0.50 * appearance_sim
                else:
                    # No template yet: fall back to geometric features
                    score = 0.40 * max(iou, 0) + 0.35 * max(center_score, 0) + 0.25 * size_sim
                
                if score > best_score:
                    best_score, best_j = score, j
            
            if best_j >= 0:
                bbox = dets[best_j][:4]
                conf = dets[best_j][4]
                
                # Update appearance template periodically (every 10 frames)
                # Only update when bbox is NOT merged (area is reasonable)
                if frame is not None:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    is_reasonable_size = 1000 < bbox_area < 200000  # not too small, not too large
                    should_update = (self.frame_idx - t.get("template_updated", 0)) > 10
                    
                    if is_reasonable_size and ("appearance_template" not in t or should_update):
                        new_template = self.extract_appearance_features(frame, bbox)
                        if new_template is not None:
                            # Exponential moving average for template stability
                            if "appearance_template" in t and t["appearance_template"] is not None:
                                t["appearance_template"] = 0.7 * t["appearance_template"] + 0.3 * new_template
                            else:
                                t["appearance_template"] = new_template
                            t["template_updated"] = self.frame_idx
                
                t["bbox"] = bbox
                t["conf"] = conf
                t["last_seen"] = self.frame_idx
                t["missed"] = 0
                unmatched.discard(best_j)

        # new tracks
        for j in unmatched:
            bbox = dets[j][:4]
            conf = dets[j][4]
            
            # Extract initial appearance template
            appearance_template = None
            if frame is not None:
                appearance_template = self.extract_appearance_features(frame, bbox)
            
            self.tracks[self.next_id] = {
                "bbox": bbox,
                "conf": conf,
                "appearance_template": appearance_template,  # Store initial appearance
                "template_updated": self.frame_idx,
                "last_seen": self.frame_idx,
                "missed": 0,
                "probs": deque(maxlen=SMOOTH_WINDOW),
                "action": "normal",
                "action_conf": 0.0,
                "last_action_frame": 0,
            }
            self.next_id += 1

        # remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return [(tid, t["bbox"], t.get("conf", 0.0)) for tid, t in self.tracks.items()]


class FireTracker:
    """IOU-based tracker for fire regions with temporal persistence."""

    def __init__(self, iou_thresh=0.25, max_missed=30):
        self.tracks = {}  # id -> dict(bbox, conf, last_seen, missed, conf_history)
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed  # keep fire track alive longer
        self.frame_idx = 0

    @staticmethod
    def iou(b1, b2):
        x1, y1, x2, y2 = b1
        a1, b1y, a2, b2y = b2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1y)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2y)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2y - b1y)
        return inter / max(area1 + area2 - inter, 1e-6)

    def update(self, detections):
        """
        detections: list of (bbox, conf) for fire regions
        Returns list of (track_id, bbox, smoothed_conf, is_active)
        """
        self.frame_idx += 1
        dets = list(detections)
        unmatched = set(range(len(dets)))
        
        # mark tracks as missed
        for tid, t in list(self.tracks.items()):
            t["missed"] = t.get("missed", 0) + 1

        # match by IOU
        for tid, t in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                iou = self.iou(t["bbox"], dets[j][0])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thresh and best_j >= 0:
                bbox, conf = dets[best_j]
                t["bbox"] = bbox
                t["conf"] = conf
                t["conf_history"].append(conf)
                t["last_seen"] = self.frame_idx
                t["missed"] = 0
                unmatched.discard(best_j)

        # new tracks
        for j in unmatched:
            bbox, conf = dets[j]
            self.tracks[self.next_id] = {
                "bbox": bbox,
                "conf": conf,
                "conf_history": deque([conf], maxlen=15),
                "last_seen": self.frame_idx,
                "missed": 0,
                "first_seen": self.frame_idx,
            }
            self.next_id += 1

        # remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        # return all active tracks (including those with missed frames for persistence)
        results = []
        for tid, t in self.tracks.items():
            smoothed_conf = float(np.mean(t["conf_history"]))
            is_active = t["missed"] == 0  # currently detected
            results.append((tid, t["bbox"], smoothed_conf, is_active))
        
        return results


# -------------------------- MAIN PIPELINE -----------------------------


def process_video_with_tracking(
    video_path,
    slowfast_checkpoint,
    yolo_model="checkpoints/customyolov11m.pt",
    out_path="output.mp4",
    conf_threshold=0.5,
    show=False,
    progress_callback=None,
    frame_callback=None,
):
    """
    Main function used by app.py.
    Returns (out_path, alerts) where alerts is a list of segments.
    """
    logger.info(f"Processing video: {video_path}")

    # Load models
    person_yolo = load_yolo_model(yolo_model)
    action_model = load_action_model(slowfast_checkpoint)
    pose_model = load_pose_model()
    fire_yolo = load_fire_model("checkpoints/best_model_fire.pt")
    fire_cnn = load_fire_verification_cnn("D:/FA25/realtime_app/checkpoints/fire_red_cnn.pth")
    tracker = PersonTracker(iou_thresh=0.15, max_missed=20)  # Very low IOU + appearance templates
    fire_tracker = FireTracker(iou_thresh=0.20, max_missed=90)  # Fire tracker with 3s persistence

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    persistence_frames = int(ACTION_PERSISTENCE_SECONDS * fps)

    frame_buffer_rgb = defaultdict(list)   # track_id -> list[np.array(H,W,3)]
    frame_buffer_out = []
    abnormal_frames = defaultdict(list)    # frame_idx -> list[dict]

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            if progress_callback and total_frames:
                if frame_idx % 10 == 0:
                    progress_callback(frame_idx / total_frames * 100.0)

            # ---------------- Fire stream with tracking ----------------
            fire_detections = []  # raw detections this frame
            if fire_yolo is not None:
                fire_res = fire_yolo(
                    frame_bgr,
                    conf=FIRE_DETECTION_CONFIG["conf"],
                    iou=FIRE_DETECTION_CONFIG["iou"],
                    imgsz=FIRE_DETECTION_CONFIG["imgsz"],
                    max_det=FIRE_DETECTION_CONFIG["max_det"],
                    classes=FIRE_DETECTION_CONFIG["classes"],
                    verbose=False,
                )
                for r in fire_res:
                    for b in (r.boxes or []):
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf.cpu().numpy()[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area < FIRE_DETECTION_CONFIG["min_area"]:
                            continue
                        
                        # Additional color-based fire filter to reject skin tones
                        x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                        crop = frame_bgr[y1_int:y2_int, x1_int:x2_int]
                        if crop.size > 0:
                            # Fire should have high saturation and specific hue (orange/red/yellow)
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            
                            # Fire characteristics:
                            # Hue: 0-30 (red-orange-yellow)
                            # Saturation: > 100 (vivid colors)
                            # Value: > 100 (bright)
                            fire_mask = (
                                ((h < 30) | (h > 150)) &  # red/orange/yellow hue
                                (s > 80) &                 # high saturation (not skin tone)
                                (v > 80)                   # bright
                            )
                            fire_ratio = np.sum(fire_mask) / fire_mask.size
                            
                            # Need at least 30% of region to have fire colors
                            if fire_ratio < 0.30:
                                continue
                        
                        # CNN verification with higher threshold
                        ok, fire_conf = verify_fire_with_cnn(fire_cnn, frame_bgr, [x1, y1, x2, y2], conf_threshold=0.50)
                        if not ok:
                            continue
                        fire_detections.append(([int(x1), int(y1), int(x2), int(y2)], fire_conf))

            # Update fire tracker and get tracked fire regions
            fire_tracks = fire_tracker.update(fire_detections)
            
            # Process tracked fires (includes persistence)
            fire_events = []
            for fire_tid, fire_bbox, fire_conf, is_active in fire_tracks:
                fire_events.append((fire_tid, fire_bbox, fire_conf, is_active))
                abnormal_frames[frame_idx].append(
                    {
                        "track_id": int(fire_tid),
                        "action": "fire",
                        "confidence": float(fire_conf),
                        "bbox": list(map(int, fire_bbox)),
                        "source": "fire",
                    }
                )

            # ---------------- Person stream ----------------
            # YOLO person detection - High recall but filter out static objects
            dets = []
            res = person_yolo(
                frame_bgr, 
                conf=0.20,          # Balanced threshold - catch people but not objects
                iou=0.35,           # Lower IOU for overlapping people
                imgsz=1280,         # Increased resolution for better detection
                max_det=300,        # Very high limit for crowded scenes
                agnostic_nms=False,
                half=False,         # Full precision for better accuracy
                verbose=False
            )
            for r in res:
                for b in (r.boxes or []):
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf.cpu().numpy()[0])
                    cls = int(b.cls.cpu().numpy()[0]) if hasattr(b, "cls") else 0
                    name = str(r.names.get(cls, "")).lower() if hasattr(r, "names") else "person"
                    if name != "person":
                        continue
                    # Allow extremely small people (very distant, occluded)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if bbox_width < 8 or bbox_height < 8:
                        continue
                    # Skip very elongated or flat boxes (likely errors)
                    aspect_ratio = bbox_width / max(bbox_height, 1)
                    if aspect_ratio > 3 or aspect_ratio < 0.2:
                        continue
                    
                    # CRITICAL: Multi-stage filter to reject ALL non-human objects (wood, furniture)
                    try:
                        crop_for_pose = frame_bgr[int(y1):int(y2), int(x1):int(x2)]
                        if crop_for_pose.size == 0:
                            continue
                        
                        # Stage 1: Skin tone detection - humans MUST have skin pixels
                        # Wood/objects don't have human skin tones
                        crop_hsv = cv2.cvtColor(crop_for_pose, cv2.COLOR_BGR2HSV)
                        # Human skin HSV ranges (multiple ranges for different skin tones)
                        # Range 1: Light to medium skin
                        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
                        # Range 2: Darker skin tones
                        lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
                        upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
                        
                        skin_mask1 = cv2.inRange(crop_hsv, lower_skin1, upper_skin1)
                        skin_mask2 = cv2.inRange(crop_hsv, lower_skin2, upper_skin2)
                        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
                        
                        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                        # Require at least 8% skin pixels (face, hands, arms)
                        if skin_ratio < 0.08:
                            continue  # No skin detected = not a person (wood/object)
                        
                        # Stage 2: Pose keypoints check
                        test_pose_res = pose_model(crop_for_pose, verbose=False)
                        if len(test_pose_res) == 0 or test_pose_res[0].keypoints is None:
                            continue  # No pose detected = not a person
                        
                        kpts = test_pose_res[0].keypoints.data[0]
                        visible_kpts = torch.sum(kpts[:, 2] > 0.3).item()
                        
                        # Require at least 6 visible keypoints
                        if visible_kpts < 6:
                            continue  # Too few keypoints = object, not person
                        
                        # Stage 3: Anatomical validation
                        # Check if detected keypoints form a valid human structure
                        key_body_parts = [0, 5, 6, 11, 12]  # nose, shoulders, hips
                        key_visible = sum(1 for idx in key_body_parts if idx < len(kpts) and kpts[idx, 2] > 0.3)
                        
                        if key_visible < 3:  # Need at least 3 key body parts visible
                            continue  # Invalid body structure = not a person
                        
                    except Exception as e:
                        continue  # Any error = skip detection
                    
                    dets.append([x1, y1, x2, y2, conf])
            dets_np = np.array(dets, dtype=float) if dets else np.zeros((0, 5), dtype=float)

            tracks = tracker.update(dets_np, frame=frame_bgr)

            # For each track, update RGB/pose buffers and maybe run classifier
            for tid, bbox, pconf in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                track_state = tracker.tracks[tid]
                track_state["_tid"] = tid  # store tid for proximity check
                
                # Initialize action state if not exists
                if "action" not in track_state:
                    track_state["action"] = "normal"
                    track_state["action_conf"] = 0.0

                # ONLY draw bbox for DANGEROUS behaviors (fall, fighting)
                # Normal people will NOT be shown
                draw_action_early = track_state.get("lock_action") or track_state.get("action", "normal")
                draw_conf_early = track_state.get("lock_conf", track_state.get("action_conf", pconf))
                
                # Only draw if action is dangerous (NOT normal)
                if draw_action_early in ["fall", "fighting"]:
                    if draw_action_early == "fall":
                        color_early = (0, 255, 255)  # yellow for fall
                    elif draw_action_early == "fighting":
                        color_early = (0, 0, 255)  # red for fighting
                    
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_early, 3)  # Thicker border
                    label_early = f"{draw_action_early.upper()}"
                    if len(frame_buffer_rgb[tid]) < MIN_FRAMES_FOR_ACTION:
                        label_early += f" (Analyzing...)"
                    else:
                        label_early += f" ({draw_conf_early:.2f})"
                    (tw, th), _ = cv2.getTextSize(label_early, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_early, -1)
                    cv2.putText(frame_bgr, label_early, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Extract pose early for instant detection
                pose = extract_pose(pose_model, frame_bgr, bbox)

                # Instant fall heuristic (before SlowFast) for fast response
                instant_fall, instant_conf = detect_instant_fall(track_state, bbox, h)
                if instant_fall:
                    lock_conf_prev = track_state.get("lock_conf", 0.0)
                    if track_state.get("lock_action") != "fall" or instant_conf > lock_conf_prev:
                        track_state["lock_action"] = "fall"
                        track_state["lock_conf"] = instant_conf
                        track_state["fall_locked"] = True  # Mark as fallen
                        track_state["fall_locked_frame"] = frame_idx
                    track_state["last_action_frame"] = frame_idx
                    abnormal_frames[frame_idx].append(
                        {
                            "track_id": int(tid),
                            "action": "fall",
                            "confidence": track_state.get("lock_conf", instant_conf),
                            "bbox": [x1, y1, x2, y2],
                            "source": "person",
                        }
                    )
                else:
                    # CRITICAL: Override Fighting → Fall if person is clearly lying down
                    # This handles case where person was Fighting then fell, but instant_fall didn't trigger yet
                    current_posture = get_posture_type(bbox, h)
                    if current_posture == 'lying' and track_state.get("lock_action") == "fighting":
                        # Person is lying down but locked to Fighting → override to Fall
                        track_state["lock_action"] = "fall"
                        track_state["lock_conf"] = 0.80  # high confidence override
                        track_state["fall_locked"] = True
                        track_state["fall_locked_frame"] = frame_idx
                        track_state["last_action_frame"] = frame_idx
                        abnormal_frames[frame_idx].append(
                            {
                                "track_id": int(tid),
                                "action": "fall",
                                "confidence": 0.80,
                                "bbox": [x1, y1, x2, y2],
                                "source": "person",
                            }
                        )

                # DISABLE instant_fighting - too many false positives
                # Only use SlowFast classifier for Fighting detection
                # This ensures higher accuracy at the cost of ~0.3s delay

                crop = frame_bgr[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                frame_buffer_rgb[tid].append(crop_rgb)
                if len(frame_buffer_rgb[tid]) > CLIP_LEN:
                    frame_buffer_rgb[tid].pop(0)

                # Run action model only when we have enough frames
                if len(frame_buffer_rgb[tid]) >= MIN_FRAMES_FOR_ACTION:
                    frames_np = np.stack(frame_buffer_rgb[tid], axis=0)  # (T,H,W,3)
                    frames_t = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2) / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
                    frames_t = ((frames_t - mean) / std).unsqueeze(0).to(DEVICE)
                    if frames_t.shape[2] < CLIP_LEN:
                        pad = CLIP_LEN - frames_t.shape[2]
                        frames_t = torch.cat([frames_t, frames_t[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)
                    else:
                        frames_t = frames_t[:, :, -CLIP_LEN:, :, :]
                    video_in = pack_pathway_input(frames_t)
                    pose_seq = torch.stack(
                        [pose for _ in range(CLIP_LEN)], dim=0
                    ).unsqueeze(0)  # (1,T,17,3) simple repeat
                    pose_seq = pose_seq.to(DEVICE)
                    with torch.no_grad():
                        logits = action_model(video_in, pose_seq)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    # temporal smoothing
                    track_state["probs"].append(probs)
                    smoothed = np.mean(track_state["probs"], axis=0)
                    p_norm, p_fall, p_fight = smoothed

                    # Posture-based disambiguation:
                    # If person is lying down, prioritize Fall over Fighting
                    current_posture = get_posture_type(bbox, h)
                    
                    # decide action with posture awareness
                    action = "normal"
                    aconf = float(p_norm)
                    
                    if current_posture == 'lying':
                        # Person is lying on ground - any abnormal action should be Fall
                        # Check if either fall or fighting probability is significant
                        if p_fall >= 0.30 or p_fight >= 0.30:
                            # Lying person with abnormal probability -> classify as Fall
                            action = "fall"
                            aconf = max(float(p_fall), float(p_fight) * 0.9)  # use higher prob
                    else:
                        # Normal threshold-based logic for standing/sitting people
                        if p_fall >= ACTION_THRESHOLDS["fall"] and p_fall >= p_fight:
                            action, aconf = "fall", float(p_fall)
                        elif p_fight >= ACTION_THRESHOLDS["fighting"]:
                            # STRICT Fighting detection: require proximity + motion
                            # Only classify Fighting if person has:
                            # 1. High confidence from SlowFast (>= 0.65)
                            # 2. Another person very close (physical contact - 25% diagonal)
                            # 3. Significant motion (12% movement - not just standing/swaying)
                            has_proximity = has_close_proximity(tid, tracks, bbox, threshold_ratio=0.25)
                            has_motion = has_significant_motion(track_state, min_motion_ratio=0.12)
                            
                            if has_proximity and has_motion:
                                action, aconf = "fighting", float(p_fight)
                            else:
                                # High fighting prob but no proximity/motion -> likely bystander
                                action = "normal"
                                aconf = float(p_norm)

                    # persistence with Fall priority
                    last_action = track_state.get("action", "normal")
                    last_frame = track_state.get("last_action_frame", 0)
                    current_lock = track_state.get("lock_action")
                    is_fallen = track_state.get("fall_locked", False)
                    
                    if action != "normal":
                        track_state["action"] = action
                        track_state["action_conf"] = aconf
                        track_state["last_action_frame"] = frame_idx
                        
                        # Lock logic with Fall priority:
                        # 1. If currently locked to Fall, ONLY update if new action is also Fall with higher conf
                        # 2. If not locked or locked to Fighting, can update to Fall or Fighting
                        if current_lock == "fall":
                            # Fallen person - only update if still Fall with higher confidence
                            if action == "fall" and aconf > track_state.get("lock_conf", 0.0):
                                track_state["lock_action"] = action
                                track_state["lock_conf"] = aconf
                            # Don't switch from Fall to Fighting
                        else:
                            # Not locked to Fall - can lock to any abnormal action
                            if (
                                current_lock != action
                                or aconf > track_state.get("lock_conf", 0.0)
                            ):
                                track_state["lock_action"] = action
                                track_state["lock_conf"] = aconf
                                if action == "fall":
                                    track_state["fall_locked"] = True
                                    track_state["fall_locked_frame"] = frame_idx
                    else:
                        # keep previous abnormal state for persistence window
                        if (
                            last_action != "normal"
                            and frame_idx - last_frame <= persistence_frames
                        ):
                            action = last_action
                            aconf = track_state.get("action_conf", aconf)
                        elif "lock_action" in track_state:
                            action = track_state["lock_action"]
                            aconf = track_state.get("lock_conf", aconf)

                    if action != "normal":
                        abnormal_frames[frame_idx].append(
                            {
                                "track_id": int(tid),
                                "action": action,
                                "confidence": aconf,
                                "bbox": [x1, y1, x2, y2],
                                "source": "person",
                            }
                        )
                
                # Update the early-drawn bbox with final classification results
                # (Already drawn at the beginning, just update track state for next frame)

            # draw fire bboxes with tracking info
            for fire_tid, fire_bbox, fconf, is_active in fire_events:
                x1, y1, x2, y2 = map(int, fire_bbox)
                # Use different color for actively detected vs persisted
                if is_active:
                    color = (0, 165, 255)  # Orange for active fire
                    thickness = 3
                else:
                    color = (0, 100, 200)  # Darker orange for persisted/tracking
                    thickness = 2
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
                label = f"FIRE ({fconf:.2f})"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_buffer_out.append(frame_rgb)
            
            # Send frame to callback for real-time display
            if frame_callback is not None:
                try:
                    frame_callback(frame_rgb, frame_idx, total_frames)
                except Exception as e:
                    logger.debug(f"Frame callback error: {e}")

        # -------- build simple alert segments --------
        alerts = []
        for fidx in sorted(abnormal_frames.keys()):
            for det in abnormal_frames[fidx]:
                alerts.append(
                    {
                        "track_id": det["track_id"],
                        "type": det["action"],
                        "start_time": round(fidx / fps, 2),
                        "end_time": round((fidx + 1) / fps, 2),
                        "duration": round(1.0 / fps, 2),
                        "confidence": round(det["confidence"], 3),
                        "source": det["source"],
                    }
                )
        logger.info(f"Total abnormal events: {len(alerts)}")

    finally:
        cap.release()

    logger.info(f"Writing output video to {out_path}")
    imageio.mimsave(out_path, frame_buffer_out, fps=fps, codec="libx264")
    return out_path, alerts
