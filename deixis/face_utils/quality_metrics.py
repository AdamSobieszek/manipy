# quality_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


from facefusion import state_manager
from facefusion.face_store import get_static_faces, set_static_faces
from facefusion.face_detector import detect_faces, detect_faces_by_angle
from facefusion.face_analyser import create_faces
from facefusion.types import Face, VisionFrame




def get_one_face(faces : List[Face], position : int = 0) -> Optional[Face]:
	if faces:
		position = min(position, len(faces) - 1)
		return faces[position]
	return None


def get_many_faces(vision_frames : List[VisionFrame]) -> List[Face]:
	many_faces : List[Face] = []

	for vision_frame in vision_frames:
		if np.any(vision_frame):
			static_faces = get_static_faces(vision_frame)
			if static_faces:
				many_faces.extend(static_faces)
			else:
				all_bounding_boxes = []
				all_face_scores = []
				all_face_landmarks_5 = []

				for face_detector_angle in [0]:
					if face_detector_angle == 0:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces(vision_frame)
					else:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces_by_angle(vision_frame, face_detector_angle)
					all_bounding_boxes.extend(bounding_boxes)
					all_face_scores.extend(face_scores)
					all_face_landmarks_5.extend(face_landmarks_5)

				if all_bounding_boxes and all_face_scores and all_face_landmarks_5 and state_manager.get_item('face_detector_score') > 0:
					faces = create_faces(vision_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)

					if faces:
						many_faces.extend(faces)
						set_static_faces(vision_frame, faces)
	return many_faces

# ===========================
# Tunables & weights
# ===========================

WEIGHTS: Dict[str, float] = {
    "det": 0.35,       # detector confidence
    "geom": 0.10,      # landmark geometry plausibility / symmetry
    "pose": 0.10,      # roll/pitch/yaw closeness to frontal
    "flip": 0.15,      # ArcFace embedding similarity under horizontal flip
    "sharp": 0.05,     # Laplacian variance in face crop
    "exposure": 0.10,  # non-over/under exposure + usable dynamic range
    "center": 0.10,    # face roughly centered like FFHQ
    "occl": 0.05       # sunglasses / hat heuristics (eyes/brows visibility)
}

# Default accept threshold. Tune on your validation set to hit target FPR/TPR.
DEFAULT_ACCEPT_THRESHOLD = 70.0

# Detector calibration (map raw detector score to [0,1])
DET_SCORE_MIN = 0.30
DET_SCORE_MAX = 0.95

# Sharpness calibration (Laplacian variance) — adjust by your generator’s outputs
LAPLACIAN_GOOD = 200.0
LAPLACIAN_BAD  = 40.0

# Exposure calibration (mean/std in [0,255])
EXPO_LOW  = 60.0
EXPO_HIGH = 190.0
EXPO_STD_MIN = 35.0

# Pose preferences for FFHQ-like crops (degrees)
POSE_MAX_YAW   = 30.0
POSE_MAX_PITCH = 20.0
POSE_MAX_ROLL  = 15.0

# Centering tolerance (fraction of frame)
CENTER_TOL = 0.18

# Eye/eyebrow occlusion heuristics
EYE_GRAD_MIN = 6.0    # mean Sobel grad magnitude inside eye polygons
EYE_LUMA_MIN = 75.0   # mean brightness inside eye polygons
BROW_GRAD_MIN = 5.0
TOP_MARGIN_MIN_FRAC = 0.03  # eyebrow-to-top margin as fraction of bbox height


@dataclass
class QualityResult:
    score: float
    subscores: Dict[str, float]
    reject: bool
    reasons: List[str]


# ===========================
# Helpers
# ===========================

def _has_68(lmk: np.ndarray) -> bool:
    try:
        return isinstance(lmk, np.ndarray) and lmk.shape[0] >= 68 and lmk.shape[1] >= 2
    except Exception:
        return False

def _pose_from_5(lmk5: np.ndarray) -> Tuple[float,float,float]:
    # order-agnostic: compute eye centers via min-x as right, max-x as left
    pts = np.array(lmk5, dtype=float)
    # guess indices: eyes are the two with smallest y (topmost); mouth corners the two lowest; nose the remaining one
    ys = pts[:,1]
    ids = np.argsort(ys)
    eyes = pts[ids[:2]]; rest = pts[ids[2:]]
    nose = rest[np.argmin(rest[:,1])]
    mouth = rest[np.argmax(rest[:,1])]
    # roll from eyes
    dx, dy = (eyes[1] - eyes[0])
    roll = np.degrees(np.arctan2(dy, dx))
    # yaw from nose x offset vs eye midpoint
    eye_mid = eyes.mean(axis=0)
    yaw = np.degrees(np.arctan2(nose[0]-eye_mid[0], np.linalg.norm(eyes[1]-eyes[0])+1e-6))
    # pitch from vertical ratios
    d1 = nose[1] - eye_mid[1]; d2 = mouth[1] - nose[1]
    pitch = np.degrees(np.arctan((d1/(d2+1e-6)) - 1.0))
    return float(yaw), float(pitch), float(roll)

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _scale01(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def _crop_from_bbox(img: np.ndarray, bbox_xyxy: np.ndarray, pad_frac: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(max(0, x1 - pad_frac * bw))
    y1 = int(max(0, y1 - pad_frac * bh))
    x2 = int(min(w - 1, x2 + pad_frac * bw))
    y2 = int(min(h - 1, y2 + pad_frac * bh))
    return img[y1:y2, x1:x2]

def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _mean_grad_mag(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mask is not None:
        vals = mag[mask > 0]
        return float(vals.mean()) if vals.size else 0.0
    return float(mag.mean())

def _polygon_mask(h: int, w: int, pts: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_i = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, pts_i, 255)
    return mask

def _estimate_pose_from_68(lmk68: np.ndarray) -> Tuple[float, float, float]:
    """
    Approximate yaw/pitch/roll from 68 landmarks.
    Roll: slope of eye centers.
    Yaw: inter-ocular distance asymmetry.
    Pitch: vertical ratio between eyes–nose–mouth.
    Returns degrees (yaw, pitch, roll).
    """
    # Indexing assumes iBUG-68: R eye = 36..41, L eye = 42..47, nose tip=30, mouth corners=48,54
    reye = lmk68[36:42].mean(axis=0)
    leye = lmk68[42:48].mean(axis=0)
    nose = lmk68[30]
    mouth_l = lmk68[48]
    mouth_r = lmk68[54]

    # Roll
    dx, dy = (leye - reye)
    roll = np.degrees(np.arctan2(dy, dx))

    # Yaw (eye distance asymmetry wrt nose x)
    eye_mid = 0.5 * (leye + reye)
    yaw = np.degrees(np.arctan2(nose[0] - eye_mid[0], np.linalg.norm(leye - reye) + 1e-6))

    # Pitch (eyes–nose–mouth vertical alignment)
    eyes_y = eye_mid[1]
    mouth_mid_y = 0.5 * (mouth_l[1] + mouth_r[1])
    d1 = nose[1] - eyes_y
    d2 = mouth_mid_y - nose[1]
    ratio = (d1 + 1e-6) / (d2 + 1e-6)
    # Pitch sign heuristic: nose above/below mid of eyes-mouth
    pitch = np.degrees(np.arctan(ratio - 1.0))

    return float(yaw), float(pitch), float(roll)

def _geom_symmetry_score(lmk68: np.ndarray, bbox_w: float) -> float:
    """Symmetry/geometry plausibility in [0,1]."""
    # Pairs (iBUG-68 symmetric indices)
    pairs = [
        (17,26),(18,25),(19,24),(20,23),(21,22),  # brows
        (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),  # eyes
        (31,35),(32,34),  # nose wings
        (48,54),(49,53),(50,52),(61,63),(60,64),(67,65),  # mouth
        (0,16),(1,15),(2,14),(3,13),(4,12),(5,11),(6,10),(7,9)  # jaw
    ]
    # Mirror across the vertical line through nose bridge (points 27..30)
    center_x = lmk68[27:31,0].mean()
    diffs = []
    for i,j in pairs:
        xi = lmk68[i,0]; xj = lmk68[j,0]
        yi = lmk68[i,1]; yj = lmk68[j,1]
        # mirrored partner distance
        d = np.hypot((xi - (2*center_x - xj)), (yi - yj))
        diffs.append(d)
    mean_diff = np.mean(diffs) if diffs else 1e9
    # Normalize by bbox width; smaller is better
    norm = mean_diff / (bbox_w + 1e-6)
    # Map: <=0.03 -> 1.0, >=0.12 -> 0.0 (tune if needed)
    return np.sqrt(_clamp01(1.0 - _scale01(norm, 0.03, 0.12)))

def _flip_landmarks5(lmk5: np.ndarray, img_w: int) -> np.ndarray:
    """Flip 5 landmarks horizontally and swap L/R (assumes order [Leye, Reye, Nose, Lmouth, Rmouth] or the reverse)."""
    flipped = lmk5.copy()
    flipped[:,0] = img_w - flipped[:,0]
    # Try both common orderings robustly by swapping 0<->1 and 3<->4
    flipped[[0,1]] = flipped[[1,0]]
    flipped[[3,4]] = flipped[[4,3]]
    return flipped

def _centering_score(bbox: np.ndarray, img_w: int, img_h: int) -> float:
    x1,y1,x2,y2 = bbox
    cx = 0.5*(x1+x2)/img_w
    cy = 0.5*(y1+y2)/img_h
    dx = abs(cx - 0.5)
    dy = abs(cy - 0.5)
    d = np.hypot(dx, dy)
    # <= CENTER_TOL -> 1.0, >= 2*CENTER_TOL -> 0.0
    return _clamp01(1.0 - _scale01(d, CENTER_TOL, 2*CENTER_TOL))

def _occlusion_score(gray: np.ndarray, lmk68: Optional[np.ndarray], bbox_xyxy: np.ndarray) -> float:
    """
    Simple sunglasses/hat heuristic using eyes & brows visibility and top margin.
    Returns [0,1]; lower means likely occluded.
    """
    if lmk68 is None or lmk68.shape[0] < 68:
        return 0.5  # unknown

    h, w = gray.shape[:2]
    x1,y1,x2,y2 = bbox_xyxy.astype(int)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Eye polygons (iBUG indices)
    right_eye = lmk68[36:42]
    left_eye  = lmk68[42:48]
    # Eyebrows rectangles
    brows = lmk68[17:27]
    brow_min_y = brows[:,1].min()
    top_margin_frac = (brow_min_y - y1) / float(bh)

    # Masks
    mask_re = _polygon_mask(h,w,right_eye)
    mask_le = _polygon_mask(h,w,left_eye)
    # Slight dilate to include rims
    mask_re = cv2.dilate(mask_re, np.ones((5,5), np.uint8), iterations=1)
    mask_le = cv2.dilate(mask_le, np.ones((5,5), np.uint8), iterations=1)

    # Stats inside masks
    re_luma = float(gray[mask_re>0].mean()) if np.any(mask_re) else 255.0
    le_luma = float(gray[mask_le>0].mean()) if np.any(mask_le) else 255.0
    re_grad = _mean_grad_mag(gray, mask_re)
    le_grad = _mean_grad_mag(gray, mask_le)

    eyes_ok = (min(re_grad, le_grad) >= EYE_GRAD_MIN) or (min(re_luma, le_luma) >= EYE_LUMA_MIN)
    brows_ok = True
    if top_margin_frac < TOP_MARGIN_MIN_FRAC:
        # Very little space above brows (big hat pulled low)
        brows_ok = False

    # Score:
    if eyes_ok and brows_ok:
        return 1.0
    if (eyes_ok and not brows_ok) or (not eyes_ok and brows_ok):
        return 0.5
    return 0.0


# ===========================
# Main scoring API
# ===========================
def score_stylegan2_ffhq(vision_frame: VisionFrame,
                         accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD) -> QualityResult:
                         
    h, w = vision_frame.shape[:2]
    faces = get_many_faces([vision_frame])

    # --- new: reject if multiple faces detected ---
    if faces and len(faces) >= 2:
        return QualityResult(
            score=0.0,
            subscores={},
            reject=True,
            reasons=[f"multiple_faces_detected:{len(faces)}"]
        )

    if not faces:
        return QualityResult(score=0.0, subscores={}, reject=True, reasons=["no_face_detected"])

    # Use the best-detected face
    face = max(faces, key=lambda f: f.score_set.get('detector', 0.0))  # type: Face
    x1, y1, x2, y2 = face.bounding_box
    bbox_w = max(1.0, x2 - x1)

    # --- det ---
    det_raw = float(face.score_set.get('detector', 0.0))
    s_det = _scale01(det_raw, DET_SCORE_MIN, DET_SCORE_MAX)

    lmk68 = face.landmark_set.get('68')
    lmk5  = face.landmark_set.get('5/68')

    # --- geom ---
    s_geom = _geom_symmetry_score(lmk68, bbox_w) if (lmk68 is not None and lmk68.shape[0] >= 68) else 0.5

    # --- pose ---
    if lmk68 is not None and lmk68.shape[0] >= 68:
        yaw, pitch, roll = _estimate_pose_from_68(lmk68)
        yaw_p = _clamp01(1.0 - abs(yaw)   / POSE_MAX_YAW)
        pit_p = _clamp01(1.0 - abs(pitch) / POSE_MAX_PITCH)
        rol_p = _clamp01(1.0 - abs(roll)  / POSE_MAX_ROLL)
        s_pose = 0.4*yaw_p + 0.3*pit_p + 0.3*rol_p
    else:
        s_pose = 0.5
    # --- flip consistency (ArcFace) ---
    s_flip = 0.5
    try:
        from facefusion.face_recognizer import calculate_face_embedding
        if lmk5 is not None:
            # ensure we have a base embedding
            emb_ref = face.embedding_norm
            if emb_ref is None or not np.isfinite(emb_ref).all():
                _, emb_ref = calculate_face_embedding(vision_frame, lmk5)

            flipped_img = cv2.flip(vision_frame, 1)
            lmk5f = _flip_landmarks5(np.array(lmk5, dtype=float), img_w=w)
            _, emb_flip = calculate_face_embedding(flipped_img, lmk5f)

            cos = _cos_sim(emb_ref, emb_flip)
            s_flip = _clamp01((cos - 0.2) / (0.95 - 0.2))
    except Exception as e:
        # optional: surface the reason to your caller for debugging
        print(f"[flip] fallback due to: {e}")
        s_flip = 0.5

    # --- sharpness & exposure (use face crop) ---
    crop = _crop_from_bbox(vision_frame, np.array([x1,y1,x2,y2]))
    if crop.size == 0:
        crop = vision_frame
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lapv = _laplacian_var(gray)
    s_sharp = _clamp01((lapv - LAPLACIAN_BAD) / (LAPLACIAN_GOOD - LAPLACIAN_BAD))

    # --- exposure (band-pass on mean + contrast) ---
    mean = float(gray.mean())
    std  = float(gray.std())

    # bandpass: 1 inside [EXPO_LOW, EXPO_HIGH]; linearly down to 0 outside
    if mean < EXPO_LOW:
        band = _scale01(mean, 0.0, EXPO_LOW)                # 0..1
    elif mean > EXPO_HIGH:
        band = _scale01(255.0 - mean, 0.0, 255.0-EXPO_HIGH) # 0..1
    else:
        band = 1.0

    spread = _scale01(std, EXPO_STD_MIN, 90.0)               # 0..1
    s_exposure = 0.7*band + 0.3*spread

    # --- centering ---
    s_center = _centering_score(np.array([x1,y1,x2,y2]), w, h)

    # --- occlusion heuristic (eyes/brows/hat) ---
    full_gray = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY)

    use68 = _has_68(lmk68)

    # GEOM
    s_geom = _geom_symmetry_score(lmk68, bbox_w) if use68 else 0.7  # a bit optimistic rather than 0.5

    # POSE
    if use68:
        yaw, pitch, roll = _estimate_pose_from_68(lmk68)
    else:
        yaw, pitch, roll = _pose_from_5(lmk5)
    yaw_p = _clamp01(1.0 - abs(yaw)   / POSE_MAX_YAW)
    pit_p = _clamp01(1.0 - abs(pitch) / POSE_MAX_PITCH)
    rol_p = _clamp01(1.0 - abs(roll)  / POSE_MAX_ROLL)
    s_pose = 0.4*yaw_p + 0.3*pit_p + 0.3*rol_p

    # OCCL
    s_occl = _occlusion_score(full_gray, lmk68, np.array([x1,y1,x2,y2])) if use68 else 0.5
    # Aggregate
    subs = {
        "det": s_det, "geom": s_geom, "pose": s_pose, "flip": s_flip,
        "sharp": s_sharp, "exposure": s_exposure, "center": s_center, "occl": s_occl
    }
    weighted = sum(WEIGHTS[k] * subs[k] for k in subs)
    score = 100.0 * weighted

    # Reject reasons
    reasons: List[str] = []
    if s_det < 0.45: reasons.append("low_detector_confidence")
    if s_geom < 0.45: reasons.append("landmark_geometry_implausible")
    if s_pose < 0.45: reasons.append("non_frontal_pose")
    if s_flip < 0.45: reasons.append("identity_inconsistent_under_flip")
    if s_sharp < 0.40: reasons.append("blurry_face")
    if s_exposure < 0.025: reasons.append("poor_exposure")
    if s_center < 0.35: reasons.append("off_center")
    if s_occl < 0.40: reasons.append("likely_occluded_eyes_or_hat")

    reject = (score < accept_threshold) or bool(reasons)
    dbg = {
    "have_68": bool(_has_68(lmk68)),
    "len_5": None if lmk5 is None else int(np.array(lmk5).shape[0]),
    "lap_var": lapv,
    "mean": mean,
    "std": std,
    "yaw_pitch_roll": (yaw, pitch, roll),
}
    print(dbg)
    return QualityResult(score=score, subscores=subs, reject=reject, reasons=reasons)


# ===========================
# Convenience wrapper
# ===========================

def should_reject(vision_frame: VisionFrame,
                  accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD) -> Tuple[bool, float, Dict[str,float], List[str]]:
    """
    Returns (reject, score, subscores, reasons)
    """
    r = score_stylegan2_ffhq(vision_frame, accept_threshold)
    return r.reject, r.score, r.subscores, r.reasons