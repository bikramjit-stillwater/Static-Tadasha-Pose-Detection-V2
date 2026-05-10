"""
Pose analyzer with visibility checking.

Key changes from previous version:
  1. Each landmark's visibility score is checked before use.
  2. Each step has a list of CRITICAL landmarks - if any are not visible
     (visibility < threshold), that step is marked as 'cannot evaluate'
     instead of producing a fake score.
  3. Frames where fewer than 4 steps are evaluable are SKIPPED entirely
     (excluded from averages and from "best frame" selection).
  4. The final report includes 'visibility_issues' so the user knows
     which body parts were missing.
"""

import cv2
import os
import math
from src.pose_detector import PoseDetector
from src.scorer import calculate_angle, validate_tadasana

MIN_QUALITY_SCORE = 50
VISIBILITY_THRESHOLD = 0.5     # landmarks below this are considered NOT visible
MIN_EVALUABLE_STEPS_PER_FRAME = 4   # a frame needs at least this many evaluable steps
MIN_FRAMES_FOR_VALID_VIDEO = 0.30   # 30% of detected frames must be analyzable

POSE_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# Critical landmarks per step - if ANY of these is not visible,
# the step is not evaluable for that frame
STEP_CRITICAL_LANDMARKS = {
    1: ["left_ankle", "right_ankle", "left_hip", "right_hip"],                    # Stance
    2: ["left_shoulder", "right_shoulder", "left_hip", "right_hip",                # Body Balance
        "left_ankle", "right_ankle"],
    3: ["left_hip", "right_hip", "left_knee", "right_knee",                        # Legs & Knees
        "left_ankle", "right_ankle"],
    4: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],               # Spine
    5: ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",            # Shoulders & Arms
        "left_wrist", "right_wrist"],
    6: ["nose", "left_shoulder", "right_shoulder"],                                # Head & Neck
}


def extract_xy(landmarks, w, h, idx):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angle_from_vertical(p_top, p_bottom):
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# -----------------------------------------------------------------------------
# Visibility checking
# -----------------------------------------------------------------------------
def landmark_is_visible(lms, idx):
    """Check if a single landmark is visible enough to trust.

    Two checks:
      1. MediaPipe's own visibility score >= threshold
      2. The (x, y) is inside the image (not extrapolated off-screen)
    """
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESHOLD:
        return False
    # Normalized coords should be within [0, 1] - outside means MediaPipe
    # is guessing about a position outside the frame
    if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
        return False
    return True


def get_step_visibility(lms):
    """Return a dict {step_num: bool} indicating whether each step's
    critical landmarks are all visible."""
    result = {}
    for step_num, names in STEP_CRITICAL_LANDMARKS.items():
        all_visible = True
        for name in names:
            idx = POSE_LANDMARKS[name]
            if not landmark_is_visible(lms, idx):
                all_visible = False
                break
        result[step_num] = all_visible
    return result


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def build_features(lms, w, h):
    """Build features dict. Includes ALL features even if some landmarks
    aren't visible - the validator will skip steps based on visibility info
    rather than missing features."""

    def safe_xy(name):
        return extract_xy(lms, w, h, POSE_LANDMARKS[name])

    ls = safe_xy("left_shoulder")
    rs = safe_xy("right_shoulder")
    lh = safe_xy("left_hip")
    rh = safe_xy("right_hip")
    lel = safe_xy("left_elbow")
    rel = safe_xy("right_elbow")
    lw_pt = safe_xy("left_wrist")
    rw_pt = safe_xy("right_wrist")
    lk = safe_xy("left_knee")
    rk = safe_xy("right_knee")
    la = safe_xy("left_ankle")
    ra = safe_xy("right_ankle")
    nose = safe_xy("nose")

    shoulder_tilt = abs(ls[1] - rs[1]) / h
    hip_tilt = abs(lh[1] - rh[1]) / h

    body_center_x = ((ls[0] + rs[0]) / 2 + (lh[0] + rh[0]) / 2) / 2
    ankle_center_x = (la[0] + ra[0]) / 2
    body_lean = abs(body_center_x - ankle_center_x) / w

    left_arm_distance = abs(lw_pt[0] - lh[0]) / w
    right_arm_distance = abs(rw_pt[0] - rh[0]) / w

    left_knee_bend = calculate_angle(lh, lk, la)
    right_knee_bend = calculate_angle(rh, rk, ra)

    ankle_distance = abs(la[0] - ra[0])
    hip_distance = abs(lh[0] - rh[0])
    stance_ratio = ankle_distance / hip_distance if hip_distance > 1 else 1.0

    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    spine_tilt = angle_from_vertical(mid_shoulders, mid_hips)

    head_offset = abs(nose[0] - mid_shoulders[0]) / w

    shoulder_y = (ls[1] + rs[1]) / 2
    ankle_y = (la[1] + ra[1]) / 2
    body_height = ankle_y - shoulder_y
    if body_height > 1:
        left_arm_drop = (lw_pt[1] - shoulder_y) / body_height
        right_arm_drop = (rw_pt[1] - shoulder_y) / body_height
    else:
        left_arm_drop = 0.5
        right_arm_drop = 0.5

    left_elbow_angle = calculate_angle(ls, lel, lw_pt)
    right_elbow_angle = calculate_angle(rs, rel, rw_pt)
    arm_closeness = abs(lw_pt[0] - rw_pt[0]) / w

    return {
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "body_lean": body_lean,
        "left_arm_distance": left_arm_distance,
        "right_arm_distance": right_arm_distance,
        "left_knee_bend": left_knee_bend,
        "right_knee_bend": right_knee_bend,
        "stance_ratio": stance_ratio,
        "spine_tilt": spine_tilt,
        "head_offset": head_offset,
        "left_arm_drop": left_arm_drop,
        "right_arm_drop": right_arm_drop,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "arm_closeness": arm_closeness,
    }


# -----------------------------------------------------------------------------
# Image generation (unchanged - just operates on best frame's landmarks)
# -----------------------------------------------------------------------------
def _crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def _crop_with_padding(img, points, padding_x_frac=0.15, padding_y_frac=0.15):
    h, w = img.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad_x = w * padding_x_frac
    pad_y = h * padding_y_frac
    x1 = min(xs) - pad_x; x2 = max(xs) + pad_x
    y1 = min(ys) - pad_y; y2 = max(ys) + pad_y
    return _crop_safe(img, x1, y1, x2, y2)


def generate_step_images(frame, lms, step_results, save_dir):
    h, w = frame.shape[:2]
    paths = {}

    pts = {name: extract_xy(lms, w, h, idx)
           for name, idx in POSE_LANDMARKS.items()}

    # Use 'evaluable' AND 'passed' to determine line color:
    #   passed -> green
    #   failed -> red
    #   not evaluable -> gray
    step_state = {s["step"]: ("passed" if s.get("passed") else
                               ("not_eval" if not s.get("evaluable") else "failed"))
                  for s in step_results}

    annotated = frame.copy()
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    GRAY = (128, 128, 128)

    def color_for(step_num):
        state = step_state.get(step_num, "failed")
        if state == "passed": return GREEN
        if state == "not_eval": return GRAY
        return RED

    def line(p1, p2, color, thick=4):
        cv2.line(annotated,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 color, thick, cv2.LINE_AA)

    def dot(p, color, r=6):
        cv2.circle(annotated, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)

    c5 = color_for(5)
    line(pts["left_shoulder"], pts["left_elbow"], c5)
    line(pts["left_elbow"], pts["left_wrist"], c5)
    line(pts["right_shoulder"], pts["right_elbow"], c5)
    line(pts["right_elbow"], pts["right_wrist"], c5)
    line(pts["left_shoulder"], pts["right_shoulder"], c5)

    c4 = color_for(4)
    mid_sh = midpoint(pts["left_shoulder"], pts["right_shoulder"])
    mid_hp = midpoint(pts["left_hip"], pts["right_hip"])
    line(mid_sh, mid_hp, c4, thick=5)

    c3 = color_for(3)
    line(pts["left_hip"], pts["left_knee"], c3)
    line(pts["left_knee"], pts["left_ankle"], c3)
    line(pts["right_hip"], pts["right_knee"], c3)
    line(pts["right_knee"], pts["right_ankle"], c3)
    line(pts["left_hip"], pts["right_hip"], c3)

    c1 = color_for(1)
    line(pts["left_ankle"], pts["right_ankle"], c1, thick=3)

    c6 = color_for(6)
    dot(pts["nose"], c6, r=10)

    for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        dot(pts[name], (255, 255, 255), r=4)

    annotated_path = os.path.join(save_dir, "annotated_full.jpg")
    cv2.imwrite(annotated_path, annotated)
    paths["annotated"] = annotated_path

    feet_pts = [pts["left_ankle"], pts["right_ankle"],
                pts["left_heel"], pts["right_heel"],
                pts["left_foot_index"], pts["right_foot_index"]]
    crop = _crop_with_padding(annotated, feet_pts, 0.15, 0.10)
    p1 = os.path.join(save_dir, "step1_stance.jpg")
    cv2.imwrite(p1, crop); paths["step_1"] = p1

    p2 = os.path.join(save_dir, "step2_body_balance.jpg")
    cv2.imwrite(p2, annotated); paths["step_2"] = p2

    leg_pts = [pts["left_hip"], pts["right_hip"],
               pts["left_knee"], pts["right_knee"],
               pts["left_ankle"], pts["right_ankle"]]
    crop = _crop_with_padding(annotated, leg_pts, 0.12, 0.05)
    p3 = os.path.join(save_dir, "step3_legs_knees.jpg")
    cv2.imwrite(p3, crop); paths["step_3"] = p3

    spine_pts = [pts["left_shoulder"], pts["right_shoulder"],
                 pts["left_hip"], pts["right_hip"]]
    crop = _crop_with_padding(annotated, spine_pts, 0.18, 0.05)
    p4 = os.path.join(save_dir, "step4_spine.jpg")
    cv2.imwrite(p4, crop); paths["step_4"] = p4

    arm_pts = [pts["left_shoulder"], pts["right_shoulder"],
               pts["left_elbow"], pts["right_elbow"],
               pts["left_wrist"], pts["right_wrist"],
               pts["left_hip"], pts["right_hip"]]
    crop = _crop_with_padding(annotated, arm_pts, 0.10, 0.10)
    p5 = os.path.join(save_dir, "step5_shoulders_arms.jpg")
    cv2.imwrite(p5, crop); paths["step_5"] = p5

    head_pts = [pts["nose"], pts["left_shoulder"], pts["right_shoulder"]]
    crop = _crop_with_padding(annotated, head_pts, 0.15, 0.20)
    p6 = os.path.join(save_dir, "step6_head_neck.jpg")
    cv2.imwrite(p6, crop); paths["step_6"] = p6

    return paths


# -----------------------------------------------------------------------------
# Aggregation across frames
# -----------------------------------------------------------------------------
from src.scorer import STEP_NAMES, STEP_CUES, STEP_WEIGHTS, BODY_PART_DESCRIPTIONS


def aggregate_step_reports(all_reports, total_frames_seen):
    """Aggregate per-frame results, properly handling 'cannot evaluate' steps.

    For each step:
      - Compute average score and fail rate ONLY over frames where step was evaluable
      - If step was evaluable in <50% of analyzed frames, mark step as cannot_evaluate
      - Track evaluable_frame_percent for transparency
    """
    if not all_reports:
        return None

    aggregated_steps = []
    total_analyzed = len(all_reports)

    for step_num in range(1, 7):
        # Pull this step's results across all frames
        evaluable_results = []
        not_evaluable_count = 0

        for report in all_reports:
            step = report["steps"][step_num - 1]
            if step["evaluable"]:
                evaluable_results.append(step)
            else:
                not_evaluable_count += 1

        evaluable_count = len(evaluable_results)
        evaluable_pct = round((evaluable_count / total_analyzed) * 100, 1) if total_analyzed > 0 else 0

        # Decide if the step is overall evaluable
        # Need at least 50% of frames to have had this step evaluable
        if evaluable_count == 0 or evaluable_pct < 50:
            body_part = BODY_PART_DESCRIPTIONS[step_num]
            aggregated_steps.append({
                "step": step_num,
                "name": STEP_NAMES[step_num],
                "cue": STEP_CUES[step_num],
                "weight": STEP_WEIGHTS[step_num],
                "evaluable": False,
                "passed_overall": False,
                "average_score": None,
                "fail_rate_percent": None,
                "evaluable_frame_percent": evaluable_pct,
                "issue": f"Cannot evaluate - {body_part} not visible in {round(100 - evaluable_pct, 1)}% of frames",
            })
            continue

        # Step is evaluable - compute averages from the frames where it was evaluable
        scores = [s["score"] for s in evaluable_results]
        fails = sum(1 for s in evaluable_results if not s["passed"])
        issues_seen = [s["issue"] for s in evaluable_results if s["issue"]]

        avg_score = round(sum(scores) / len(scores), 1)
        fail_rate = round(fails / len(evaluable_results) * 100, 1)
        most_common = max(set(issues_seen), key=issues_seen.count) if issues_seen else None

        aggregated_steps.append({
            "step": step_num,
            "name": STEP_NAMES[step_num],
            "cue": STEP_CUES[step_num],
            "weight": STEP_WEIGHTS[step_num],
            "evaluable": True,
            "passed_overall": fail_rate < 25,
            "average_score": avg_score,
            "fail_rate_percent": fail_rate,
            "evaluable_frame_percent": evaluable_pct,
            "issue": most_common,
        })

    # Final score: average per-frame final scores from analyzable frames
    valid_finals = [r["final_score"] for r in all_reports if r.get("final_score") is not None]
    final_score = (int(round(sum(valid_finals) / len(valid_finals)))
                   if valid_finals else 0)
    final_score = max(0, min(100, final_score))

    # Significant issues - only from steps that were evaluable
    significant_issues = [
        s["issue"] for s in aggregated_steps
        if s["evaluable"] and s["issue"] and s["fail_rate_percent"] >= 25
    ]

    # Visibility issues - body parts not visible enough
    visibility_issues = [
        BODY_PART_DESCRIPTIONS[s["step"]]
        for s in aggregated_steps
        if not s["evaluable"]
    ]

    # Coverage stats
    frames_analyzable = total_analyzed
    frames_skipped = total_frames_seen - total_analyzed
    coverage_pct = round((frames_analyzable / total_frames_seen) * 100, 1) if total_frames_seen > 0 else 0

    return {
        "final_score": final_score,
        "steps": aggregated_steps,
        "issues": significant_issues,
        "visibility_issues": visibility_issues,
        "frames_analyzable": frames_analyzable,
        "frames_skipped": frames_skipped,
        "coverage_pct": coverage_pct,
    }


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def analyze_video(video_path, save_frames_dir=None):
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)

    all_reports = []          # frames with >= MIN_EVALUABLE_STEPS_PER_FRAME evaluable steps
    best_score = -1
    best_frame = None
    best_landmarks = None
    best_step_results = None

    total_frames_seen = 0     # frames where MediaPipe detected a pose

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        results = detector.detect(frame)

        if results.pose_landmarks:
            total_frames_seen += 1
            lms = results.pose_landmarks.landmark
            step_visibility = get_step_visibility(lms)
            features = build_features(lms, w, h)
            report = validate_tadasana(features, step_visibility)

            # Only include this frame in scoring if enough steps were evaluable
            if report.get("frame_evaluable") and report.get("final_score") is not None:
                all_reports.append(report)

                if report["final_score"] > best_score:
                    best_score = report["final_score"]
                    best_frame = frame.copy()
                    best_landmarks = lms
                    best_step_results = report["steps"]

    cap.release()

    # No usable frames at all
    if total_frames_seen == 0:
        return {
            "final_score": 0,
            "issues": ["No body pose detected in the video"],
            "steps": [],
            "best_frame_path": None,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": (
                "No body pose was detected in this video. "
                "Please re-record with good lighting and your full body in frame."
            ),
            "visibility_issues": [],
            "frames_analyzable": 0,
            "frames_skipped": 0,
            "coverage_pct": 0,
        }

    # Body detected but most frames were not fully visible
    if not all_reports:
        # Try to save SOMETHING for the user to see, even though it's not analyzable
        return {
            "final_score": 0,
            "issues": [],
            "steps": [],
            "best_frame_path": None,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": (
                "Could not analyze your pose - your full body was not clearly visible "
                "in any frame. Please re-record with full body in frame, good lighting, "
                "and the pose held steadily."
            ),
            "visibility_issues": ["Most body parts were not visible enough to evaluate"],
            "frames_analyzable": 0,
            "frames_skipped": total_frames_seen,
            "coverage_pct": 0,
        }

    aggregated = aggregate_step_reports(all_reports, total_frames_seen)

    # Generate images from best frame
    step_image_paths = {}
    annotated_path = None
    best_frame_path = None
    if best_frame is not None and save_frames_dir:
        best_frame_path = os.path.join(save_frames_dir, "best_pose_frame.jpg")
        cv2.imwrite(best_frame_path, best_frame)
        step_image_paths = generate_step_images(
            best_frame, best_landmarks, best_step_results, save_frames_dir
        )
        annotated_path = step_image_paths.get("annotated")

    # Quality warnings
    low_quality = False
    low_quality_msg = None

    if best_score < MIN_QUALITY_SCORE:
        low_quality = True
        low_quality_msg = (
            f"The best frame in this video scored only {best_score}/100. "
            "The pose may not have been clearly Tadasana. For more accurate "
            "results, please re-record with: full body in frame, good lighting, "
            "and the pose held steadily."
        )

    # Low coverage warning
    coverage_pct = aggregated["coverage_pct"]
    if coverage_pct < 50:
        low_quality = True
        low_quality_msg = (
            f"Only {coverage_pct}% of detected frames had your full body clearly "
            "visible. Re-record with your full body in the frame for a complete analysis."
        )

    return {
        "final_score": aggregated["final_score"],
        "issues": aggregated["issues"],
        "steps": aggregated["steps"],
        "best_frame_path": best_frame_path,
        "annotated_path": annotated_path,
        "step_image_paths": step_image_paths,
        "low_quality_warning": low_quality,
        "low_quality_message": low_quality_msg,
        "visibility_issues": aggregated["visibility_issues"],
        "frames_analyzable": aggregated["frames_analyzable"],
        "frames_skipped": aggregated["frames_skipped"],
        "coverage_pct": coverage_pct,
    }
