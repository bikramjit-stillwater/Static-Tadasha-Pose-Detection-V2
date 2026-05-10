"""
Tadasana validator - visibility-aware version.

Each step can return one of three states:
  - PASSED       (score >= passing threshold)
  - FAILED       (evaluable but score below threshold)
  - NOT EVALUABLE (critical landmarks not visible in this frame)

Final score is computed from EVALUABLE steps only, with weights renormalised.
"""

import math


# -----------------------------------------------------------------------------
# Geometry helper
# -----------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def score_value(deviation, ideal_max, fail_min, curve="quadratic"):
    """Convert deviation into 0-100 score."""
    if deviation <= ideal_max:
        return 100.0
    if deviation >= fail_min:
        return 0.0
    span = fail_min - ideal_max
    over = deviation - ideal_max
    progress = over / span
    if curve == "quadratic":
        return round(100.0 * (1.0 - progress) ** 2, 1)
    return round(100.0 * (1.0 - progress), 1)


# -----------------------------------------------------------------------------
# Step metadata
# -----------------------------------------------------------------------------
STEP_NAMES = {
    1: "Stance",
    2: "Body Balance",
    3: "Legs & Knees",
    4: "Spine",
    5: "Shoulders & Arms",
    6: "Head & Neck",
}

STEP_CUES = {
    1: "Stand with feet together or at hip-distance - not wider",
    2: "Press the four corners of each foot into the floor evenly",
    3: "Lift kneecaps gently - straight but never locked",
    4: "Lengthen the spine - tailbone tucks down, crown lifts up",
    5: "Stretch arms straight up overhead, palms together, elbows straight",
    6: "Keep the head balanced between the arms, gaze soft and forward",
}

# Per-step weights (sum to 1.0)
STEP_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.25, 6: 0.10}

# Friendly names for body parts when explaining what was not visible
BODY_PART_DESCRIPTIONS = {
    1: "feet",
    2: "full body (shoulders, hips, ankles)",
    3: "legs (hips to ankles)",
    4: "torso (shoulders and hips)",
    5: "arms (shoulders, elbows, wrists)",
    6: "head and shoulders",
}


# -----------------------------------------------------------------------------
# Per-step validators (only called when step is evaluable)
# -----------------------------------------------------------------------------
def check_stance(features):
    ratio = features["stance_ratio"]
    if ratio <= 1.1:
        return {"step": 1, "passed": True, "score": 100.0, "issue": None}
    score = score_value(ratio - 1.1, 0.0, 0.6, "quadratic")
    return {
        "step": 1,
        "passed": ratio <= 1.25,
        "score": score,
        "issue": "Feet are too wide apart - bring them to hip-width or together",
    }


def check_body_balance(features):
    body_lean = features["body_lean"]
    score = score_value(body_lean, 0.02, 0.08, "quadratic")
    passed = body_lean <= 0.035
    return {
        "step": 2,
        "passed": passed,
        "score": score,
        "issue": None if passed else "Body is leaning - distribute weight evenly across both feet",
    }


def check_legs_knees(features):
    left = features["left_knee_bend"]
    right = features["right_knee_bend"]
    bent = left < 168 or right < 168
    locked = left > 178 or right > 178

    if not bent and not locked:
        return {"step": 3, "passed": True, "score": 100.0, "issue": None}
    if locked and not bent:
        worst = max(left, right)
        return {
            "step": 3, "passed": False,
            "score": score_value(worst - 178, 0.0, 6.0, "quadratic"),
            "issue": "Knees are locked - keep them soft and active, not rigid",
        }
    if bent and not locked:
        worst = min(left, right)
        return {
            "step": 3, "passed": False,
            "score": score_value(168 - worst, 0.0, 18.0, "quadratic"),
            "issue": "Knees are bent - gently straighten without locking",
        }
    return {
        "step": 3, "passed": False, "score": 30.0,
        "issue": "One knee bent and the other locked - aim for soft and even",
    }


def check_spine(features):
    spine_tilt = features["spine_tilt"]
    score = score_value(spine_tilt, 2.0, 10.0, "quadratic")
    passed = spine_tilt <= 4.0
    return {
        "step": 4, "passed": passed, "score": score,
        "issue": None if passed else "Spine is not vertical - tailbone down, crown of head up",
    }


def check_shoulders_arms(features):
    shoulder_tilt = features["shoulder_tilt"]
    h_left = features["left_arm_distance"]
    h_right = features["right_arm_distance"]
    v_left = features["left_arm_drop"]
    v_right = features["right_arm_drop"]

    # Sub-score 1: shoulders level
    s_score = score_value(shoulder_tilt, 0.015, 0.07, "quadratic")

    # Sub-score 2: arms close to body horizontally
    worst_h = max(h_left, h_right)
    h_score = score_value(worst_h, 0.05, 0.18, "quadratic")

    # Sub-score 3: arms raised overhead
    worst_v = min(v_left, v_right)
    if worst_v <= -0.25:
        v_score = 100.0
    elif worst_v <= 0.0:
        v_score = round(100.0 * ((-worst_v) / 0.25) ** 2, 1)
    else:
        v_score = 0.0

    # Sub-score 4: elbows straight
    e_left = features.get("left_elbow_angle", 180)
    e_right = features.get("right_elbow_angle", 180)
    worst_elbow = min(e_left, e_right)
    if worst_elbow >= 165:
        elbow_score = 100.0
    else:
        elbow_score = score_value(165 - worst_elbow, 0.0, 35.0, "quadratic")

    # Sub-score 5: arms close together overhead
    arm_closeness = features.get("arm_closeness", 0.05)
    closeness_score = score_value(arm_closeness, 0.05, 0.40, "quadratic")

    # Sub-score 6: symmetry
    asymmetry = abs(v_left - v_right)
    symmetry_score = score_value(asymmetry, 0.04, 0.20, "quadratic")

    score = round(
        s_score * 0.10 +
        v_score * 0.40 +
        elbow_score * 0.20 +
        closeness_score * 0.15 +
        symmetry_score * 0.15,
        1,
    )

    issues = []
    if shoulder_tilt > 0.025:
        issues.append("Shoulders are uneven")
    if worst_v > -0.05:
        issues.append("Arms are not raised - stretch them straight up overhead")
    elif worst_v > -0.20:
        issues.append("Reach arms higher - extend fully overhead")
    if worst_elbow < 160:
        issues.append("Elbows are bent - straighten the arms")
    if arm_closeness > 0.30:
        issues.append("Bring the arms closer together overhead")
    if asymmetry > 0.10:
        issues.append("One arm is higher than the other - keep them even")

    passed = len(issues) == 0
    issue = " - ".join(issues) if issues else None

    return {"step": 5, "passed": passed, "score": score, "issue": issue}


def check_head_neutral(features):
    head_offset = features["head_offset"]
    score = score_value(head_offset, 0.025, 0.10, "quadratic")
    passed = head_offset <= 0.05
    return {
        "step": 6, "passed": passed, "score": score,
        "issue": None if passed else "Head is tilting forward or sideways - keep it balanced",
    }


STEP_FUNCTIONS = {
    1: check_stance,
    2: check_body_balance,
    3: check_legs_knees,
    4: check_spine,
    5: check_shoulders_arms,
    6: check_head_neutral,
}


# -----------------------------------------------------------------------------
# Main validator with visibility handling
# -----------------------------------------------------------------------------
def validate_tadasana(features, step_visibility=None):
    """
    Run the 6-step validation, handling missing landmarks gracefully.

    `step_visibility` is a dict {1: True/False, ...} indicating which
    steps have all their critical landmarks visible.

    Returns a dict containing:
      - final_score: int 0-100 (from evaluable steps only) or None if no steps evaluable
      - frame_evaluable: True if at least MIN_EVALUABLE_STEPS were evaluable
      - steps: list of step result dicts (each marked evaluable True/False)
      - issues: list of issue strings from FAILED EVALUABLE steps only
      - visibility_issues: list of body parts not visible
    """
    if step_visibility is None:
        step_visibility = {i: True for i in range(1, 7)}

    step_results = []
    visibility_issues = []

    for step_num in range(1, 7):
        is_evaluable = step_visibility.get(step_num, True)

        if not is_evaluable:
            body_part = BODY_PART_DESCRIPTIONS[step_num]
            step_results.append({
                "step": step_num,
                "name": STEP_NAMES[step_num],
                "evaluable": False,
                "passed": None,
                "score": None,
                "issue": f"Cannot evaluate - {body_part} not visible in frame",
                "cue": STEP_CUES[step_num],
                "weight": STEP_WEIGHTS[step_num],
            })
            visibility_issues.append(body_part)
        else:
            try:
                result = STEP_FUNCTIONS[step_num](features)
                result["name"] = STEP_NAMES[step_num]
                result["evaluable"] = True
                result["cue"] = STEP_CUES[step_num]
                result["weight"] = STEP_WEIGHTS[step_num]
                step_results.append(result)
            except (KeyError, TypeError, ValueError):
                # Feature was missing or invalid - treat as not evaluable
                body_part = BODY_PART_DESCRIPTIONS[step_num]
                step_results.append({
                    "step": step_num,
                    "name": STEP_NAMES[step_num],
                    "evaluable": False,
                    "passed": None,
                    "score": None,
                    "issue": f"Cannot evaluate - {body_part} not visible in frame",
                    "cue": STEP_CUES[step_num],
                    "weight": STEP_WEIGHTS[step_num],
                })
                visibility_issues.append(body_part)

    # Calculate final score from evaluable steps only
    evaluable = [s for s in step_results if s["evaluable"]]
    MIN_EVALUABLE_STEPS = 4

    frame_evaluable = len(evaluable) >= MIN_EVALUABLE_STEPS

    if not evaluable:
        return {
            "final_score": None,
            "frame_evaluable": False,
            "steps": step_results,
            "issues": [],
            "visibility_issues": visibility_issues,
        }

    # Renormalize weights to sum to 1 across evaluable steps
    total_weight = sum(s["weight"] for s in evaluable)
    base_score = 0.0
    for s in evaluable:
        renormalized = s["weight"] / total_weight
        base_score += s["score"] * renormalized

    # Apply compound penalties to evaluable steps only
    worst = min(s["score"] for s in evaluable)
    very_bad = sum(1 for s in evaluable if s["score"] < 20)
    critical = sum(1 for s in evaluable if s["score"] < 40)

    final_score = base_score
    if very_bad >= 2:
        final_score *= 0.55
    elif very_bad >= 1:
        final_score *= 0.75
    elif critical >= 2:
        final_score *= 0.85

    if worst < 50:
        final_score = min(final_score, 78.0)
    if worst < 30:
        final_score = min(final_score, 60.0)
    if worst < 15:
        final_score = min(final_score, 45.0)

    final_score = max(0, min(100, int(round(final_score))))

    issues = [s["issue"] for s in step_results
              if s["evaluable"] and s["issue"]]

    return {
        "final_score": final_score,
        "frame_evaluable": frame_evaluable,
        "steps": step_results,
        "issues": issues,
        "visibility_issues": visibility_issues,
    }


def score_tadasana(features, step_visibility=None):
    """Backwards-compatible wrapper."""
    report = validate_tadasana(features, step_visibility)
    return report["final_score"], report["issues"]
