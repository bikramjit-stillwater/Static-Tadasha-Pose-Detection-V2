"""
Tadasana (Mountain Pose) validator - Arms Overhead Version.

Visibility rule:
  If a body part required for a step is NOT VISIBLE in the frame,
  that step automatically scores 0 with a clear "not visible" message.
  The 0 is included in the weighted average (penalizes missing body parts).

Target pose: standing tall with arms raised straight overhead, palms together,
feet together or hip-width, body stretched upward like a mountain.

6 ground-truth steps, each scored 0-100 with quadratic curves.
Compound penalties applied when multiple steps fail.
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
# "Not visible" result helpers
# -----------------------------------------------------------------------------
def _not_visible(step_num, name, body_part, cue):
    """Return a step result for a body part that is NOT visible.
    Score is 0 and the issue clearly says it's not visible."""
    return {
        "step": step_num,
        "name": name,
        "passed": False,
        "score": 0.0,
        "issue": f"Cannot evaluate - {body_part} not visible in the frame",
        "cue": cue,
        "not_visible": True,
    }


# =============================================================================
# Step 1 - Stance: feet together or hip-width
#   Required body parts: both ankles, both hips
# =============================================================================
def check_stance(features, visible=True):
    if not visible:
        return _not_visible(
            1, "Stance", "feet",
            "Stand with feet together or at hip-distance",
        )

    ratio = features["stance_ratio"]
    if ratio <= 1.1:
        return {
            "step": 1, "name": "Stance",
            "passed": True, "score": 100.0, "issue": None,
            "cue": "Stand with feet together or at hip-distance",
            "not_visible": False,
        }
    score = score_value(ratio - 1.1, 0.0, 0.6, "quadratic")
    return {
        "step": 1, "name": "Stance",
        "passed": ratio <= 1.25, "score": score,
        "issue": "Feet are too far apart - bring them to hip-width or together",
        "cue": "Stand with feet together or at hip-distance",
        "not_visible": False,
    }


# =============================================================================
# Step 2 - Body Balance
#   Required body parts: shoulders, hips, ankles
# =============================================================================
def check_body_balance(features, visible=True):
    if not visible:
        return _not_visible(
            2, "Body Balance", "full body (shoulders, hips, feet)",
            "Press the four corners of each foot into the floor evenly",
        )

    body_lean = features["body_lean"]
    score = score_value(body_lean, 0.025, 0.09, "quadratic")
    passed = body_lean <= 0.04
    return {
        "step": 2, "name": "Body Balance",
        "passed": passed, "score": score,
        "issue": None if passed else "Body is leaning - distribute weight evenly across both feet",
        "cue": "Press the four corners of each foot into the floor evenly",
        "not_visible": False,
    }


# =============================================================================
# Step 3 - Legs & Knees
#   Required body parts: hips, knees, ankles
# =============================================================================
def check_legs_knees(features, visible=True):
    if not visible:
        return _not_visible(
            3, "Legs & Knees", "legs (hips, knees, ankles)",
            "Lift kneecaps gently - straight but never locked",
        )

    left = features["left_knee_bend"]
    right = features["right_knee_bend"]
    bent = left < 168 or right < 168
    locked = left > 178 or right > 178

    if not bent and not locked:
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": True, "score": 100.0, "issue": None,
            "cue": "Lift kneecaps gently - straight but never locked",
            "not_visible": False,
        }
    if locked and not bent:
        worst = max(left, right)
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": False,
            "score": score_value(worst - 178, 0.0, 6.0, "quadratic"),
            "issue": "Knees are locked - keep them soft and active, not rigid",
            "cue": "Lift kneecaps gently - straight but never locked",
            "not_visible": False,
        }
    if bent and not locked:
        worst = min(left, right)
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": False,
            "score": score_value(168 - worst, 0.0, 18.0, "quadratic"),
            "issue": "Knees are bent - gently straighten without locking",
            "cue": "Lift kneecaps gently - straight but never locked",
            "not_visible": False,
        }
    return {
        "step": 3, "name": "Legs & Knees",
        "passed": False, "score": 30.0,
        "issue": "One knee bent and the other locked - aim for soft and even",
        "cue": "Lift kneecaps gently - straight but never locked",
        "not_visible": False,
    }


# =============================================================================
# Step 4 - Spine
#   Required body parts: shoulders, hips
# =============================================================================
def check_spine(features, visible=True):
    if not visible:
        return _not_visible(
            4, "Spine", "torso (shoulders and hips)",
            "Lengthen the spine - tailbone tucks down, crown lifts up",
        )

    spine_tilt = features["spine_tilt"]
    score = score_value(spine_tilt, 2.5, 11.0, "quadratic")
    passed = spine_tilt <= 5.0
    return {
        "step": 4, "name": "Spine",
        "passed": passed, "score": score,
        "issue": None if passed else "Spine is not vertical - tailbone down, crown of head up",
        "cue": "Lengthen the spine - tailbone tucks down, crown lifts up",
        "not_visible": False,
    }


# =============================================================================
# Step 5 - Shoulders & Arms (arms raised overhead)
#   Required body parts: shoulders, elbows, wrists
# =============================================================================
def check_shoulders_arms(features, visible=True):
    if not visible:
        return _not_visible(
            5, "Shoulders & Arms", "arms (shoulders, elbows, wrists)",
            "Stretch arms straight up overhead, palms together, elbows straight",
        )

    v_left = features["left_arm_drop"]
    v_right = features["right_arm_drop"]
    worst_v = max(v_left, v_right)

    if worst_v <= -0.25:
        raised_score = 100.0
    elif worst_v <= 0.0:
        raised_score = round(100.0 * ((-worst_v) / 0.25) ** 2, 1)
    else:
        raised_score = 0.0

    e_left = features["left_elbow_angle"]
    e_right = features["right_elbow_angle"]
    worst_elbow = min(e_left, e_right)
    if worst_elbow >= 165:
        elbow_score = 100.0
    else:
        elbow_score = score_value(165 - worst_elbow, 0.0, 35.0, "quadratic")

    arm_closeness = features["arm_closeness"]
    closeness_score = score_value(arm_closeness, 0.05, 0.40, "quadratic")

    asymmetry = abs(v_left - v_right)
    symmetry_score = score_value(asymmetry, 0.04, 0.20, "quadratic")

    score = round(
        raised_score * 0.45 +
        elbow_score * 0.25 +
        closeness_score * 0.15 +
        symmetry_score * 0.15,
        1,
    )

    issues = []
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

    return {
        "step": 5, "name": "Shoulders & Arms",
        "passed": passed, "score": score,
        "issue": issue,
        "cue": "Stretch arms straight up overhead, palms together, elbows straight",
        "not_visible": False,
    }


# =============================================================================
# Step 6 - Head & Neck
#   Required body parts: nose, shoulders
# =============================================================================
def check_head_neutral(features, visible=True):
    if not visible:
        return _not_visible(
            6, "Head & Neck", "head and shoulders",
            "Keep the head balanced between the arms, gaze soft and forward",
        )

    head_offset = features["head_offset"]
    score = score_value(head_offset, 0.03, 0.11, "quadratic")
    passed = head_offset <= 0.06
    return {
        "step": 6, "name": "Head & Neck",
        "passed": passed, "score": score,
        "issue": None if passed else "Head is tilting - keep it balanced, gaze forward",
        "cue": "Keep the head balanced between the arms, gaze soft and forward",
        "not_visible": False,
    }


# Step weights (sum to 1.0)
STEP_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.25, 6: 0.10}


def validate_tadasana(features, step_visibility=None):
    """Run 6-step validation with compound penalties.

    Args:
        features: dict of geometric features (from build_features)
        step_visibility: dict {step_num: True/False} - True if all body parts
                         needed for that step are visible. If None, assumes
                         all visible (backwards compatible).
    """
    if step_visibility is None:
        step_visibility = {i: True for i in range(1, 7)}

    step_results = [
        check_stance(features, step_visibility.get(1, True)),
        check_body_balance(features, step_visibility.get(2, True)),
        check_legs_knees(features, step_visibility.get(3, True)),
        check_spine(features, step_visibility.get(4, True)),
        check_shoulders_arms(features, step_visibility.get(5, True)),
        check_head_neutral(features, step_visibility.get(6, True)),
    ]

    base_score = 0.0
    for s in step_results:
        s["weight"] = STEP_WEIGHTS[s["step"]]
        base_score += s["score"] * s["weight"]

    worst = min(s["score"] for s in step_results)
    very_bad = sum(1 for s in step_results if s["score"] < 20)
    critical = sum(1 for s in step_results if s["score"] < 40)

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

    final_score = int(round(final_score))
    final_score = max(0, min(100, final_score))

    issues = [s["issue"] for s in step_results if s["issue"]]

    return {
        "final_score": final_score,
        "steps": step_results,
        "issues": issues,
    }


def score_tadasana(features, step_visibility=None):
    report = validate_tadasana(features, step_visibility)
    return report["final_score"], report["issues"]
