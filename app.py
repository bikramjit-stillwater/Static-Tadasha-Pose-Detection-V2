import os
import tempfile
from datetime import datetime
import streamlit as st

from src.pose_analyzer import analyze_video
from src.feedback import get_gemini_feedback


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Tadasana Pose Analysis",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Custom CSS - dashboard-style design
# =============================================================================
CUSTOM_CSS = """
<style>
section.main > div.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

* { box-sizing: border-box; }

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {height: 0; visibility: hidden;}
.stDeployButton {display: none;}

/* Page header */
.page-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e8edf3;
}
.page-header h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 !important;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.page-header .subtitle {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.35rem;
}

/* Section headings */
.stMarkdown h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.01em;
}
.stMarkdown h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.4rem !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 0.875rem !important;
    line-height: 1.5 !important;
    color: #334155;
}

/* Score hero card */
.score-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 14px;
    padding: 1.5rem;
    color: white;
    box-shadow: 0 4px 20px rgba(15,23,42,0.15);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 380px;
}
.score-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.score-value {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    color: white;
    letter-spacing: -0.04em;
}
.score-suffix {
    font-size: 1.5rem;
    color: #94a3b8;
    font-weight: 400;
}
.score-band {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.band-excellent { background: #10b981; color: white; }
.band-good      { background: #3b82f6; color: white; }
.band-mixed     { background: #f59e0b; color: white; }
.band-poor      { background: #ef4444; color: white; }

.issues-block { margin-top: 1.5rem; }
.issues-block h4 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 600;
    margin: 0 0 0.6rem 0;
}
.issue-item {
    background: rgba(255,255,255,0.08);
    border-left: 3px solid #fbbf24;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    border-radius: 4px;
    font-size: 0.82rem;
    color: #e2e8f0;
    line-height: 1.4;
}
.no-issues {
    background: rgba(16,185,129,0.15);
    border-left: 3px solid #10b981;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.82rem;
    color: #d1fae5;
}

/* Annotated image card */
.image-card {
    background: white;
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
    min-height: 380px;
    display: flex;
    flex-direction: column;
}
.image-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.image-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
}
.image-card-legend {
    font-size: 0.7rem;
    color: #64748b;
}
.legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.3rem;
    vertical-align: middle;
}

/* IMAGE SIZE CONSTRAINTS */
.annotated-wrap {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.annotated-wrap [data-testid="stImage"] {
    width: auto !important;
    max-width: 100% !important;
}
.annotated-wrap [data-testid="stImage"] img {
    max-height: 340px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 8px;
    margin: 0 auto;
    display: block;
}

/* Step cards */
.step-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: all 0.15s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.step-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transform: translateY(-1px);
}
.step-card.passed { border-left: 3px solid #10b981; }
.step-card.failed { border-left: 3px solid #ef4444; }

.step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.step-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.2;
}
.step-status-icon { font-size: 1.1rem; }

.step-score-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #f1f5f9;
}
.step-score-big {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
}
.step-score-big.pass { color: #10b981; }
.step-score-big.fail { color: #ef4444; }
.step-score-suffix {
    font-size: 0.75rem;
    color: #94a3b8;
}
.step-fail-rate {
    font-size: 0.7rem;
    color: #94a3b8;
}

/* STEP IMAGE SIZE CONSTRAINT */
.step-image-wrap {
    width: 100%;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f8fafc;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.step-image-wrap [data-testid="stImage"] {
    width: 100% !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.step-image-wrap [data-testid="stImage"] img {
    max-height: 180px !important;
    max-width: 100% !important;
    width: auto !important;
    object-fit: contain !important;
}

.step-issue {
    background: #fef2f2;
    border-left: 2px solid #ef4444;
    padding: 0.45rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #991b1b;
    line-height: 1.4;
    margin-bottom: 0.35rem;
}
.step-passed-msg {
    background: #f0fdf4;
    border-left: 2px solid #10b981;
    padding: 0.45rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #166534;
    margin-bottom: 0.35rem;
}
.step-cue {
    background: #f8fafc;
    border-left: 2px solid #3b82f6;
    padding: 0.45rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    color: #475569;
    line-height: 1.4;
    font-style: italic;
}
.step-cue strong { font-style: normal; color: #1e3a5f; }

/* Upload area */
[data-testid="stFileUploader"] section {
    padding: 1rem !important;
    border: 1.5px dashed #cbd5e1;
    border-radius: 8px;
    background-color: #f8fafc;
    transition: border-color 0.15s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #3b82f6;
    background-color: #f1f5f9;
}

/* Buttons */
.stButton > button {
    background-color: #1e3a5f;
    color: white;
    font-weight: 600;
    padding: 0.55rem 1.5rem;
    border-radius: 8px;
    border: none;
    font-size: 0.875rem;
    transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.stButton > button:hover {
    background-color: #0f172a;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,23,42,0.15);
}
.stButton > button[kind="primary"] {
    background-color: #2563eb;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8;
}

/* Video player constrained */
[data-testid="stVideo"] {
    border-radius: 8px;
    overflow: hidden;
    max-width: 480px;
    margin-bottom: 0.5rem;
}
[data-testid="stVideo"] video {
    max-height: 360px;
    border-radius: 8px;
}

/* Alerts */
[data-testid="stAlert"] {
    padding: 0.55rem 0.85rem !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.4 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin-bottom: 0.4rem;
}

/* Text area */
.stTextArea textarea {
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    background-color: #fafbfc !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 0.85rem !important;
    color: #1e293b !important;
}

/* Dividers */
hr {
    margin: 1.25rem 0 !important;
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
}

/* Mobile */
@media (max-width: 768px) {
    .page-header h1 { font-size: 1.4rem !important; }
    .score-value { font-size: 3rem; }
    .score-hero, .image-card { min-height: auto; }
    .step-image-wrap { height: 150px; }
    .annotated-wrap [data-testid="stImage"] img { max-height: 280px !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Helpers
# =============================================================================
def get_output_dir(subdir: str) -> str:
    base = os.environ.get("OUTPUT_DIR")
    if base:
        path = os.path.join(base, subdir)
    else:
        local = os.path.join("output", subdir)
        try:
            os.makedirs(local, exist_ok=True)
            test = os.path.join(local, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            path = local
        except OSError:
            path = os.path.join(tempfile.gettempdir(), "tadasana", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def score_band(score: int):
    if score >= 85: return ("Excellent", "band-excellent")
    if score >= 70: return ("Good", "band-good")
    if score >= 50: return ("Mixed", "band-mixed")
    return ("Needs work", "band-poor")


# =============================================================================
# Page header
# =============================================================================
st.markdown("""
<div class='page-header'>
  <h1>🧘 Tadasana Pose Analysis</h1>
  <div class='subtitle'>Upload your Tadasana video and get a step-by-step alignment report with personalized feedback.</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Output dirs + session state
# =============================================================================
recordings_dir = get_output_dir("recordings")
frames_dir = get_output_dir("extracted_frames")

for key, default in [
    ("video_path", None),
    ("analysis_result", None),
    ("gemini_feedback", None),
    ("capture_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# Upload section - VIDEO ONLY
# =============================================================================
if not st.session_state.analysis_result:
    st.markdown("## Upload Your Pose Video")

    upload_col, _ = st.columns([2, 1])
    with upload_col:
        uploaded_video = st.file_uploader(
            "Select a Tadasana video file (MP4, MOV, AVI, or MKV)",
            type=["mp4", "mov", "avi", "mkv"],
        )
        st.caption(
            "Tip: record yourself in good lighting with your full body in frame, "
            "and hold the pose steady for a few seconds."
        )
else:
    uploaded_video = None
    if st.button("← Analyze a different video"):
        for k in ["video_path", "analysis_result", "gemini_feedback", "capture_done"]:
            st.session_state[k] = None
        st.rerun()

# Save uploaded video
if uploaded_video is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(recordings_dir, f"uploaded_{timestamp}.mp4")
    with open(save_path, "wb") as f:
        f.write(uploaded_video.read())
    st.session_state.video_path = save_path
    st.session_state.capture_done = True
    st.success("✓ Video uploaded successfully.")

# Preview + analyze
if st.session_state.video_path and not st.session_state.analysis_result:
    st.markdown("### Preview")
    pv1, _ = st.columns([2, 1])
    with pv1:
        st.video(st.session_state.video_path)

    if st.button("🔍 Analyze Pose", type="primary"):
        with st.spinner("Analyzing pose and generating feedback..."):
            try:
                result = analyze_video(st.session_state.video_path, frames_dir)
                feedback_text = get_gemini_feedback(
                    result["final_score"], result["issues"],
                    steps=result.get("steps"),
                )
                st.session_state.analysis_result = result
                st.session_state.gemini_feedback = feedback_text
                st.rerun()
            except Exception as e:
                st.session_state.analysis_result = {
                    "final_score": 0, "issues": [f"Analysis failed: {str(e)}"],
                    "steps": [], "best_frame_path": None,
                    "annotated_path": None, "step_image_paths": {},
                    "low_quality_warning": True,
                    "low_quality_message": f"Analysis failed: {str(e)}",
                }
                st.session_state.gemini_feedback = (
                    f"Could not generate full pose feedback.\n\nReason: {str(e)}"
                )
                st.rerun()

# =============================================================================
# Display results
# =============================================================================
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    score = result["final_score"]
    band_label, band_class = score_band(score)

    if result.get("low_quality_warning"):
        st.warning(result.get("low_quality_message")
                   or "Low confidence in this analysis - please re-record.")

    # ===== Hero row =====
    hero_left, hero_right = st.columns([1, 1.5])

    with hero_left:
        issues_html = ""
        if result["issues"]:
            for issue in result["issues"][:3]:
                issues_html += f"<div class='issue-item'>{issue}</div>"
        else:
            issues_html = "<div class='no-issues'>✓ No major issues detected</div>"

        st.markdown(f"""
        <div class='score-hero'>
          <div>
            <div class='score-label'>Final Score</div>
            <div>
              <span class='score-value'>{score}</span>
              <span class='score-suffix'>/100</span>
            </div>
            <span class='score-band {band_class}'>{band_label}</span>
          </div>
          <div class='issues-block'>
            <h4>Top Issues</h4>
            {issues_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        annotated_path = result.get("annotated_path")
        st.markdown("""
        <div class='image-card'>
          <div class='image-card-header'>
            <div class='image-card-title'>Annotated Best Frame</div>
            <div class='image-card-legend'>
              <span class='legend-dot' style='background:#22c55e'></span>passed
              &nbsp;
              <span class='legend-dot' style='background:#ef4444'></span>needs work
            </div>
          </div>
          <div class='annotated-wrap'>
        """, unsafe_allow_html=True)

        if annotated_path and os.path.exists(annotated_path):
            st.image(annotated_path)
        elif result.get("best_frame_path") and os.path.exists(result["best_frame_path"]):
            st.image(result["best_frame_path"])

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ===== Step grid (3 columns) =====
    steps = result.get("steps") or []
    step_imgs = result.get("step_image_paths") or {}

    if steps:
        st.markdown("## Step-by-Step Breakdown")
        st.caption("Each card zooms into the body part being checked.")

        for i in range(0, len(steps), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j >= len(steps):
                    continue
                s = steps[i + j]
                step_num = s["step"]
                with col:
                    passed = s["passed_overall"]
                    pf_class = "passed" if passed else "failed"
                    score_class = "pass" if passed else "fail"
                    icon = "✅" if passed else "⚠️"

                    st.markdown(f"""
                    <div class='step-card {pf_class}'>
                      <div class='step-header'>
                        <div class='step-name'>Step {step_num}: {s['name']}</div>
                        <div class='step-status-icon'>{icon}</div>
                      </div>
                      <div class='step-score-row'>
                        <div>
                          <span class='step-score-big {score_class}'>{s['average_score']}</span>
                          <span class='step-score-suffix'>/100</span>
                        </div>
                        <div class='step-fail-rate'>
                          Failed in {s['fail_rate_percent']}% of frames
                        </div>
                      </div>
                      <div class='step-image-wrap'>
                    """, unsafe_allow_html=True)

                    img_key = f"step_{step_num}"
                    img_path = step_imgs.get(img_key)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path)

                    st.markdown("</div>", unsafe_allow_html=True)

                    if s["issue"]:
                        st.markdown(
                            f"<div class='step-issue'><strong>Issue:</strong> {s['issue']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='step-passed-msg'><strong>✓</strong> Looks good for this step.</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<div class='step-cue'><strong>Cue:</strong> {s['cue']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    # ===== Final feedback =====
    st.markdown("## Coach's Feedback")
    st.caption("Generated based on your step-by-step report.")
    st.text_area(
        "Feedback",
        value=st.session_state.gemini_feedback or "No feedback generated.",
        height=320,
        label_visibility="collapsed",
    )

elif not st.session_state.video_path:
    st.info("👆 Upload a Tadasana video above, then click **Analyze Pose**.")
