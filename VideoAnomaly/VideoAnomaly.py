"""
Video Anomaly Detection — (YouTube Crash Video) + BLIP
=====================================================
Downloads a crash video from YouTube, extracts frames at a set rate,
runs each frame through the BLIP captioning model, and flags frames
whose descriptions contain anomaly-related keywords (crash, fire, etc.).

Outputs: annotated screenshots, collision-only copies, a timeline chart,
a frame grid overview, and a full JSON report.
"""

import os
import sys
import json
import math
import shutil
import stat
import time
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path

# CONFIGURATION!!!
OUTPUT_DIR      = "crash_output_real"
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "frames")
COLLISION_DIR   = os.path.join(OUTPUT_DIR, "collisions")
VIDEO_CACHE     = os.path.join(OUTPUT_DIR, "input_video.mp4")
REPORT_PATH     = os.path.join(OUTPUT_DIR, "report.json")
CHART_PATH      = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
GRID_PATH       = os.path.join(OUTPUT_DIR, "all_frames_grid.jpg")

# YouTube URL of the crash video to analyze
YOUTUBE_URL = "https://www.youtube.com/watch?v=OAFWcAzFA98"

# If you already have the video downloaded locally, put its path here.
# Leave as empty string "" to always download from YouTube.
LOCAL_VIDEO_PATH = r"C:\Users\gutom\OneDrive\Desktop\DrApurbaTasks\VideoAnomaly\crash_output_real\input_video.mp4"

# --- ALTERAÇÃO FEITA AQUI ---
EXTRACT_FPS   = 1   # 0.5 FPS means 1 frame every 2 seconds/ 1 means 1 second per frame, etc.
MAX_FRAMES    = 60    # cap total frames to keep processing time reasonable
CHUNK_SIZE    = 15    # number of frames grouped into one temporal chunk

# BLIP model to use. "large" is more accurate, "base" is faster on CPU.
BLIP_MODEL = "Salesforce/blip-image-captioning-large"

# Keywords that indicate an anomaly in the BLIP caption
ANOMALY_KEYWORDS = [
    "crash", "collision", "accident", "impact", "smash",
    "fire", "smoke", "flames", "burning", "explosion",
    "fight", "attack", "assault", "violence",
    "fall", "falling", "collapsed", "on the ground",
    "running", "fleeing", "chasing",
    "gun", "weapon", "knife",
    "overturned", "flipped", "wrong way",
    "emergency", "ambulance", "police",
    "car", "vehicle", "truck", "motorcycle",
]

# Minimum number of keyword matches needed to mark a frame as anomalous
ANOMALY_THRESHOLD = 1

# FOLDER CLEANUP — wipe previous results so old images don't accumulate
def clean_output_dirs():
    """
    Deletes and recreates the output sub-folders (frames & collisions) so that
    images from a previous run don't pile up and waste disk space.
    The video cache is intentionally kept to avoid re-downloading.
    Includes error handling for Windows locked files (PermissionError).
    """
    print("  Cleaning previous output folders...")

    def handle_remove_readonly(func, path, exc):
        """Helper to remove read-only files on Windows."""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass # Ignore if it still fails

    # Remove individual sub-directories that hold generated images / reports
    for folder in [SCREENSHOTS_DIR, COLLISION_DIR]:
        if os.path.exists(folder):
            try:
                # Use onexc for Python 3.12+ compatibility
                shutil.rmtree(folder, onexc=handle_remove_readonly)
                print(f"    Deleted: {folder}")
            except Exception as e:
                print(f"    [WARNING] Could not completely delete {folder}. Is it open? Error: {e}")

    # Remove leftover chart / grid / report files from the last run
    for leftover in [REPORT_PATH, CHART_PATH, GRID_PATH]:
        if os.path.exists(leftover):
            try:
                os.remove(leftover)
                print(f"    Deleted: {leftover}")
            except Exception as e:
                print(f"    [WARNING] Could not delete {leftover}: {e}")

    # Small delay to let the OS release file handles before recreating
    time.sleep(0.5)

    # Re-create the now-empty directories
    os.makedirs(OUTPUT_DIR,      exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(COLLISION_DIR,   exist_ok=True)

    print("  Output folders are clean and ready.\n")

# VIDEO DOWNLOAD — supports local file or YouTube via yt-dlp
def download_video() -> str:
    """
    Returns a local path to the video file.
    Priority: LOCAL_VIDEO_PATH > cached download > fresh YouTube download.
    """
    # Use a local file if the user provided one and it actually exists
    if LOCAL_VIDEO_PATH and Path(LOCAL_VIDEO_PATH).exists():
        print(f"  Using local video: {LOCAL_VIDEO_PATH}")
        return LOCAL_VIDEO_PATH

    # Reuse a previously downloaded copy (the cache is NOT wiped on each run)
    if Path(VIDEO_CACHE).exists():
        cap = cv2.VideoCapture(VIDEO_CACHE)
        if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 10:
            cap.release()
            size_mb = Path(VIDEO_CACHE).stat().st_size / 1024 / 1024
            print(f"  Using cached download ({size_mb:.1f} MB): {VIDEO_CACHE}")
            return VIDEO_CACHE
        cap.release()

    print("  Downloading video from YouTube (this may take a few minutes)...")

    # Check that yt-dlp is available before trying to use it
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "=" * 60)
        print("ERROR: yt-dlp is not installed or not found in PATH.")
        print("Install it with:  pip install yt-dlp")
        print("Then run the script again.")
        print("=" * 60)
        sys.exit(1)

    # Download the MP4 stream
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "--output", VIDEO_CACHE,
        YOUTUBE_URL,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("  Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")
        sys.exit(1)

    if not Path(VIDEO_CACHE).exists():
        print("  Download failed – output file not found.")
        sys.exit(1)

    return VIDEO_CACHE

# FRAME EXTRACTION — sample the video at EXTRACT_FPS frames per second
def extract_frames(video_path: str) -> list:
    """
    Opens the video and returns up to MAX_FRAMES evenly-spaced PIL Images
    together with their original frame index and timestamp in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_native / native_fps
    
    # How many native frames to skip between each sampled frame
    step         = max(1, round(native_fps / EXTRACT_FPS))

    print(f"  Duration    : {duration_s:.1f}s")
    print(f"  Native FPS  : {native_fps:.0f}")
    print(f"  Sampling    : every {step} frames -> ~{native_fps / step:.1f} effective fps")

    frames = []
    idx    = 0

    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, bgr = cap.read()
        if not ret:
            break
        # Only keep frames that fall on the sampling interval
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append({
                "image"    : Image.fromarray(rgb),
                "frame_idx": idx,
                "time_sec" : round(idx / native_fps, 2),
            })
        idx += 1

    cap.release()
    print(f"  Frames extracted: {len(frames)}")
    return frames


# LOAD BLIP — download weights once, then cache them locally via HuggingFace
def load_blip():
    """
    Loads the BLIP image-captioning model and its processor.
    Uses GPU if available, otherwise falls back to CPU (slower).
    """
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device.upper()}")
    print(f"  Model  : {BLIP_MODEL}")
    if device == "cpu":
        print("  Tip: CPU is slow. Switch to 'Salesforce/blip-image-captioning-base' for speed.")

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model     = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    print("  Model loaded.")
    return processor, model, device

# FRAME DESCRIPTION — generate a natural-language caption for one frame
def describe_frame(pil_image, processor, model, device: str) -> str:
    """
    Feeds a single PIL image into BLIP with a dashcam-style prompt and
    returns the generated caption string.
    """
    import torch

    prompt = "a dashcam video frame showing"
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
            repetition_penalty=1.3,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    # Strip the prompt prefix from the output so only the description remains
    if caption.lower().startswith(prompt.lower()):
        caption = caption[len(prompt):].strip().lstrip(",").strip()
    return caption if caption else "description unavailable"

# ANOMALY SCORING — count how many keywords appear in the caption
def score_frame(description: str) -> tuple:
    """
    Scans the caption for anomaly keywords.
    Returns (hit_count, matched_keywords_list, is_anomaly_bool).
    """
    text    = description.lower()
    matched = [kw for kw in ANOMALY_KEYWORDS if kw in text]
    hits    = len(matched)
    return hits, matched, hits >= ANOMALY_THRESHOLD


# SAVE ANNOTATED SCREENSHOT — overlays status bar and colored border on frame
def save_frame_screenshot(pil_image, frame_info, description,
                           is_anomaly, chunk_idx, keywords) -> str:
    """
    Creates an annotated image with a colored status bar at the bottom.
    Anomalous frames also get a red border and are copied to COLLISION_DIR.
    """
    W, H   = pil_image.size
    BAR_H  = 80

    # Create a canvas tall enough for the image + status bar
    canvas = Image.new("RGB", (W, H + BAR_H), (0, 0, 0))
    canvas.paste(pil_image, (0, 0))
    draw   = ImageDraw.Draw(canvas)

    # Red bar for anomalies, green for normal frames
    bar_bg = (160, 15, 15) if is_anomaly else (15, 110, 15)
    draw.rectangle([(0, H), (W, H + BAR_H)], fill=bar_bg)

    status_label = "** ANOMALY DETECTED **" if is_anomaly else "Normal"
    draw.text((10, H + 5),  status_label,      fill=(255, 255, 255))
    draw.text((10, H + 24),
              f"Frame {frame_info['frame_idx']}  |  t={frame_info['time_sec']:.2f}s  |  Chunk {chunk_idx + 1}",
              fill=(210, 210, 210))

    max_chars  = W // 7
    desc_short = description[:max_chars - 2] + ".." if len(description) > max_chars else description
    draw.text((10, H + 43), f"BLIP: {desc_short}", fill=(185, 185, 185))

    if keywords:
        draw.text((10, H + 61), f"Keywords: {', '.join(keywords)}", fill=(255, 180, 80))

    # Draw a thick red border around anomalous frames for quick visual scanning
    if is_anomaly:
        draw.rectangle([(0, 0), (W - 1, H + BAR_H - 1)],
                       outline=(220, 20, 20), width=5)

    fname    = (f"frame_{frame_info['frame_idx']:05d}"
                f"_t{frame_info['time_sec']:.2f}s"
                f"_{'ANOMALY' if is_anomaly else 'normal'}.jpg")
    out_path = os.path.join(SCREENSHOTS_DIR, fname)
    canvas.save(out_path, quality=88)

    # Save a separate copy in the collisions folder for easy review
    if is_anomaly:
        coll_fname = f"COLLISION_frame{frame_info['frame_idx']:05d}_t{frame_info['time_sec']:.2f}s.jpg"
        canvas.save(os.path.join(COLLISION_DIR, coll_fname), quality=92)

    return out_path

# CHUNKING — group frames into fixed-size temporal windows
def make_chunks(frame_results: list) -> list:
    """
    Divides the list of processed frames into chunks of CHUNK_SIZE.
    A chunk is flagged as anomalous if at least one of its frames is anomalous.
    """
    chunks = []
    for i in range(0, len(frame_results), CHUNK_SIZE):
        group       = frame_results[i : i + CHUNK_SIZE]
        anom_frames = [f for f in group if f["is_anomaly"]]
        chunks.append({
            "index"        : len(chunks),
            "frame_start"  : group[0]["frame_idx"],
            "frame_end"    : group[-1]["frame_idx"],
            "time_start"   : group[0]["time_sec"],
            "time_end"     : group[-1]["time_sec"],
            "n_frames"     : len(group),
            "anomaly_count": len(anom_frames),
            "anomaly"      : len(anom_frames) > 0,
            "descriptions" : [f["description"] for f in group],
            "all_keywords" : list({kw for f in anom_frames for kw in f["keywords"]}),
        })
    return chunks


# TIMELINE CHART — bar chart of keyword hits per chunk over time
def save_timeline_chart(chunks: list):
    """
    Generates a dark-themed bar chart showing anomaly keyword hits
    for each temporal chunk, with annotation arrows on anomalous bars.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    for c in chunks:
        color = "#e84545" if c["anomaly"] else "#4fc3f7"
        w     = max(0.4, (c["time_end"] - c["time_start"]) * 0.82)
        ax.bar(c["time_start"], c["anomaly_count"], width=w, align="edge",
               color=color, alpha=0.90, zorder=3)
        if c["anomaly"]:
            mid = c["time_start"] + w / 2
            ax.annotate(f"ANOMALY\n~{c['time_start']:.0f}s",
                        xy=(mid, c["anomaly_count"]),
                        xytext=(mid + 1.5, c["anomaly_count"] + 0.4),
                        fontsize=8, fontweight="bold", color="white",
                        arrowprops=dict(arrowstyle="->", color="#ff4545", lw=1.5))

    ax.axhline(ANOMALY_THRESHOLD, color="#ffd700", linestyle="--", linewidth=2, zorder=4)
    ax.set_title("Anomaly Detection — Keyword Hits per Temporal Chunk",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("Time (seconds)", color="white", fontsize=11)
    ax.set_ylabel("Keyword hits", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for sp in ax.spines.values():
        sp.set_edgecolor("#ffffff22")
    ax.grid(axis="y", color="#ffffff1a", linestyle="--", lw=0.8, zorder=0)
    handles = [
        mpatches.Patch(color="#4fc3f7", label="Normal chunk"),
        mpatches.Patch(color="#e84545", label="Anomalous chunk"),
        plt.Line2D([0], [0], color="#ffd700", linestyle="--",
                   label=f"Threshold = {ANOMALY_THRESHOLD}"),
    ]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white",
              edgecolor="#ffffff33", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved: {CHART_PATH}")


# FRAME GRID — thumbnail overview of all processed frames in one image

def save_frame_grid(frame_results: list):
    """
    Assembles all annotated frame thumbnails into a single grid image
    so you can review the entire video at a glance.
    """
    if not frame_results:
        return

    N_COLS  = 6
    THUMB_W = 210
    THUMB_H = 130
    LABEL_H = 52
    CELL_H  = THUMB_H + LABEL_H
    n_rows  = math.ceil(len(frame_results) / N_COLS)

    grid = Image.new("RGB", (N_COLS * THUMB_W, n_rows * CELL_H), (14, 14, 26))
    draw = ImageDraw.Draw(grid)

    for pos, fr in enumerate(frame_results):
        col, row = pos % N_COLS, pos // N_COLS
        x, y     = col * THUMB_W, row * CELL_H
        is_an    = fr["is_anomaly"]

        thumb = fr["image"].resize((THUMB_W, THUMB_H), Image.LANCZOS)
        grid.paste(thumb, (x, y))

        bg = (145, 15, 15) if is_an else (15, 95, 25)
        draw.rectangle([(x, y + THUMB_H), (x + THUMB_W - 1, y + CELL_H - 1)], fill=bg)

        status = "ANOMALY" if is_an else "Normal"
        draw.text((x + 4, y + THUMB_H + 3),  status,
                  fill=(255, 80, 80) if is_an else (80, 255, 100))
        draw.text((x + 4, y + THUMB_H + 18),
                  f"t={fr['time_sec']:.1f}s  frm={fr['frame_idx']}",
                  fill=(200, 200, 200))
        desc_short = (fr["description"][:30] + ".."
                      if len(fr["description"]) > 32 else fr["description"])
        draw.text((x + 4, y + THUMB_H + 33), desc_short, fill=(165, 165, 165))

        border_col = (210, 25, 25) if is_an else (25, 155, 55)
        draw.rectangle([(x, y), (x + THUMB_W - 1, y + CELL_H - 1)],
                       outline=border_col, width=3)

    grid.save(GRID_PATH, quality=90)
    print(f"  Frame grid saved: {GRID_PATH}")


# MAIN PIPELINE!!!
def run():
    print("=" * 68)
    print("VIDEO ANOMALY DETECTION — Crash Video (YouTube + BLIP)")
    print("=" * 68)
    print(f"Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/")
    print()

    # Step 0: clean up previous run artifacts
    print("[Step 0] Cleaning previous output...")
    clean_output_dirs()

    # Step 1: get the video 
    print("[Step 1] Obtaining video...")
    video_path = download_video()
    print()

    # Step 2: pull frames from the video 
    print("[Step 2] Extracting frames...")
    frames = extract_frames(video_path)
    if not frames:
        print("ERROR: No frames were extracted.")
        sys.exit(1)
    print()

    # Step 3: load the BLIP captioning model 
    processor, model, device = load_blip()
    print()

    # Step 4: caption each frame and score for anomalies
    print("[Step 4] Analyzing each frame with BLIP...")
    print(f"  {'Frame':<7} {'Time':>7}  {'Status':<22}  Description")
    print("  " + "-" * 75)

    frame_results = []

    for i, fr in enumerate(frames):
        # Generate a natural-language description of the frame
        desc                    = describe_frame(fr["image"], processor, model, device)
        # Count how many anomaly keywords appear in that description
        hits, keywords, is_anom = score_frame(desc)
        chunk_est               = i // CHUNK_SIZE
        # Save an annotated screenshot (and a collision copy if anomalous)
        ss_path                 = save_frame_screenshot(
            fr["image"], fr, desc, is_anom, chunk_est, keywords)

        frame_results.append({
            "frame_idx"  : fr["frame_idx"],
            "time_sec"   : fr["time_sec"],
            "description": desc,
            "kw_hits"    : hits,
            "keywords"   : keywords,
            "is_anomaly" : is_anom,
            "screenshot" : ss_path,
            "image"      : fr["image"],   # kept in memory for the grid, removed before JSON export
        })

        status = "[ANOMALY] " if is_anom else "[Normal]  "
        kw_str = f"  -> {', '.join(keywords)}" if keywords else ""
        print(f"  {fr['frame_idx']:<7} {fr['time_sec']:>6.2f}s  {status}  {desc[:48]}{kw_str}")

    print()

    # Step 5: group frames into temporal chunks
    print("[Step 5] Grouping frames into temporal chunks...")
    chunks         = make_chunks(frame_results)
    anomaly_chunks = sum(1 for c in chunks if c["anomaly"])

    print(f"  {len(chunks)} chunks  |  {anomaly_chunks} anomalous  |  {len(chunks) - anomaly_chunks} normal")
    print()
    print(f"  {'Chunk':<6} {'Time window':^18} {'Frames':>6}  {'Anom. frames':>12}  Status")
    print("  " + "-" * 58)
    for c in chunks:
        st = "[ANOMALY] " if c["anomaly"] else "[Normal]  "
        print(f"  {c['index'] + 1:<6} {c['time_start']:>5.1f}s-{c['time_end']:>5.1f}s  "
              f"{c['n_frames']:>6}  {c['anomaly_count']:>12}  {st}")
    print()

    # Step 6: save visual outputs
    print("[Step 6] Saving visualizations")
    save_timeline_chart(chunks)
    save_frame_grid(frame_results)
    print()

    # Drop PIL Image objects before serializing to JSON (not JSON-serializable)
    for fr in frame_results:
        fr.pop("image", None)
    for c in chunks:
        c.pop("frames", None)

    # Step 7: write the JSON report 
    report = {
        "generated_at": datetime.now().isoformat(),
        "video_path"  : os.path.abspath(video_path),
        "blip_model"  : BLIP_MODEL,
        "device"      : device,
        "config": {
            "extract_fps"      : EXTRACT_FPS,
            "max_frames"       : MAX_FRAMES,
            "chunk_size"       : CHUNK_SIZE,
            "anomaly_threshold": ANOMALY_THRESHOLD,
            "anomaly_keywords" : ANOMALY_KEYWORDS,
        },
        "summary": {
            "total_frames"  : len(frame_results),
            "normal_frames" : sum(1 for f in frame_results if not f["is_anomaly"]),
            "anomaly_frames": sum(1 for f in frame_results if f["is_anomaly"]),
            "total_chunks"  : len(chunks),
            "normal_chunks" : len(chunks) - anomaly_chunks,
            "anomaly_chunks": anomaly_chunks,
        },
        "frames": frame_results,
        "chunks": chunks,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Step 7] Report saved: {REPORT_PATH}")

    # Final summary 
    print()
    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"  Total frames    : {report['summary']['total_frames']}")
    print(f"  Normal frames   : {report['summary']['normal_frames']}")
    print(f"  Anomalous frames: {report['summary']['anomaly_frames']}")
    print(f"  Anomalous chunks: {anomaly_chunks} / {len(chunks)}")
    print()

    anom_list = [f for f in frame_results if f["is_anomaly"]]
    if anom_list:
        print("  ANOMALOUS FRAMES:")
        for f in anom_list:
            print(f"    t={f['time_sec']:.2f}s  frame={f['frame_idx']}")
            print(f"    Description : {f['description']}")
            print(f"    Keywords    : {f['keywords']}")
            print(f"    Screenshot  : {f['screenshot']}")
            print()
    else:
        print("  No anomalies detected in this video.")
        print("  (This is expected if the video does not contain crash/fire/etc.)")

    print("  GENERATED FILES:")
    print(f"    All frame screenshots : {SCREENSHOTS_DIR}/  ({len(frame_results)} files)")
    print(f"    Collision frames      : {COLLISION_DIR}/")
    print(f"    Frame grid overview   : {GRID_PATH}")
    print(f"    Anomaly timeline chart: {CHART_PATH}")
    print(f"    JSON report           : {REPORT_PATH}")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run()