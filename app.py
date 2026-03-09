from flask import Flask, request, jsonify
import cv2
import requests
import numpy as np
import base64
import tempfile
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    data = request.json

    video_url = data.get('video_url')         # Google Drive direct download URL
    timestamps = data.get('timestamps', [])    # List of seconds, e.g. [2, 6, 10]
    job_name = data.get('job_name', 'job')

    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    # Download the video to a temp file
    try:
        response = requests.get(video_url, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 500

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 500

        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If no timestamps provided, extract one frame from the middle
        if not timestamps:
            timestamps = [duration / 2]

        frames = []
        for ts in timestamps:
            # Clamp timestamp to valid range
            ts = min(max(float(ts), 0), duration - 0.1)
            frame_number = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Encode frame as JPEG then base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append({
                    "timestamp": ts,
                    "base64": b64,
                    "shot_index": timestamps.index(ts)
                })
            else:
                frames.append({
                    "timestamp": ts,
                    "base64": None,
                    "shot_index": timestamps.index(ts),
                    "error": "Could not read frame at this timestamp"
                })

        cap.release()

        return jsonify({
            "job_name": job_name,
            "video_metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": duration
            },
            "frames": frames
        })

    finally:
        os.unlink(tmp_path)  # Always clean up the temp file


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
