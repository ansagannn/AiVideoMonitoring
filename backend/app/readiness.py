"""Report observed stream/inference readiness without triggering downloads."""
from pathlib import Path
import importlib.util
import os
import time
import analyzer
from stream_capture import FrameCache


def model_status():
    path = Path(os.getenv('AI_MONITOR_YOLO_MODEL', str(Path(__file__).resolve().parents[1]/'models/yolov8n.pt')))
    return {
        'model_loaded': analyzer._model is not None,
        'weights_present': path.is_file(),
        'dependency_present': importlib.util.find_spec('ultralytics') is not None,
        'load_failed': analyzer._model_failed,
        'action_method': 'temporal_bbox_heuristics',
        'validated_on_client_cameras': False,
    }


def camera_status(source, runtime):
    frame = FrameCache.get_instance().get(source.id)
    age = max(0, time.time()-frame.captured_at) if frame else None
    state = runtime.status().get(source.id)
    inference = runtime.detections_payload(source.id)
    demo = source.source_type == 'retail_scene'
    fresh = bool(source.enabled and state and state.online and age is not None and age < (75 if source.source_type == 'jpeg_snapshot' else 5))
    model = model_status()
    ready = fresh and not demo and model['model_loaded'] and inference['status'] == 'running'
    reason = 'ready' if ready else ('disabled' if not source.enabled else 'demo_source' if demo else 'no_fresh_frame' if not fresh else 'model_unavailable')
    return {
        'camera_id': source.id,
        'stream_connected': fresh,
        'frame_age_seconds': round(age,2) if age is not None else None,
        'ai_ready': ready,
        'reason': reason,
        'analysis_status': inference['status'],
        'model': model,
        'action_analysis_available': ready and source.source_type != 'jpeg_snapshot',
        'capture_interval_seconds': 60 if source.source_type == 'jpeg_snapshot' else 1 / source.fps_limit,
    }
