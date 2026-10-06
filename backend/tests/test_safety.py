from unittest.mock import patch
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
import db
from main import app
from safety import SafetyEngine
from models import Zone, MonitoringSettings, utc_now
from tracking import RuntimeTrack, MultiCameraTracker, RuntimeDetection, zone_for_bbox
from stream_capture import FrameCache, CapturedFrame, StreamSource, grab_frame, close_captures
from analyzer import analyze_frame
from notifications import Notifier


def track(box,local_id=1,zone=None):
    now=utc_now()
    return RuntimeTrack(local_id,'cam-hypermarket-frozen',now,now,box,.9,zone=zone)


def source():
    s=db.list_camera_sources()[0]
    s.source_type='rtsp'
    return s


def test_fall_transition_and_debounce():
    engine=SafetyEngine();s=source();t=track([50,20,90,160]);settings=MonitoringSettings(fall_hold_seconds=1)
    engine.evaluate(s,[t],settings,now=0)
    t.bbox=[30,100,170,140]
    assert engine.evaluate(s,[t],settings,now=.5)=='normal'
    assert engine.evaluate(s,[t],settings,now=1)=='normal'
    assert engine.evaluate(s,[t],settings,now=1.6)=='fall'
    engine.evaluate(s,[t],settings,now=2)
    assert len(db.list_events(event_type='fall'))==1


def test_horizontal_alone_is_not_fall_and_low_confidence_ignored():
    e=SafetyEngine();s=source();t=track([10,90,150,120]);settings=MonitoringSettings()
    e.evaluate(s,[t],settings,now=0)
    assert e.evaluate(s,[t],settings,now=3)=='normal'
    t.confidence=.1
    assert e.evaluate(s,[t],settings,now=4)=='normal'
    assert not db.list_events()


def test_danger_needs_actual_polygon_and_suspicious_dwell():
    z=Zone(id='danger',name='Danger',kind='restricted',polygon=[(0,0),(.3,0),(.3,1),(0,1)])
    assert zone_for_bbox([70,10,90,80],[z],(100,100)) is None
    assert zone_for_bbox([0,10,20,80],[z],(100,100))==z
    s=source();e=SafetyEngine();t=track([0,10,20,80],zone=z)
    assert e.evaluate(s,[t],MonitoringSettings(),now=0)=='danger'
    with pytest.raises(ValueError): Zone(id='bad',name='Bad',kind='restricted')
    z=Zone(id='shelf',name='Shelf',kind='shelf');t.zone=z;t.dwell_by_zone[z.id]='2000-01-01T00:00:00Z'
    assert e.evaluate(s,[t],MonitoringSettings(),now=1)=='suspicious'


def test_fight_requires_sustained_motion_and_proximity():
    s=source();engine=SafetyEngine();settings=MonitoringSettings(fight_motion_threshold=.2)
    a=track([10,10,70,130],1);b=track([35,10,95,130],2)
    engine.evaluate(s,[a,b],settings,now=0)
    for index in range(1,5):
        a.bbox=[v+(20 if j%2==0 else 0) for j,v in enumerate(a.bbox)]
        b.bbox=[v+(20 if j%2==0 else 0) for j,v in enumerate(b.bbox)]
        result=engine.evaluate(s,[a,b],settings,now=index*.3)
    assert result=='fight'
    assert len(db.list_events(event_type='fight'))==1


def test_tracker_one_to_one_and_fall_identity():
    tracker=MultiCameraTracker()
    first=tracker.update('c',[RuntimeDetection([40,20,80,140],.9)],[],(200,200))[0]
    new=tracker.update('c',[RuntimeDetection([20,100,150,140],.9)],[],(200,200))[0]
    assert new.id==first.id
    tracks=tracker.update('c',[RuntimeDetection([20,100,150,140],.9),RuntimeDetection([22,100,152,140],.9)],[],(200,200))
    assert len(tracks)==2


def test_settings_and_sources_survive_restart():
    s=source();s.url='rtsp://local/changed';s.enabled=False
    db.upsert_camera_source(s);db.save_settings(MonitoringSettings(fall_hold_seconds=4))
    db.init_db()
    saved=next(x for x in db.list_camera_sources() if x.id==s.id)
    assert saved.url==s.url and not saved.enabled
    assert db.load_settings().fall_hold_seconds==4


def test_missing_model_not_success():
    with patch('analyzer._get_model',return_value=None):
        result=analyze_frame('test',np.zeros((20,20,3),dtype=np.uint8))
    assert not result.model_available


def test_api_snapshot_review_permissions():
    frame=np.full((30,30,3),150,dtype=np.uint8);jpeg=cv2.imencode('.jpg',frame)[1].tobytes()
    FrameCache.get_instance().update(CapturedFrame(camera_id='cam-hypermarket-frozen',jpeg_bytes=jpeg,numpy_frame=frame))
    with TestClient(app) as client:
        assert client.get('/api/events').status_code==401
        assert client.get('/api/reports/shift.csv').status_code==401
        login=client.post('/api/auth/login',json={'username':'admin','password':'admin123'}).json()
        headers={'Authorization':'Bearer '+login['access_token']}
        for kind in ['normal','suspicious','fight','fall','danger']:
            response=client.post('/api/events/simulate',headers=headers,json={'camera_id':'cam-hypermarket-frozen','event_type':kind})
            assert response.status_code==200,response.text
            event=response.json();assert event['detection_mode']=='demo'
            assert client.get(event['snapshot_url'],headers=headers).content==jpeg
        FrameCache.get_instance().update(CapturedFrame(camera_id='cam-hypermarket-frozen',jpeg_bytes=b'different',numpy_frame=frame))
        assert client.get(event['snapshot_url'],headers=headers).content==jpeg
        result=client.post('/api/events/'+event['id']+'/feedback',headers=headers,json={'status':'confirmed','reviewed_by':'spoof','note':'Checked'}).json()
        assert result['reviewed_by']=='admin' and result['status']=='confirmed'
        assert client.post('/api/events/simulate',headers=headers,json={'camera_id':'missing','event_type':'fall'}).status_code==404
        assert client.post('/api/telegram/callback',json={}).status_code==403
        op=client.post('/api/auth/login',json={'username':'operator','password':'operator123'}).json()
        op_headers={'Authorization':'Bearer '+op['access_token']}
        assert client.post('/api/events/simulate',headers=op_headers,json={}).status_code==403
        assert client.put('/api/settings',headers=op_headers,json=MonitoringSettings().model_dump()).status_code==403


def test_delivery_only_once_after_success_and_skips_normal():
    with db.connect() as con: con.execute('CREATE TABLE alert_attempts (event_id TEXT PRIMARY KEY,next_attempt REAL NOT NULL)')
    risk=db.create_event('cam-hypermarket-frozen','fall',rule_id='manual_simulate')
    normal=db.create_event('cam-hypermarket-frozen','normal',rule_id='manual_simulate')
    with patch('telegram.send_message') as sender:
        sender.return_value.sent=True
        notifier=Notifier();notifier.dispatch();notifier.dispatch()
        assert sender.call_count==1
        assert 'ДЕМО' in sender.call_args[0][0]
    assert db.get_event(risk.id).telegram_sent
    assert not db.get_event(normal.id).telegram_sent


def test_video_advances(tmp_path):
    path=tmp_path/'test.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),8,(64,64))
    for brightness in [30,90,150]: writer.write(np.full((64,64,3),brightness,dtype=np.uint8))
    writer.release();s=StreamSource(camera_id='file',name='File',url=str(path),stream_type='demo_video')
    a,b=grab_frame(s),grab_frame(s)
    assert a is not None and b is not None
    assert b.numpy_frame.mean()>a.numpy_frame.mean()+30
    close_captures()


def test_camera_readiness_never_marks_demo_or_stale_frame_as_real_ai():
    from readiness import camera_status
    from camera_runtime import CameraRuntime
    from models import StreamStatus
    s=source();s.enabled=True;runtime=CameraRuntime()
    with patch('readiness.model_status',return_value={'model_loaded':True}), patch.object(runtime,'status',return_value={s.id:StreamStatus(online=True,has_frame=True)}), patch.object(runtime,'detections_payload',return_value={'status':'running'}):
        assert camera_status(s,runtime)['reason']=='no_fresh_frame'
        frame=np.zeros((10,10,3),dtype=np.uint8)
        FrameCache.get_instance().update(CapturedFrame(camera_id=s.id,jpeg_bytes=b'frame',numpy_frame=frame))
        assert camera_status(s,runtime)['ai_ready']
        s.source_type='retail_scene'
        assert camera_status(s,runtime)['reason']=='demo_source'
        assert not camera_status(s,runtime)['ai_ready']


def test_public_sources_and_hls_capture_dispatch():
    from defaults import default_sources, PUBLIC_SOURCE_PAGES
    public = [s for s in default_sources() if s.id in PUBLIC_SOURCE_PAGES]
    assert len(public) == 4 and all(s.enabled for s in public)
    assert len([s for s in public if s.source_type == 'jpeg_snapshot']) == 2
    hls = next(s for s in public if s.source_type == 'hls')
    _, jpeg = cv2.imencode('.jpg', np.zeros((40,60,3), dtype=np.uint8))
    with patch('stream_capture._grab_hls_frame', return_value=jpeg.tobytes()) as video, patch('stream_capture._grab_mjpeg_frame') as mjpeg:
        frame = grab_frame(StreamSource(hls.id, hls.name, hls.url, 'hls'))
        assert frame.width == 60 and frame.height == 40
        video.assert_called_once_with(hls.id, hls.url)
        mjpeg.assert_not_called()


def test_operator_mjpeg_uses_configured_url_and_nonblocking_capture(monkeypatch):
    from defaults import default_sources
    monkeypatch.setenv('AI_MONITOR_PUBLIC_ONLY', '1')
    monkeypatch.setenv('AI_MONITOR_OPERATOR_MJPEG_URL', 'http://camera.example:8090/mjpg/video.mjpg')
    camera = next(s for s in default_sources() if s.id == 'cam-public-operator-212604')
    assert camera.enabled and camera.source_type == 'live_mjpeg'
    assert camera.url == 'http://camera.example:8090/mjpg/video.mjpg'
    _, jpeg = cv2.imencode('.jpg', np.zeros((40,60,3), dtype=np.uint8))
    with patch('stream_capture._grab_hls_frame', return_value=jpeg.tobytes()) as video, patch('stream_capture._grab_mjpeg_frame') as blocking:
        frame = grab_frame(StreamSource(camera.id, camera.name, camera.url, camera.source_type))
        assert frame.width == 60 and frame.height == 40
        video.assert_called_once_with(camera.id, camera.url)
        blocking.assert_not_called()


def test_snapshot_readiness_respects_minute_refresh():
    import time
    from readiness import camera_status
    from models import StreamStatus
    s = next(s for s in db.list_camera_sources() if s.id == 'cam-public-yellowstone-arch')
    FrameCache.get_instance().update(CapturedFrame(s.id,b'',np.zeros((5,5,3),dtype=np.uint8),captured_at=time.time()-30))
    class Runtime:
        def status(self): return {s.id:StreamStatus(online=True,has_frame=True)}
        def detections_payload(self, camera_id): return {'status':'running'}
    with patch('readiness.model_status',return_value={'model_loaded':True}):
        result = camera_status(s, Runtime())
    assert result['stream_connected'] and result['ai_ready']
    assert not result['action_analysis_available']
    assert result['capture_interval_seconds'] == 60


def test_free_demo_skips_yolo_import(monkeypatch):
    from analyzer import _get_model
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != 'ultralytics'
        return original(name, *args, **kwargs)
    monkeypatch.setenv('AI_MONITOR_DISABLE_INFERENCE','1')
    with patch('builtins.__import__', side_effect=guarded):
        assert _get_model() is None
    monkeypatch.setenv('AI_MONITOR_PUBLIC_ONLY','1')
    from defaults import default_sources
    assert all(s.id.startswith('cam-public-') for s in default_sources())


def test_onnx_person_boxes_rescale_filter_and_suppress():
    from analyzer import _decode_people
    output = np.zeros((1,84,3), dtype=np.float32)
    output[0,:4,0] = [80,100,40,80]
    output[0,4,0] = .9
    output[0,:4,1] = [81,101,40,80]
    output[0,4,1] = .8
    output[0,:4,2] = [200,200,40,40]
    output[0,6,2] = .95  # car: ignore
    detections = _decode_people(output, 2, 640, 480, .35)
    assert len(detections) == 1
    assert [detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2] == [120,120,200,280]


def test_caltrans_catalog_uses_official_stream_urls_only():
    import json
    import stream_capture as capture
    capture._catalogs.clear()
    payload = {"data":[{"cctv":{"imageData":{"streamingVideoURL":"https://wzmedia.dot.ca.gov/D10/test.stream/playlist.m3u8"}}},
                       {"streamingVideoURL":"https://untrusted.example/playlist.m3u8"}]}
    with patch('stream_capture.urlopen') as fetch:
        fetch.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        url = capture._catalog_stream('catalog-test', 'https://cwwp2.dot.ca.gov/data/d10/cctv/cctvStatusD10.json')
    assert url == 'https://wzmedia.dot.ca.gov/D10/test.stream/playlist.m3u8'
    assert len(capture._catalogs['catalog-test']['urls']) == 1
    capture._catalogs.clear()
