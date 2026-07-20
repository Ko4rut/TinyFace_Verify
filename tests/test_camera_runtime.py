import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.camera_runtime import CameraRuntime
from pathlib import Path


def test_init_creates_empty_session():
    mock_camera = Mock()
    
    runtime = CameraRuntime(
        camera = mock_camera,
        required_frames=5,
        sample_interval=0.15,
        session_timeout=2.0,
    )
    
    assert runtime.camera is mock_camera
    assert runtime.required_frames == 5
    assert runtime.sample_interval == 0.15
    assert runtime.session_timeout == 2.0

    assert len(runtime.selected_frames) == 0
    assert runtime.session_started_at is None
    assert runtime.last_sampled_at is None
    
def test_reset_session_clears_all_session_data():
    runtime = CameraRuntime(camera=Mock())
    runtime.selected_frames.append(
        np.zeros((10, 10, 3), dtype=np.uint8)
    )
    runtime.session_started_at = 10.0
    runtime.last_sampled_at = 11.0
    runtime.reset_session()
    
    assert len(runtime.selected_frames) == 0
    assert runtime.session_started_at is None
    assert runtime.last_sampled_at is None
    
def test_session_does_not_timeout_before_it_starts():
    runtime = CameraRuntime(
        camera=Mock(),
        session_timeout=2.0,
    )

    result = runtime.session_has_timed_out(now=100.0)

    assert result is False
    
def test_session_does_not_timeout_before_it_starts():
    runtime = CameraRuntime(
        camera=Mock(),
        session_timeout=2.0,
    )

    result = runtime.session_has_timed_out(now=100.0)

    assert result is False
    
def test_session_does_not_timeout_before_limit():
    runtime = CameraRuntime(
        camera=Mock(),
        session_timeout=2.0,
    )
    runtime.session_started_at = 10.0

    result = runtime.session_has_timed_out(now=11.9)

    assert result is False
    
def test_session_times_out_at_limit():
    runtime = CameraRuntime(
        camera=Mock(),
        session_timeout=2.0,
    )
    runtime.session_started_at = 10.0

    result = runtime.session_has_timed_out(now=12.0)

    assert result is True
    
def test_should_sample_first_frame():
    runtime = CameraRuntime(
        camera=Mock(),
        sample_interval=0.15,
    )

    result = runtime.should_sample(now=10.0)

    assert result is True
    
def test_should_not_sample_before_interval():
    runtime = CameraRuntime(
        camera=Mock(),
        sample_interval=0.15,
    )
    runtime.last_sampled_at = 10.0

    result = runtime.should_sample(now=10.14)

    assert result is False
    
def test_should_sample_at_interval():
    runtime = CameraRuntime(
        camera=Mock(),
        sample_interval=1.0,
    )
    runtime.last_sampled_at = 10.0

    assert runtime.should_sample(now=11.0) is True
    
def test_collect_first_frame_starts_session():
    runtime = CameraRuntime(camera=Mock())
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    runtime.collect_frame(fake_frame, now=20.0)

    assert len(runtime.selected_frames) == 1
    assert runtime.session_started_at == 20.0
    assert runtime.last_sampled_at == 20.0
    
def test_collect_next_frame_keeps_original_start_time():
    runtime = CameraRuntime(camera=Mock())
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    runtime.collect_frame(fake_frame, now=20.0)
    runtime.collect_frame(fake_frame, now=20.5)

    assert len(runtime.selected_frames) == 2
    assert runtime.session_started_at == 20.0
    assert runtime.last_sampled_at == 20.5
    
def test_collect_frame_stores_independent_copy():
    runtime = CameraRuntime(camera=Mock())
    original_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    runtime.collect_frame(original_frame, now=20.0)

    original_frame[0, 0] = [255, 255, 255]

    stored_frame = runtime.selected_frames[0]

    assert np.array_equal(
        stored_frame[0, 0],
        [0, 0, 0],
    )
    
def test_session_is_not_complete_with_four_frames():
    runtime = CameraRuntime(
        camera=Mock(),
        required_frames=5,
    )
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for index in range(4):
        runtime.collect_frame(fake_frame, now=float(index))

    assert runtime.is_session_complete() is False
    
def test_session_is_complete_with_five_frames():
    runtime = CameraRuntime(
        camera=Mock(),
        required_frames=5,
    )
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for index in range(5):
        runtime.collect_frame(fake_frame, now=float(index))

    assert runtime.is_session_complete() is True    
    
def test_selected_frames_does_not_exceed_required_frames():
    runtime = CameraRuntime(
        camera=Mock(),
        required_frames=5,
    )

    for index in range(6):
        frame = np.full(
            (2, 2, 3),
            fill_value=index,
            dtype=np.uint8,
        )
        runtime.collect_frame(frame, now=float(index))

    assert len(runtime.selected_frames) == 5

    first_remaining_frame = runtime.selected_frames[0]

    assert np.all(first_remaining_frame == 1)