import numpy as np

from src.capture.frame_session import FrameSession


def test_first_sample_starts_session_and_copies_frame() -> None:
    session = FrameSession(required_frames=2)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    session.add(frame, now=10.0)
    frame[0, 0] = 255

    assert session.collected_count == 1
    assert session.started_at == 10.0
    assert session.last_sampled_at == 10.0
    assert np.array_equal(session.get_frames()[0][0, 0], [0, 0, 0])


def test_sampling_interval_and_timeout_are_enforced_by_queries() -> None:
    session = FrameSession(sample_interval=0.15, timeout=2.0)

    assert session.should_sample(10.0) is True
    assert session.has_timed_out(10.0) is False

    session.add(np.zeros((1, 1), dtype=np.uint8), now=10.0)

    assert session.should_sample(10.14) is False
    assert session.should_sample(10.15) is True
    assert session.has_timed_out(11.99) is False
    assert session.has_timed_out(12.0) is True


def test_session_keeps_only_required_number_of_frames() -> None:
    session = FrameSession(required_frames=2)

    for value in range(3):
        session.add(np.full((1, 1), value, dtype=np.uint8), now=float(value))

    assert session.is_complete is True
    assert [int(frame[0, 0]) for frame in session.get_frames()] == [1, 2]


def test_reset_clears_session_state() -> None:
    session = FrameSession()
    session.add(np.zeros((1, 1), dtype=np.uint8), now=1.0)

    session.reset()

    assert session.collected_count == 0
    assert session.started_at is None
    assert session.last_sampled_at is None
