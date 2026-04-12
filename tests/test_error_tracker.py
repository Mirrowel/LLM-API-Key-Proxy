import threading
import pytest

from rotator_library.error_tracker import ErrorTracker, get_error_tracker
import rotator_library.error_tracker as error_tracker_module

@pytest.fixture(autouse=True)
def reset_error_tracker():
    """Reset the global _error_tracker to None before and after each test."""
    original_tracker = error_tracker_module._error_tracker
    if error_tracker_module._error_tracker:
        error_tracker_module._error_tracker.clear()
    error_tracker_module._error_tracker = None
    yield
    if error_tracker_module._error_tracker:
        error_tracker_module._error_tracker.clear()
    error_tracker_module._error_tracker = original_tracker

def test_get_error_tracker_returns_instance():
    """Verify that get_error_tracker returns an ErrorTracker instance."""
    tracker = get_error_tracker()
    assert isinstance(tracker, ErrorTracker)

def test_get_error_tracker_singleton():
    """Verify that get_error_tracker returns the same instance on multiple calls."""
    tracker1 = get_error_tracker()
    tracker2 = get_error_tracker()
    assert tracker1 is tracker2

def test_get_error_tracker_thread_safe_initialization():
    """Verify thread-safe lazy initialization of the error tracker singleton."""
    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results = []

    def worker():
        try:
            barrier.wait(timeout=5.0)
            results.append(get_error_tracker())
        except threading.BrokenBarrierError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Thread hung indefinitely"

    assert len(results) == num_threads

    # All threads should have received the exact same instance
    first_instance = results[0]
    for instance in results[1:]:
        assert instance is first_instance
