import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.client.executor import RequestExecutor


class ExecutorSessionForwardingTests(unittest.TestCase):
    def test_non_streaming_acquire_forwards_session_id(self):
        source = inspect.getsource(RequestExecutor._execute_non_streaming)

        self.assertIn("session_id=context.session_id", source)

    def test_streaming_acquire_forwards_session_id(self):
        source = inspect.getsource(RequestExecutor._execute_streaming)

        self.assertIn("session_id=context.session_id", source)


if __name__ == "__main__":
    unittest.main()
