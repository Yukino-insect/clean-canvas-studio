"""内容修复子进程进度桥的回归测试。"""

from __future__ import annotations

import time
import unittest

from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall


class ContentRepairProgressBridgeTests(unittest.TestCase):
    def test_finish_waits_for_all_progress_messages_before_returning(self):
        """FINISH 之前排队的进度必须全部转发，不能只留下启动时的 1%。"""
        bridge = SubtitleRemoverRemoteCall()
        received = []

        def receive_progress(progress, _finished):
            # 放大消费者和生产者的速度差，复现任务结束时队列仍有积压的情形。
            time.sleep(0.01)
            received.append(progress)

        bridge.register_update_progress_callback(receive_progress)
        try:
            for progress in (1, 25, 50, 75, 100):
                bridge.remote_call_update_progress(bridge.queue, progress, progress == 100)
            bridge.remote_call_finish(bridge.queue)

            self.assertTrue(bridge.wait_for_finish(timeout=2))
            self.assertEqual(received, [1, 25, 50, 75, 100])
        finally:
            bridge.stop()
            bridge.queue.close()
            bridge.queue.join_thread()


if __name__ == "__main__":
    unittest.main()
