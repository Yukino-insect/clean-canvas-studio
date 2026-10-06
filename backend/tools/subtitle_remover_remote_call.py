import multiprocessing
import threading
from enum import Enum

class Command(Enum):
    """子进程向 GUI 进程发送的有序消息类型。"""
    FINISH = 0,
    PROGRESS = 1,
    LOG = 2,
    MANAGE_PROCESS = 3,
    ERROR = 4,
    UPDATE_PREVIEW_WITH_COMP = 5,

class SubtitleRemoverRemoteCall:
    """将子进程回调转发到 GUI 进程的队列桥。

    ``multiprocessing.Process`` 不能直接调用 Qt 槽函数，因此子进程只写入
    ``queue``；守护线程在主进程取出消息后再执行对应 callback。
    """
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.callbacks = {}
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        """持续分发队列消息；收到 FINISH 后停止消费并释放线程状态。"""
        try:
            while self.running:
                cmd, args = self.queue.get(block=True)
                if cmd == Command.FINISH:
                    break
                callback = self.callbacks.get(cmd)
                if callback:
                    callback(*args)
        finally:
            self.running = False

    def stop(self):
        """请求本地消费线程停止；不主动终止发送消息的子进程。"""
        self.running = False

    def register_update_progress_callback(self, callback):
        self.callbacks[Command.PROGRESS] = callback

    def register_log_callback(self, callback):
        self.callbacks[Command.LOG] = callback

    def register_manage_process_callback(self, callback):
        self.callbacks[Command.MANAGE_PROCESS] = callback

    def register_update_preview_with_comp_callback(self, callback):
        self.callbacks[Command.UPDATE_PREVIEW_WITH_COMP] = callback

    def register_error_callback(self, callback):
        self.callbacks[Command.ERROR] = callback

    @staticmethod
    def remote_call_update_progress(queue, progress, isFinished):
        queue.put((Command.PROGRESS, (progress, isFinished,)))

    @staticmethod
    def remote_call_append_log(queue, *args):
        queue.put((Command.LOG, (*args,)))

    @staticmethod
    def remote_call_finish(queue, *args):
        queue.put((Command.FINISH, (None,)))

    @staticmethod
    def remote_call_catch_error(queue, e):
        queue.put((Command.ERROR, (e,)))

    @staticmethod
    def remote_call_manage_process(queue, pid):
        queue.put((Command.MANAGE_PROCESS, (pid,)))

    @staticmethod
    def remote_call_update_preview_with_comp(queue, *args):
        queue.put((Command.UPDATE_PREVIEW_WITH_COMP, (*args,)))
