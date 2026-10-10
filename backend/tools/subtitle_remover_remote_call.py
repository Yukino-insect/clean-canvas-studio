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
        # Keep the consumer thread so the caller can wait for the child
        # process' explicit FINISH message.  Stopping this thread as soon as
        # Process.join() returns loses messages that have reached the pipe but
        # have not yet been dispatched to the GUI.
        self._consumer_thread = threading.Thread(target=self.run, daemon=True)
        self._consumer_thread.start()

    def run(self):
        """持续分发队列消息；收到 FINISH 后停止消费并释放线程状态。"""
        try:
            # FINISH is placed after every preceding progress/log/preview
            # message by the same child process.  Therefore consuming through
            # this sentinel guarantees that the GUI has received all updates
            # for the task before this bridge is considered complete.
            while True:
                cmd, args = self.queue.get(block=True)
                if cmd == Command.FINISH:
                    break
                callback = self.callbacks.get(cmd)
                if callback:
                    callback(*args)
        finally:
            self.running = False

    def stop(self):
        """强制停止本地消费线程；仅用于子进程未发送 FINISH 的异常场景。"""
        if not self._consumer_thread.is_alive():
            return
        self.running = False
        # ``Queue.get()`` 可能正在阻塞。放入本地 FINISH 哨兵可使它及时退出；
        # 这个方法只在超时/中断回退路径调用，不会截断正常任务的消息流。
        self.queue.put((Command.FINISH, ()))
        self._consumer_thread.join(timeout=1)

    def wait_for_finish(self, timeout=None):
        """等待消费者处理完子进程发来的 FINISH 哨兵。

        返回 ``True`` 表示所有排在 FINISH 之前的消息均已分发；返回 ``False``
        表示调用方应按异常或中断路径主动停止桥接线程。
        """
        self._consumer_thread.join(timeout=timeout)
        return not self._consumer_thread.is_alive()

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
