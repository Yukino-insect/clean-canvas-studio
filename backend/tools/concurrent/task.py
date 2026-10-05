import functools
from typing import Optional

from PySide6.QtCore import QObject, Signal, QRunnable

from .future import Future


class WorkerSignal(QObject):
    finished = Signal(object)


class BaseTask(QRunnable):
    def __init__(self, _id: int, future: Future):
        super().__init__()
        self._signal = WorkerSignal()  # Signal(object)
        self._future = future
        self._id = _id
        self._exception: Optional[BaseException] = None
        self._semaphore = future.semaphore

    @property
    def finished(self):
        return self._signal.finished

    @property
    def signal(self):
        return self._signal

    def _taskDone(self, **data):
        for d in data.items():
            self._future.setExtra(*d)
        try:
            # 应用关闭后，Qt 可能已销毁 WorkerSignal，但全局线程池中的任务
            # 才刚结束。此时不再向已删除对象发信号，仍需释放 Future 等待者。
            self._signal.finished.emit(self._future)
        except RuntimeError:
            pass
        finally:
            self._semaphore.release(1)


class Task(BaseTask):
    def __init__(self, _id: int, future: Future, target: functools.partial, args, kwargs):
        super().__init__(_id=_id, future=future)
        self._target = target
        self._kwargs = kwargs
        self._args = args

    def run(self) -> None:
        try:
            self._taskDone(result=self._target(*self._args, **self._kwargs))
        except Exception as exception:
            self._taskDone(exception=exception)
