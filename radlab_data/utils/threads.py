import threading

from typing import List
from abc import ABC, abstractmethod


class ThreadWorker(ABC):
    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._thread = threading.Thread(target=self._task)
        self._enabled = True

    def start(self) -> None:
        if not self._enabled:
            raise Exception("Cannot run again a disabled task.")
        if self._debug:
            print(f"Task running {self.__class__.__name__}")
        self._thread.start()

    @abstractmethod
    def _task(self) -> None:
        pass

    def disable(self):
        if self._debug:
            print(f"Task disabled {self.__class__.__name__}")
        self._enabled = False

    def join(self):
        self._thread.join()
        if self._debug:
            print(f"Task finished {self.__class__.__name__}")


class WorkerCluster:
    def __init__(
        self, name: str, threads: List[ThreadWorker], debug: bool = False
    ) -> None:
        self._debug = debug
        self._name = self.__class__.__name__
        if name is not None:
            self._name = name
        self._threads = threads

    def start(self):
        if self._debug:
            print(f"Running cluster {self._name}")
        for thread in self._threads:
            thread.start()

    def disable(self):
        if self._debug:
            print(f"Disabling cluster {self._name}")
        for thread in self._threads:
            thread.disable()

    def join(self):
        if self._debug:
            print(f"Joining cluster {self._name}")
        for thread in self._threads:
            thread.join()
