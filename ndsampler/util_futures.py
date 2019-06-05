# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import concurrent.futures

__all__ = ['SerialExecutor']


# class FakeCondition(object):
#     def aquire(self):
#         pass

#     def release(self):
#         pass


class SerialFuture(concurrent.futures.Future):
    """
    Non-threading / multiprocessing version of future for drop in compatibility
    with concurrent.futures.
    """
    def __init__(self, func, *args, **kw):
        super(SerialFuture, self).__init__()
        self.func = func
        self.args = args
        self.kw = kw
        # self._condition = FakeCondition()
        self._run_count = 0
        # fake being finished to cause __get_result to be called
        self._state = concurrent.futures._base.FINISHED

    def _run(self):
        result = self.func(*self.args, **self.kw)
        self.set_result(result)
        self._run_count += 1

    def _Future__get_result(self):
        # overrides private __getresult method
        if not self._run_count:
            self._run()
        return self._result


class SerialExecutor(object):
    """
    Implements the concurrent.futures API around a single-threaded backend

    Example:
        >>> with SerialExecutor() as executor:
        >>>     futures = []
        >>>     for i in range(100):
        >>>         f = executor.submit(lambda x: x + 1, i)
        >>>         futures.append(f)
        >>>     for f in concurrent.futures.as_completed(futures):
        >>>         assert f.result() > 0
        >>>     for i, f in enumerate(futures):
        >>>         assert i + 1 == f.result()
        """
    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, tb):
        pass

    def submit(self, func, *args, **kw):
        return SerialFuture(func, *args, **kw)

    def shutdown(self):
        pass


class Executor(object):
    """
    Wrapper around a specific executor.

    Abstracts Serial, Thread, and Process Executor via arguments.
    """

    def __init__(self, mode='serial', max_workers=0):
        from concurrent import futures
        if mode == 'serial' or max_workers == 0:
            backend = SerialExecutor()
        elif mode == 'thread':
            backend = futures.ThreadPoolExecutor(max_workers=max_workers)
        elif mode == 'process':
            backend = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.backend = backend

    def __enter__(self):
        return self.backend.__enter__()

    def __exit__(self, ex_type, ex_value, tb):
        return self.backend.__exit__(ex_type, ex_value, tb)

    def submit(self, func, *args, **kw):
        return self.backend.submit(func, *args, **kw)

    def shutdown(self):
        return self.backend.shutdown()