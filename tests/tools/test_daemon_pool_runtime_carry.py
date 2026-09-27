"""Runnable with stdlib-only Python 3.13/3.14 as well as the canonical runner."""
from concurrent.futures.thread import _threads_queues
from contextvars import ContextVar
import threading

from tools.daemon_pool import DaemonThreadPoolExecutor


def test_initializer_and_per_submission_context_survive_worker_reuse():
    profile = ContextVar("profile", default="unset")
    local = threading.local()
    initialized = []
    def initialize(value):
        local.value = value
        initialized.append(threading.get_ident())
    def work():
        seen = profile.get()
        profile.set("worker-local-mutation")
        thread = threading.current_thread()
        return seen, local.value, thread.daemon, thread not in _threads_queues
    with DaemonThreadPoolExecutor(max_workers=1, initializer=initialize, initargs=("initialized",)) as pool:
        for value in ("profile-A", "profile-B", "profile-A"):
            token = profile.set(value)
            try:
                future = pool.submit(work)
                profile.set("after-submit")
                assert future.result(timeout=5) == (value, "initialized", True, True)
                assert profile.get() == "after-submit"
            finally:
                profile.reset(token)
        assert len(initialized) == 1


def test_worker_limit_and_queued_work_complete():
    started = threading.Barrier(3)
    release = threading.Event()
    def blocked():
        started.wait(timeout=5)
        assert release.wait(timeout=5)
        return threading.get_ident()
    pool = DaemonThreadPoolExecutor(max_workers=2)
    try:
        busy = [pool.submit(blocked) for _ in range(2)]
        started.wait(timeout=5)
        queued = pool.submit(lambda: "queued")
        assert not queued.done()
        assert len(pool._threads) == 2
        release.set()
        assert len({future.result(timeout=5) for future in busy}) == 2
        assert queued.result(timeout=5) == "queued"
    finally:
        release.set()
        pool.shutdown(wait=True)


if __name__ == "__main__":
    import json
    import platform
    test_initializer_and_per_submission_context_survive_worker_reuse()
    test_worker_limit_and_queued_work_complete()
    print(json.dumps({"python": platform.python_version(), "tests_passed": 2,
                      "context_per_submit": True, "initializer": True,
                      "daemon_unregistered": True, "max_workers": True}))
