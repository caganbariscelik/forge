import queue
import threading
from typing import Any


class LogBroker:
    """In-process pub/sub for streaming a run's logs to any number of SSE clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[queue.Queue]] = {}

    def subscribe(self, run_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if q in subs:
                subs.remove(q)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(run_id, []))
        for q in subs:
            q.put(event)


broker = LogBroker()
