"""Per-attempt wall time and measured spans, without prompts or credentials.

Span totals may overlap (e.g. depth inference alongside provider requests), so
they are never summed or subtracted to claim an exclusive compute breakdown.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import threading
import time

_current = ContextVar('reconstruction_timing', default=None)


class RunTiming:
    def __init__(self):
        self.started = time.monotonic()
        self.spans = []
        self.lock = threading.Lock()

    def add(self, category, started, ended, ok):
        with self.lock:
            self.spans.append(dict(category=category, start_sec=round(started-self.started, 6),
                                   seconds=round(max(0, ended-started), 6), success=ok))

    def summary(self, success):
        with self.lock:
            spans = list(self.spans)
        totals = {}
        for span in spans:
            totals[span['category']] = round(totals.get(span['category'], 0) + span['seconds'], 6)
        elapsed = round(time.monotonic()-self.started, 6)
        return dict(wall_seconds=elapsed, success=success, target_seconds=300,
                    within_target=elapsed <= 300, measured_span_seconds=totals, spans=spans,
                    notice='Measured spans can overlap. Provider spans include transport/read time, '
                           'not pure model inference. Uninstrumented work is not estimated.')


@contextmanager
def measure(category):
    tracker = _current.get()
    start = time.monotonic()
    ok = False
    try:
        yield
        ok = True
    finally:
        if tracker:
            tracker.add(category, start, time.monotonic(), ok)


def measured_call(category, function, *args, **kwargs):
    with measure(category):
        return function(*args, **kwargs)


@contextmanager
def measured_context_entry(category, manager):
    with ExitStack() as stack:
        measured_call(category, stack.enter_context, manager)
        yield


def profile_pipeline(function):
    @wraps(function)
    def wrapped(self, job_id, input_type, file_paths, output_dir, *args, **kwargs):
        tracker = RunTiming()
        token = _current.set(tracker)
        success = False
        try:
            result = function(self, job_id, input_type, file_paths, output_dir, *args, **kwargs)
            success = result.success
            result.metrics.extra_details['run_timing'] = tracker.summary(success)
            return result
        finally:
            summary = tracker.summary(success)
            _current.reset(token)
            # Diagnostic persistence must never hide the reconstruction result.
            try:
                Path(output_dir, 'run_timing.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            except OSError:
                pass
    return wrapped
