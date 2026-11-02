"""Small deterministic gate for the QLT-04 sustained-load dispatch rule."""

import math


class SustainedLoadGate:
    def __init__(self, threshold=85.0, resume_below=70.0, samples=5, interval_seconds=60.0):
        if not (math.isfinite(threshold) and math.isfinite(resume_below)):
            raise ValueError("load thresholds must be finite")
        if not (0 <= resume_below < threshold):
            raise ValueError("resume threshold must be below load threshold")
        if type(samples) is not int or samples < 1:
            raise ValueError("sample count must be a positive integer")
        if not (math.isfinite(interval_seconds) and interval_seconds > 0):
            raise ValueError("sample interval must be positive and finite")
        self.threshold = threshold
        self.resume_below = resume_below
        self.required_samples = samples
        self.interval_seconds = interval_seconds
        self.high_samples = 0
        self.paused = False
        self.last_sample_at = None

    def observe(self, load, monotonic_now):
        if not (math.isfinite(load) and math.isfinite(monotonic_now)):
            raise ValueError("load and sample time must be finite")
        if self.last_sample_at is not None:
            elapsed = monotonic_now - self.last_sample_at
            if elapsed < self.interval_seconds:
                raise ValueError("load samples must be at least one interval apart")
            if elapsed >= 2 * self.interval_seconds:
                self.high_samples = 0
        self.last_sample_at = monotonic_now
        if self.paused:
            if load < self.resume_below:
                self.paused = False
                self.high_samples = 0
        else:
            self.high_samples = self.high_samples + 1 if load > self.threshold else 0
            if self.high_samples >= self.required_samples:
                self.paused = True
        return self.paused

    def missed(self):
        """A missing observation breaks consecutiveness; an existing pause remains."""
        self.high_samples = 0
        self.last_sample_at = None
        return self.paused
