"""Bounded single-room service envelope; deadlines do not imply quality."""
import math

MAX_VIDEO_SECONDS=90
SOFT_SECONDS=180
HARD_SECONDS=300
SAFETY_LIMIT_SECONDS=1950


def validate_video_duration(seconds):
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 1<=seconds<=MAX_VIDEO_SECONDS:
        raise ValueError('ReSplat accepts one single-room walkthrough of 1–90 seconds. Your original upload is preserved.')


def timing_state(started,now):
    elapsed=max(0.,now-started)
    return dict(elapsed_seconds=round(elapsed,1),soft_seconds=SOFT_SECONDS,hard_seconds=HARD_SECONDS,
                remaining_seconds=max(0,round(HARD_SECONDS-elapsed,1)),
                soft_target_exceeded=elapsed>SOFT_SECONDS,deadline_exceeded=elapsed>=HARD_SECONDS,
                scope='Processing after upload; upload, queue and VM startup excluded',
                forced_abort_at_target=False,successful_room_guaranteed=False)
