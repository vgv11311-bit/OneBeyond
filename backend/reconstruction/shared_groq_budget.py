"""Host-wide Groq request gate; CPU/export stages remain concurrent.

One shared gate is conservative across keys/models because credentials can
belong to the same organization. No credentials or images are persisted here.
"""
import json
import re
import time
import uuid
from pathlib import Path
from contextlib import contextmanager
from .process_lock import process_lock
from .run_timing import measured_call, measured_context_entry

ROOT=Path(__file__).resolve().parents[1]/'storage'/'provider_budget'


def reset_seconds(value):
    units={'ms':.001,'s':1,'m':60,'h':3600}
    return sum(float(n)*units[u] for n,u in re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h)',str(value)))


def _read():
    try: return json.loads((ROOT/'groq.json').read_text())
    except (FileNotFoundError,ValueError): return {}


def _write(state):
    path=ROOT/('groq-'+uuid.uuid4().hex+'.tmp')
    try:
        path.write_text(json.dumps(state))
        path.replace(ROOT/'groq.json')
    finally: path.unlink(missing_ok=True)


def expected_wait(estimate):
    """Read-only routing hint; the locked request gate remains authoritative."""
    state=_read(); now=time.time()
    until=state.get('cooldown_until',0)
    if state.get('reset_at',0)>now and state.get('remaining_tokens',float('inf'))<estimate:
        until=max(until,state['reset_at'])
    return max(0,until-now)


class Budget:
    def __init__(self,state): self.state=state
    def report(self,headers,usage=None,estimated_input=None):
        now=time.time()
        try:
            reset=reset_seconds(headers.get('x-ratelimit-reset-tokens',''))
            remaining=float(headers.get('x-ratelimit-remaining-tokens'))
            if reset>0:
                self.state.update(remaining_tokens=remaining,reset_at=now+reset)
        except (TypeError,ValueError): pass
        try:
            limit=float(headers.get('x-ratelimit-limit-tokens'))
            if limit>0: self.state['limit_tokens']=limit
        except (TypeError,ValueError): pass
        if usage and estimated_input:
            actual=usage.get('prompt_tokens',0)
            if isinstance(actual,(int,float)) and actual>0:
                # Calibration is a scheduling estimate only. Provider 429 and
                # Retry-After remain authoritative; never weaken evidence.
                ratio=actual/estimated_input
                self.state['input_ratio']=max(.5,min(2,ratio*1.2))
        _write(self.state)
    def defer(self,seconds):
        self.state['cooldown_until']=max(self.state.get('cooldown_until',0),time.time()+seconds)
        _write(self.state)


@contextmanager
def request_gate(estimate,remaining_budget,status,output_tokens=0):
    with measured_context_entry('provider_lock_wait', process_lock(ROOT/'groq.lock',timeout=360,waiting=lambda:remaining_budget(1))):
        state=_read()
        now=time.time()
        if state.get('reset_at',0)<=now:
            state.pop('remaining_tokens',None)
        if output_tokens:
            estimate=(estimate-output_tokens)*state.get('input_ratio',1)+output_tokens
        until=state.get('cooldown_until',0)
        if state.get('remaining_tokens',float('inf'))<estimate:
            until=max(until,state.get('reset_at',0))
        pause=until-now
        if pause>0:
            available=remaining_budget(pause+.1)
            if available+1e-6<pause+.1:
                from .provider_status import ProviderUnavailableError
                raise ProviderUnavailableError('The provider wait exceeds the remaining photo-analysis budget. Verified stages and the upload are saved for retry.')
            status(f'Shared Groq budget: waiting {round(pause)} seconds for the provider window; other reconstruction stages remain independent...')
            measured_call('quota_wait', time.sleep, pause+.1)
            state.pop('remaining_tokens',None)
            state.pop('reset_at',None)
            state.pop('cooldown_until',None)
        gate=Budget(state)
        try:
            yield gate
        except Exception as error:
            if getattr(error,'code',None)==429:
                # Parse the provider body once. An impossible output reservation
                # is not a consumed minute window: imposing 60s here caused a
                # pointless wait even after the caller corrected the reservation.
                from .groq_scene_engine import _short_rate_cooldown
                delay=_short_rate_cooldown(error)
                gate.report(getattr(error,'headers',{}) or {})
                if delay:
                    gate.defer(delay)
            raise
