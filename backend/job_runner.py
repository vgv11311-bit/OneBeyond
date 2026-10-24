"""Private subprocess entry point; requests and credentials are not command arguments."""
import sys
import os
from app import load_job_state, run_reconstruction_worker

if __name__ == '__main__':
    job_id = sys.argv[1]
    state = load_job_state(job_id)
    if state is None:
        raise SystemExit(2)
    from app import cancellation_path
    if state.get('attempt_id'):
        os.environ['ONE_BEYOND_CANCEL_MARKER']=str(cancellation_path(job_id,state['attempt_id']).resolve())
    run_reconstruction_worker(job_id, state['engine_id'], state.get('reconstruction_options', {}))
