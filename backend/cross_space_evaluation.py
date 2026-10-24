"""Read-only saved-export evaluation. Never equates geometry with photo fidelity."""
import argparse
import hashlib
import json
from pathlib import Path

from reconstruction.geometry_audit import audit_scene


def evaluate(manifest, root):
    rows = []
    tuning_groups = {c.get('scene_group') for c in manifest['cases'] if c.get('used_for_tuning')}
    tuning_hashes = set()
    for case in manifest['cases']:
        if case.get('used_for_tuning') and case.get('source'):
            source = root / case['source']
            if source.is_file():
                tuning_hashes.add(hashlib.sha256(source.read_bytes()).hexdigest())
    for case in manifest['cases']:
        row = dict(id=case['id'], space_type=case['space_type'],
                   structural_status='not_tested', visual_status='not_tested',
                   fresh_provider_status='not_tested', held_out_status='not_claimed')
        rows.append(row)
        if case.get('held_out'):
            if case.get('used_for_tuning') or case.get('scene_group') in tuning_groups:
                row.update(held_out_status='invalid_tuning_overlap', structural_status='blocked')
                continue
            row['held_out_status'] = 'pending_real_capture'
        required = [case.get(k) for k in ('source', 'scene_input', 'model')]
        if not all(required) or not all((root / p).is_file() for p in required if p):
            row['reason'] = 'Required source/export evidence is unavailable.'
            continue
        row['source_sha256'] = hashlib.sha256((root / case['source']).read_bytes()).hexdigest()
        if case.get('held_out'):
            if row['source_sha256'] in tuning_hashes:
                row.update(held_out_status='invalid_source_overlap', structural_status='blocked')
                continue
            row['held_out_status'] = 'eligible_saved_export' if case.get('real_capture') else 'invalid_non_real_capture'
            if not case.get('real_capture'):
                row['structural_status'] = 'blocked'
                continue
        try:
            payload = json.loads((root / case['scene_input']).read_text(encoding='utf-8'))
            row['audit'] = audit_scene(payload, root / case['model'])
            row['structural_status'] = 'pass' if not row['audit']['issue_count'] else 'fail'
        except Exception as error:
            row.update(structural_status='error', reason=f'{type(error).__name__}: {error}')
    return dict(cases=rows, scope='Saved-export structural replay, not fresh reconstruction or visual acceptance.',
                held_out_complete=any(r['held_out_status']=='eligible_saved_export' and r['structural_status']=='pass' for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = evaluate(json.loads(args.manifest.read_text(encoding='utf-8')), root)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(1 if any(r['structural_status'] in {'fail','error','blocked'} for r in result['cases']) else 0)
