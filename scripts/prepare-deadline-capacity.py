"""Read-only capacity contingency inventory. Never starts, creates or reserves."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

CLI = Path('C:/Users/jalpa/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/GoogleCloudCLI-590.0.0/google-cloud-sdk/bin/gcloud.cmd')
PROJECT = 'project-c7993406-5bfd-4d1f-ae4'


def read(command):
    allowed = {('compute', 'instances', 'describe'), ('compute', 'reservations', 'list'),
               ('compute', 'accelerator-types', 'list'), ('compute', 'regions', 'describe')}
    if tuple(command[:3]) not in allowed:
        raise ValueError('Only inventory reads are permitted')
    result = subprocess.run([str(CLI), *command, '--project=' + PROJECT, '--format=json'],
                            capture_output=True, text=True, encoding='utf-8', timeout=90)
    if result.returncode:
        raise RuntimeError('Read-only cloud inventory failed')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    vm = read(['compute', 'instances', 'describe', 'onebeyondgpu', '--zone=us-west2-c'])
    reservations = read(['compute', 'reservations', 'list'])
    accelerators = read(['compute', 'accelerator-types', 'list', '--filter=name=nvidia-tesla-t4'])
    regions = {}
    # Backups remain designs: no paid test or new VM is silently created.
    for region in ['us-west2', 'us-west1', 'us-east1', 'asia-south1']:
        record = read(['compute', 'regions', 'describe', region])
        regions[region] = [q for q in record.get('quotas', [])
                           if q['metric'] in {'CPUS', 'NVIDIA_T4_GPUS'}]
    report = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  project=PROJECT, vm={key: vm.get(key) for key in
                  ['name', 'status', 'machineType', 'guestAccelerators', 'minCpuPlatform',
                   'reservationAffinity', 'scheduling']},
                  reservations=[{key: r.get(key) for key in ['name', 'zone', 'status', 'specificReservation']}
                                for r in reservations],
                  t4_supported_zones=sorted(a['zone'].split('/')[-1] for a in accelerators
                                            if a['name'] == 'nvidia-tesla-t4'),
                  regional_quota=regions, free_capacity_verified=False,
                  new_vm_created=False, reservation_purchased=False,
                  pending_user_choices=['31 October IST start/end time', 'maximum total paid budget',
                                        'backup zone/provider and permission to create a replacement VM'],
                  notice='Supported zones and unused quota do not establish free hardware or deadline availability.')
    (args.output / 'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    draft = dict(status='local_draft_not_submitted', instance_count=1, zone='us-west2-c',
                 machine_type='n1-standard-4', gpu_type='nvidia-tesla-t4', gpu_count=1,
                 existing_vm_affinity=vm.get('reservationAffinity'),
                 start_time=None, end_time=None, total_budget=None,
                 billing='On-demand reservation charges begin at successful creation, even with VM stopped; stopping VM does not stop reservation billing.',
                 limitations='Creation may fail for the same capacity shortage. Future GPU requests require eligible configuration and account/sales review; no assumption of instant future T4 approval.',
                 not_a_second_vm=True, tax_information_unchanged=True)
    (args.output / 'reservation-draft.json').write_text(json.dumps(draft, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='read_only_inventory_complete',
                          supported_zones=len(report['t4_supported_zones']),
                          reservations=len(reservations), vm_status=vm['status'],
                          free_capacity_verified=False, reservation_purchased=False)))


if __name__ == '__main__':
    main()
