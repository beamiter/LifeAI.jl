#!/usr/bin/env python3
"""Submit and inspect an immutable one-node, eight-GPU Ripple inference service.

Run on DSW with /isaac-sim/python.sh for the Alibaba SDK and credential proxy.
All credential values stay in the SDK; requests contain only an API-key path.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def client(config):
    from alibabacloud_credentials.client import Client as CredentialsClient
    from alibabacloud_credentials.models import Config as CredentialsConfig
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    credentials = CredentialsClient(CredentialsConfig(type='credentials_uri', credentials_uri=config['credentials_uri']))
    return Client(Config(credential=credentials, region_id=config['region'], endpoint=config['endpoint']))


def snapshot(config, name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,59}', name):
        raise ValueError('name must contain only letters, digits, underscores or hyphens')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    if dirty:
        raise RuntimeError('Commit the deployment code before creating a DLC snapshot')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path(config['deployments_root']) / f'{name}-{stamp}'
    root.mkdir(parents=True, exist_ok=False)
    code = root / 'code';code.mkdir()
    archive = subprocess.check_output(['git', 'archive', '--format=tar', revision], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tar:
        for member in tar.getmembers():
            target = (code / member.name).resolve()
            if not target.is_relative_to(code.resolve()):
                raise RuntimeError('Archive path escapes snapshot')
        tar.extractall(code)
    (code / 'CODE_REVISION').write_text(revision + '\n')
    checks = []
    for file in sorted(code.rglob('*')):
        if file.is_file():
            checks.append(hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + str(file.relative_to(code)))
    (code / 'CODE_SHA256SUMS').write_text('\n'.join(checks) + '\n')
    credentials = root / 'credentials';credentials.mkdir(mode=0o700)
    key = credentials / 'api_key';key.write_text(secrets.token_urlsafe(32) + '\n');key.chmod(0o600)
    (root / 'results').mkdir()
    return root, revision


def build_request(config, root, revision):
    code = root / 'code'
    env = {
        'TZ': 'Asia/Shanghai', 'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'offline',
        'JULIA_NUM_THREADS': '16', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
        'JULIA_DEPOT_PATH': config['julia_depot'], 'JULIA_CUDA_SOFT_MEMORY_LIMIT': '80%',
        'NVIDIA_TF32_OVERRIDE': '0', 'NCCL_DEBUG': 'INFO', 'NCCL_ALGO': 'Ring',
        'NCCL_IB_DISABLE': '0', 'NCCL_IB_HCA': 'erdma', 'NCCL_NET_GDR_LEVEL': 'SYS', 'NCCL_P2P_LEVEL': 'SYS',
        'LIFEAI_CODE_REVISION': revision, 'LIFEAI_API_KEY_FILE': str(root / 'credentials/api_key'),
        'LIFEAI_MODEL_DIR': config['model'], 'LIFEAI_REFERENCE_DIR': config['reference'],
        'LIFEAI_RESULT_DIR': str(root / 'results'), 'LIFEAI_JULIA': config['julia'],
        'LIFEAI_CODE_ROOT': str(code), 'LIFEAI_SERVICE_PORT': str(config['port'])
    }
    return {
        'DisplayName': 'lifeai-' + root.name,
        'Description': f'Ripple-65B original BF16, tensor parallel 8; code {revision}; persistent private inference service',
        'JobType': 'PyTorchJob', 'WorkspaceId': config['workspace_id'], 'ResourceId': config['resource_quota_id'],
        'Priority': config['priority'], 'SchedulingStrategy': 'Binpack',
        'UserVpc': config['vpc'], 'DataSources': config['data_sources'], 'Envs': env,
        'JobSpecs': [{'Type': 'Worker', 'PodCount': 1, 'Image': config['image'],
                      'ResourceConfig': {'CPU': config['cpu'], 'GPU': '8', 'Memory': config['memory'], 'SharedMemory': config['shared_memory']},
                      'RestartPolicy': 'Never'}],
        'UserCommand': 'exec bash ' + shlex.quote(str(code / 'scripts/run_ripple_tp8_dlc.sh'))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/deployment/dlc_ripple_tp8.json')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create');create.add_argument('--name', required=True);create.add_argument('--dry-run', action='store_true')
    status = sub.add_parser('status');status.add_argument('job_id')
    stop = sub.add_parser('stop');stop.add_argument('job_id')
    args = parser.parse_args();config = json.loads(args.config.read_text())
    if args.command == 'create':
        root, revision = snapshot(config, args.name)
        request = build_request(config, root, revision)
        write_json(root / 'request.json', request)
        print(json.dumps({'run_dir': str(root), 'code_revision': revision, 'request': request}, ensure_ascii=False))
        if args.dry_run:
            print('Dry run: no DLC job created');return
        from alibabacloud_pai_dlc20201203.models import CreateJobRequest
        response = client(config).create_job(CreateJobRequest().from_map(request))
        job_id = response.body.job_id
        receipt = {'job_id': job_id, 'run_dir': str(root), 'code_revision': revision,
                   'console_url': f'https://pai.console.aliyun.com/?regionId={config["region"]}&workspaceId={config["workspace_id"]}#/dlc2/jobs/{job_id}'}
        write_json(root / 'job.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False))
    elif args.command == 'status':
        from alibabacloud_pai_dlc20201203.models import GetJobRequest
        job = client(config).get_job(args.job_id, GetJobRequest()).body.to_map()
        result = {key: job.get(key) for key in ['JobId', 'DisplayName', 'Status', 'ReasonCode', 'ReasonMessage']}
        result['Pods'] = [{key: pod.get(key) for key in ['PodId', 'Status', 'IP', 'PodIP', 'HostIP', 'NodeName']} for pod in job.get('Pods', [])]
        print(json.dumps(result, ensure_ascii=False))
    else:
        connection = client(config)
        import inspect
        if len(inspect.signature(connection.stop_job).parameters) >= 2:
            from alibabacloud_pai_dlc20201203.models import StopJobRequest
            connection.stop_job(args.job_id, StopJobRequest())
        else:
            connection.stop_job(args.job_id)
        print(json.dumps({'job_id': args.job_id, 'stop_requested': True}))


if __name__ == '__main__':
    main()
