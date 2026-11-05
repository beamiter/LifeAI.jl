#!/usr/bin/env python3
"""Record GPU/RSS usage and allow a Julia profile signal through shared JFS."""
import argparse
import csv
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pid', type=int)
parser.add_argument('result_dir', type=Path)
args = parser.parse_args()
with (args.result_dir / 'resources.csv').open('w', buffering=1) as output:
    writer = csv.writer(output)
    writer.writerow(['timestamp_utc', 'gpu', 'memory_used_mib', 'utilization_percent', 'power_watts', 'julia_rss_kib'])
    while True:
        try:
            os.kill(args.pid, 0)
            status = Path(f'/proc/{args.pid}/status').read_text()
            rss = next((line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')), '')
        except (ProcessLookupError, FileNotFoundError):
            break
        query = args.result_dir / 'profile.request'
        if query.exists():
            query.unlink()
            os.kill(args.pid, signal.SIGUSR1)
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], text=True, capture_output=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if result.returncode == 0:
            for row in csv.reader(result.stdout.splitlines()):
                writer.writerow([stamp, *[item.strip() for item in row], rss])
        time.sleep(2)
