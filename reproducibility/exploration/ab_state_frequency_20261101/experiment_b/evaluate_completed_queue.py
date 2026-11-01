"""Evaluate completed B fits while later independent fits run.

Only scheduling differs from experiment_b.py --stage evaluate: wait until the
immutable completed-training record exists before reading a weights archive.
The frozen numerical/evaluation implementation is called unchanged.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
path=Path(__file__).resolve().parent.parent/'experiment_b.py'
spec=importlib.util.spec_from_file_location('b_completed_queue',path)
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
original_load=B.load

def wait_for_completed_weights(path):
    path=Path(path)
    if path.name.endswith('_weights.npz'):
        record=path.with_name(path.name.replace('_weights.npz','_training.json'))
        while not record.exists():
            print('WAIT_FOR_COMPLETED_FIT',path.parent.name,path.name,flush=True)
            time.sleep(10)
        completed=json.loads(record.read_text(encoding='utf-8'))
        assert completed['status']=='completed' and completed['weights_sha256']==B.sha(path)
    return original_load(path)

B.load=wait_for_completed_weights
with B.threadpool_limits(limits=1):B.evaluate()
