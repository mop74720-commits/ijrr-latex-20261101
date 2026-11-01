"""Disjoint process scheduling of the frozen B evaluator; no numerical changes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b_disjoint_evaluation',OUT.parent/'experiment_b.py')
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)
parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1,2],required=True);args=parser.parse_args()
protocol=B.plan();models=B.model_list();partition=json.loads((OUT/'evaluation_partition_plan.json').read_text(encoding='utf-8'))
assert partition['protocol_sha256']==B.sha(OUT/'protocol.json')
assert partition['numerical_implementation_sha256']==B.sha(OUT.parent/'experiment_b.py')
assignments=partition['workers'][str(args.worker)]
# Each numerical call receives the same full frozen protocol. Only its iteration
# domain is restricted to one preassigned field/seed output directory.
B.plan=lambda:protocol
original_load=B.load;original_save=B.save

def completed_load(path):
    path=Path(path)
    if path.name.endswith('_weights.npz'):
        record=path.with_name(path.name.replace('_weights.npz','_training.json'))
        while not record.exists():
            print('WAIT',args.worker,path.parent.name,path.name,flush=True);time.sleep(10)
        r=json.loads(record.read_text(encoding='utf-8'))
        assert r['status']=='completed' and r['weights_sha256']==B.sha(path)
    return original_load(path)

def isolated_save(path,value):
    path=Path(path)
    if path==OUT/'protected_after_evaluate.json':
        path=OUT/f'worker_{args.worker}_protected_after_evaluate.json'
    original_save(path,value)

B.load=completed_load;B.save=isolated_save
with B.threadpool_limits(limits=1):
    for assignment in assignments:
        model=next(x for x in models if x['name']==assignment['level'])
        B.model_list=lambda model=model:[model]
        B.SEEDS=[assignment['seed']]
        B.evaluate()
        print('PARTITION_COMPLETED',args.worker,assignment,flush=True)
