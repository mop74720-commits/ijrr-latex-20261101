"""Prospective confirmation of at most one validation-nominated setting.

Only writes within this exploration. Never calls production main functions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
sys.dont_write_bytecode = True
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / 'confirmation'
SEEDS = [307, 419, 541, 653, 809]

def module(name):
    spec = importlib.util.spec_from_file_location('confirm_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result

signal = module('signal_strength')
sample = module('sample_budget')
save = signal.save_json
sha = signal.sha256

def nominate():
    path = OUT / 'frozen_nomination.json'
    if path.exists():
        frozen = json.loads(path.read_text(encoding='utf-8'))
        for rel, digest in frozen['pilot_evidence_sha256'].items():
            if sha(HERE / rel) != digest:
                raise RuntimeError('Frozen pilot evidence changed: ' + rel)
        if frozen['confirmation_seeds'] != SEEDS:
            raise RuntimeError('Confirmation seeds changed')
        for rel, digest in frozen['execution_inputs_sha256'].items():
            if sha(HERE / rel) != digest:
                raise RuntimeError('Frozen execution input changed: ' + rel)
        return frozen
    candidates = []
    evidence = {}
    for name, count, updates, saved in sample.CONDITIONS:
        ratios = []
        for seed in sample.SEEDS:
            rows = []
            for group in ('GR', 'GC'):
                p = HERE / 'sample_budget' / name / f'seed_{seed}' / f'{group}_record.json'
                row = json.loads(p.read_text(encoding='utf-8'))
                if row['status'] != 'completed':
                    raise RuntimeError(f'Incomplete pilot {p}')
                evidence[str(p.relative_to(HERE))] = sha(p)
                rows.append(row['common_scale_clean_gradient_rms'])
            ratios.append(rows[1] / rows[0])
        candidates.append(dict(name=name, train_trajectories=count, updates=updates,
                               amplitude_multiplier=1., noise_bound=1e-6, gc_over_gr=ratios))
    for noise in signal.NOISES:
        ratios = []
        for seed in signal.SEEDS:
            rows = []
            for group in ('GR', 'GC'):
                p = HERE / 'signal_strength' / signal.noise_key(noise) / f'seed_{seed}' / f'{group}_training.json'
                row = json.loads(p.read_text(encoding='utf-8'))
                if row['status'] != 'completed':
                    raise RuntimeError(f'Incomplete pilot {p}')
                evidence[str(p.relative_to(HERE))] = sha(p)
                rows.append(row['clean_full_validation_scaled_gradient_rms'])
            ratios.append(rows[1] / rows[0])
        candidates.append(dict(name='amplitude2_' + signal.noise_key(noise), train_trajectories=12,
                               updates=10000, amplitude_multiplier=2., noise_bound=noise, gc_over_gr=ratios))
    for row in candidates:
        row['mean_log_gr_over_gc'] = float(-np.mean(np.log(row['gc_over_gr'])))
        row['gc_wins'] = sum(x < 1 for x in row['gc_over_gr'])
        row['eligible'] = row['mean_log_gr_over_gc'] > 0 and row['gc_wins'] >= 2
    eligible = [r for r in candidates if r['eligible']]
    chosen = max(eligible, key=lambda r: r['mean_log_gr_over_gc']) if eligible else None
    frozen = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), chosen=chosen, candidates=candidates,
                  pilot_evidence_sha256=evidence, confirmation_seeds=SEEDS,
                  execution_inputs_sha256={p: sha(HERE / p) for p in ('confirmation.py', 'signal_strength.py', 'sample_budget.py', 'protocol.json', 'confirmation_numerical_amendment.json')},
                  primary='h0/64 continuous tip RMSE and full-state relative L2, each seed averages8trajectories',
                  numerical_check='Repeat h0/32; report direction changes and refinement shifts',
                  rule='Mean paired improvement>=5% and GC lower in>=4/5seeds on BOTH primary outcomes',
                  no_test_based_selection=True, protected=signal.protected_check())
    save(path, frozen)
    print('NOMINATED', json.dumps(chosen), flush=True)
    return frozen

def data(seed, setting):
    return signal.generate_dataset(seed, setting['amplitude_multiplier'], setting['noise_bound'], OUT / 'data')[0]

def prepare_original():
    # Safe prospective preparation only: no fitting or test evaluation, independent of selection.
    for seed in SEEDS:
        signal.generate_dataset(seed, 1., 1e-6, OUT / 'prospective_amplitude1_data')

def train():
    frozen = nominate()
    setting = frozen['chosen']
    if setting is None:
        return
    _, trainer, _ = signal.modules()
    trainer.torch.set_num_threads(1)
    for seed in SEEDS:
        if setting['amplitude_multiplier'] == 1. and setting['noise_bound'] == 1e-6:
            full, diag = signal.generate_dataset(seed, 1., 1e-6, OUT / 'prospective_amplitude1_data')
            dataset_path = OUT / 'prospective_amplitude1_data' / f'seed_{seed}.npz'
        else:
            full = data(seed, setting)
            dataset_path = OUT / 'data' / f'seed_{seed}.npz'
        order, chosen, indices = sample.selection(seed, full, setting['train_trajectories'])
        arrays = sample.training_arrays(full, indices)
        dest = OUT / f'seed_{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        for group in ('GR', 'GC'):
            path = dest / f'{group}_training.json'
            if path.exists():
                old = json.loads(path.read_text(encoding='utf-8'))
                if old['dataset_sha256'] != sha(dataset_path) or old['nomination_sha256'] != sha(OUT / 'frozen_nomination.json'):
                    raise RuntimeError('Existing confirmation provenance changed')
                if old['weights_sha256'] != sha(dest / f'{group}_weights.npz'):
                    raise RuntimeError('Existing confirmation weights changed')
                continue
            print('CONFIRM_FIT_START', seed, group, setting['name'], flush=True)
            weights, row, history = trainer.train_one(seed, group, arrays, setting['updates'], 512)
            np.savez_compressed(dest / f'{group}_weights.npz', **weights)
            row.update(sample.field_metrics(weights, full, trainer, np.maximum(np.sqrt(np.mean(full['train_labels']**2, axis=0)), 1e-6)))
            row.update(status='completed', seed=seed, group=group, setting=setting, train_trajectory_ids=chosen.tolist(),
                       dataset_path=str(dataset_path.relative_to(HERE)), dataset_sha256=sha(dataset_path),
                       nomination_sha256=sha(OUT / 'frozen_nomination.json'), weights_sha256=sha(dest / f'{group}_weights.npz'))
            save(dest / f'{group}_history.json', history)
            save(path, row)
            print('CONFIRM_FIT_DONE', seed, group, row['training_seconds'], row['clean_field_relative_l2'], flush=True)

def evaluate_seed(seed, divisor):
    nominate()
    _, trainer, refine = signal.modules()
    dest = OUT / f'seed_{seed}'
    training = [json.loads((dest / f'{g}_training.json').read_text(encoding='utf-8')) for g in ('GR', 'GC')]
    p = HERE / training[0]['dataset_path']
    if sha(p) != training[0]['dataset_sha256'] or training[1]['dataset_sha256'] != training[0]['dataset_sha256']:
        raise RuntimeError('Dataset provenance mismatch')
    arrays = signal.load_npz(p)
    times = arrays['test_times']
    h = float(arrays['teacher_step_tau']) / divisor
    stride = int(arrays['teacher_sample_stride']) * divisor
    nsteps = (len(times) - 1) * stride
    nu, d = arrays['nu'], arrays['d']
    cayley, inverse = refine.cayley_preconditioner(h, nu)
    ref = arrays['test_reference_states'].transpose(1, 0, 2)
    teacher, res, maxit, meanit = refine.teacher_rollout(arrays['test_initials'], h, nsteps, stride, cayley, inverse, nu, d)
    angles = (np.arange(len(times)) * stride)[:, None] * (2 * np.arctan(h * nu / 2))[None, :]
    h0 = refine.linear_states(arrays['test_initials'], angles)
    save(dest / f'baselines_h{divisor}.json', dict(seed=seed, divisor=divisor,
        teacher_residual=float(res), H0=signal.trajectory_metrics(h0, ref, teacher, arrays['tip_map_m']),
        teacher=signal.trajectory_metrics(teacher, ref, teacher, arrays['tip_map_m'])))
    for group, row in zip(('GR', 'GC'), training):
        path = dest / f'{group}_h{divisor}.json'
        wp = dest / f'{group}_weights.npz'
        if row['weights_sha256'] != sha(wp):
            raise RuntimeError('Weight provenance mismatch')
        if row['nomination_sha256'] != sha(OUT / 'frozen_nomination.json'):
            raise RuntimeError('Training nomination changed')
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if (old['weights_sha256'] != sha(wp) or old['dataset_sha256'] != sha(p)
                    or old['nomination_sha256'] != sha(OUT / 'frozen_nomination.json')
                    or old['trajectory_sha256'] != sha(dest / f'{group}_h{divisor}.npz')):
                raise RuntimeError('Cached evaluation provenance changed')
            continue
        weights = signal.load_npz(wp)
        params = tuple(np.ascontiguousarray(weights[k]) for k in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'scale')) + (float(weights['energy_scale']), np.ascontiguousarray(weights['nu']), bool(weights['prior']))
        points = arrays['val_midpoints'][:128]
        gap = float(np.max(np.abs(np.array([refine.neural_field(x, *params)[1] for x in points]) - trainer.numpy_energy_field(points, weights)[1])))
        if gap > 2e-12:
            raise RuntimeError('Field export mismatch')
        started = time.perf_counter()
        pred, residual, maxiter, meaniter = refine.neural_rollout(arrays['test_initials'], h, nsteps, stride, cayley, inverse, params)
        trajectory = dest / f'{group}_h{divisor}.npz'
        np.savez_compressed(trajectory, times_seconds=times, predicted_states=pred, continuous_reference=ref, H0=h0, teacher=teacher)
        result = dict(seed=seed, group=group, divisor=divisor, status='completed', h_tau=h, internal_steps=nsteps,
                      elapsed_seconds=time.perf_counter()-started, field_export_gap=gap,
                      maximum_implicit_residual=float(residual), max_iterations=int(maxiter),
                      weights_sha256=sha(wp), trajectory_sha256=sha(trajectory),
                      dataset_sha256=sha(p), nomination_sha256=sha(OUT / 'frozen_nomination.json'),
                      metrics=signal.trajectory_metrics(pred, ref, teacher, arrays['tip_map_m']))
        save(path, result)
        print('CONFIRM_ROLLOUT', seed, group, divisor, json.dumps(result['metrics']['continuous']), flush=True)

def evaluate(divisor):
    # Two independent paired seeds, compiled routines release the GIL.
    signal.modules()
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda seed: evaluate_seed(seed, divisor), SEEDS))

def summarize():
    rows = []
    for divisor in (32, 64):
        for key in ('tip_rmse_mm_mean', 'state_relative_l2_mean'):
            pairs, baseline = [], []
            for seed in SEEDS:
                dest = OUT / f'seed_{seed}'
                paths = [dest / f'{g}_h{divisor}.json' for g in ('GR', 'GC')]
                if not all(p.exists() for p in paths):
                    break
                values = [json.loads(p.read_text(encoding='utf-8'))['metrics']['continuous'][key] for p in paths]
                pairs.append(values)
                baseline.append(json.loads((dest / f'baselines_h{divisor}.json').read_text(encoding='utf-8'))['H0']['continuous'][key])
            if len(pairs) != 5:
                continue
            values = np.array(pairs)
            improvement = 1 - values[:, 1] / values[:, 0]
            rows.append(dict(divisor=divisor, metric=key, GR_mean=float(values[:,0].mean()), GC_mean=float(values[:,1].mean()),
                             GR_sample_sd=float(values[:,0].std(ddof=1)), GC_sample_sd=float(values[:,1].std(ddof=1)),
                             H0_mean=float(np.mean(baseline)), H0_sample_sd=float(np.std(baseline,ddof=1)),
                             paired_improvement_sample_sd=float(improvement.std(ddof=1)),
                             GC_wins=int(np.sum(improvement>0)), paired_relative_improvements=improvement.tolist(),
                             mean_paired_relative_improvement=float(improvement.mean()), pairs=values.tolist(),
                             pragmatic_rule_passed=bool(improvement.mean()>=.05 and np.sum(improvement>0)>=4)))
    primary = [row for row in rows if row['divisor']==64]
    combined = all(row['pragmatic_rule_passed'] for row in primary) if len(primary)==2 else None
    save(OUT / 'summary.json', dict(results=rows, repetition_unit='seed',
         both_primary_outcomes_passed=combined, no_formal_significance_claim=True, protected=signal.protected_check()))
    print(json.dumps(rows, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'nominate', 'train', 'evaluate', 'summary'], required=True)
    parser.add_argument('--divisor', choices=[32,64], type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage == 'prepare': prepare_original()
        elif args.stage == 'nominate': nominate()
        elif args.stage == 'train': train()
        elif args.stage == 'evaluate': evaluate(args.divisor)
        elif args.stage == 'summary': summarize()
    save(OUT / 'protected_latest.json', signal.protected_check())
