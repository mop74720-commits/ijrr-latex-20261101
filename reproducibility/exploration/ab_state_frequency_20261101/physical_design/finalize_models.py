"""Final three-level conditional model-space pressure test.

The earlier protocol.json/design.json document a superseded pre-training design.
They are preserved; this entry point alone supplies the final experiment B levels.
"""
from pathlib import Path
import json
import numpy as np
from build_models import (ROOT, OUT, SPEC_PATH, RAW_SOURCE, SLOPE_GATE, sha, save,
                          build_model, project_initials, energy_geometry_bound,
                          save_model)

FIELDS = (.03, .06, .12)


def main():
    protocol = dict(
        status='Final experiment B protocol; supersedes pre-training protocol.json',
        fields_T=list(FIELDS), elements=32, modes=2, slope_gate=SLOPE_GATE,
        pairing='Same original full-FE initial displacement/velocity, then mass-orthogonal projection into each bias-dependent modal subspace; projection loss is reported.',
        magnetic_model='Original constant dipole and full cosine of approximate tip slope. No exact-angle substitution or changed magnetic constitutive law.',
        geometry='Elementwise Hermite-slope extrema and conservative energy bootstrap. Curvature/strain are proxies with no invented material limit.',
        physical_validity='Conditional pressure test of the declared approximate model, not calibrated proof of admissible magnetization/B range or accurate physical nonlinear coefficients.',
        baseline_numeric_anchor='Original reference_model_spec literals and original data/weights may be reused at 30mT; reconstructed model differs by at most 1.3e-14 in d.',
        source_hashes={'reference_model_spec.json': sha(SPEC_PATH),
                       'run_beam_validation.py': sha(RAW_SOURCE),
                       'build_models.py': sha(OUT/'build_models.py')})
    protocol_path = OUT/'final_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Final protocol changed after lock')
    if not protocol_path.exists():
        save(protocol_path, protocol)
    old = json.loads(SPEC_PATH.read_text(encoding='utf-8-sig'))
    baseline = build_model(.03)
    gaps = {key: float(np.max(np.abs(baseline[key]-np.asarray(old[source]))))
            for key,source in [('omega','omega_rad_s'),('nu','nu'),('d','slope_map_d'),
                               ('modal_scale','modal_scale'),('tip_map_m','tip_map_m')]}
    if max(gaps.values()) > 1e-9:
        raise RuntimeError(gaps)
    levels=[]
    for field in FIELDS:
        model=build_model(field)
        name=f'B{round(field*1000):03d}mT'
        path=OUT/(name+'.npz')
        if path.exists():
            with np.load(path) as prior:
                for key,value in model.items():
                    if isinstance(value,(np.ndarray,float,int)) and not np.array_equal(prior[key],value):
                        raise RuntimeError('Existing model has different parameters: '+name+'/'+key)
        else:
            save_model(model,path)
        initials=[];projections={}
        for seed in (17,43,91,137,211):
            with np.load(ROOT/f'results/learning/data/seed_{seed}.npz') as data:
                for split in ('train','val','test'):
                    x,diag=project_initials(data[split+'_initials'],model,baseline)
                    initials.append(x);projections[f'{seed}/{split}']=diag
        initials=np.concatenate(initials)
        screen=energy_geometry_bound(initials,model)
        if not screen['continuous_model_energy_screen_passed']:
            raise RuntimeError('Do not train: geometry gate failed '+name)
        levels.append(dict(name=name,field_T=field,omega=model['omega'].tolist(),
            omega0=model['omega0'],nu=model['nu'].tolist(),d=model['d'].tolist(),
            energy_scale_J=model['energy_scale_J'],tip_map_m=model['tip_map_m'].tolist(),
            initial_projection=projections,screen=screen,model_npz_sha256=sha(path)))
    save(OUT/'final_design.json',dict(protocol_sha256=sha(protocol_path),
        baseline_reconstruction_max_gaps=gaps,levels=levels,
        excluded_preliminary_level='45mT files are preserved design history and must not be included in experiment B.',
        physics_warning='The approximation theta=slope is geometrically inconsistent at the same fourth order as the retained magnetic residual. This is an original-model pressure test, not a validated nonlinear-rod experiment.'))
    print(json.dumps(dict(final_design_sha256=sha(OUT/'final_design.json'),
        levels=[dict(name=x['name'],bound=x['screen']['all_space_slope_energy_bound'],
            projection_q=max(d['displacement_mass_relative_max'] for d in x['initial_projection'].values()),
            projection_v=max(d['velocity_mass_relative_max'] for d in x['initial_projection'].values())) for x in levels])))


if __name__=='__main__':
    main()
