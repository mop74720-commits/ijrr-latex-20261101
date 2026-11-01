"""Check rebuilt equations against the original dimensional FE equations."""
import importlib.util
import numpy as np
from build_models import RAW_SOURCE, OUT, assemble, build_model, sha, save


def main():
    before=sha(RAW_SOURCE)
    spec=importlib.util.spec_from_file_location('ab_readonly_original_beam',RAW_SOURCE)
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)  # Original numerical main is guarded.
    old_m,old_k,_,_=original.assemble(32)
    m,k=assemble(32)
    mass_gap=float(np.max(np.abs(m-old_m)))
    stiffness_gap=float(np.max(np.abs(k-old_k)))
    if mass_gap!=0 or stiffness_gap!=0:
        raise RuntimeError('FE matrices differ from original source')
    rng=np.random.default_rng(20261101)
    entries=[]
    for field in (.03,.06,.12):
        model=build_model(field)
        x=rng.normal(0,.1,(29,4))
        q=x[:,:2];a=q*model['modal_scale'];theta=q@model['d']
        b=model['phi'][-1]
        # Original dimensional modal acceleration: phi.T f = a_ddot.
        original_accel=-a@(model['phi'].T@model['elastic_stiffness']@model['phi']).T-model['energy_scale_J']*np.sin(theta)[:,None]*b
        canonical_accel=(-model['nu']*q-(np.sin(theta)-theta)[:,None]*model['d'])*(
            model['modal_scale']*model['omega']*model['omega0'])
        absolute=float(np.max(np.abs(canonical_accel-original_accel)))
        relative=float(np.linalg.norm(canonical_accel-original_accel)/np.linalg.norm(original_accel))
        fine=build_model(field,elements=64,modes=8)
        coarse8=build_model(field,elements=32,modes=8)
        frequency_change=np.abs(fine['omega']-coarse8['omega'])/fine['omega']
        if relative>1e-8:raise RuntimeError('Canonical and dimensional dynamics differ')
        entries.append(dict(field_T=field,dimensional_acceleration_absolute_gap=absolute,
            dimensional_acceleration_relative_l2_gap=relative,
            linear_frequency_relative_change_32_to_64elements=frequency_change.tolist(),
            scope='Mesh check is for linearized modal frequencies only; not nonlinear solution or physical-model validation.'))
    after=sha(RAW_SOURCE)
    if after!=before:raise RuntimeError('Original source changed during read-only check')
    result=dict(original_source_sha256=before,original_source_unchanged=True,
                mass_matrix_max_gap=mass_gap,elastic_matrix_max_gap=stiffness_gap,levels=entries)
    save(OUT/'reconstruction_checks.json',result)
    print(result)


if __name__=='__main__':
    main()
