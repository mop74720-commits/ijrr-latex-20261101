"""Rebuild bias-dependent beam models and screen the full FE geometry.

Only writes into this experiment's physical_design directory.  The same
constant-dipole, small-slope model is retained; this is not material calibration.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
sys.dont_write_bytecode = True

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SPEC_PATH = ROOT/'results/learning/inputs/reference_model_spec.json'
RAW_SOURCE = Path('E:/回写/重写工作区/beam_validation/run_beam_validation.py')
FIELDS = (.03, .045, .06)
SLOPE_GATE = .15

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def assemble(elements=32):
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8-sig'))
    p = spec['parameters_SI']
    le = p['length_m']/elements
    ke = p['bending_rigidity_N_m2']/le**3*np.array([
        [12,6*le,-12,6*le],[6*le,4*le**2,-6*le,2*le**2],
        [-12,-6*le,12,-6*le],[6*le,2*le**2,-6*le,4*le**2]])
    me = p['line_mass_kg_m']*le/420*np.array([
        [156,22*le,54,-13*le],[22*le,4*le**2,13*le,-3*le**2],
        [54,13*le,156,-22*le],[-13*le,-3*le**2,-22*le,4*le**2]])
    mass = np.zeros((2*(elements+1),)*2)
    stiffness = np.zeros_like(mass)
    for j in range(elements):
        ix=np.arange(2*j,2*j+4)
        mass[np.ix_(ix,ix)] += me
        stiffness[np.ix_(ix,ix)] += ke
    return mass[2:,2:], stiffness[2:,2:]

def build_model(field_T, elements=32, modes=2):
    spec=json.loads(SPEC_PATH.read_text(encoding='utf-8-sig'))
    p=spec['parameters_SI'];mass,elastic=assemble(elements)
    tip=np.zeros(2*elements);tip[-1]=1
    E0=p['dipole_A_m2']*field_T
    linear=elastic+E0*np.outer(tip,tip)
    eigenvalues,phi=eigh(linear,mass)
    if eigenvalues.min()<=0:raise RuntimeError('Non-positive magnetic tangent stiffness')
    omega=np.sqrt(eigenvalues[:modes]);phi=phi[:,:modes]
    phi*=np.sign(phi[-1])[None,:]
    omega0=omega[0];scale=np.sqrt(E0/(omega0*omega))
    scaled=np.vstack([np.zeros((2,modes)),phi*scale])
    le=p['length_m']/elements
    # Per-element exact quadratic slope S(xi)=A xi^2+B xi+C.
    a=[];b=[];c=[]
    for j in range(elements):
        w0,t0,w1,t1=scaled[2*j:2*j+4]
        a.append(6*(w0-w1)/le+3*(t0+t1))
        b.append(6*(w1-w0)/le-4*t0-2*t1)
        c.append(t0)
    return dict(field_T=float(field_T),elements=int(elements),modes=int(modes),
        mass=mass,elastic_stiffness=elastic,linear_stiffness=linear,phi=phi,
        omega=omega,omega0=float(omega0),nu=omega/omega0,modal_scale=scale,
        energy_scale_J=float(E0),d=phi[-1]*scale,tip_map_m=phi[-2]*scale,
        slope_polynomial_a=np.array(a),slope_polynomial_b=np.array(b),
        slope_polynomial_c=np.array(c),element_length=le,
        radius_m=p['radius_m'],length_m=p['length_m'])

def project_initials(original_states, model, baseline=None):
    """Match baseline full-FE displacement/velocity before modal projection.

    Returns (new canonical states, projection diagnostics).  Projection is not
    called exact physical-state equality when the new modal subspace differs.
    """
    if baseline is None:baseline=build_model(.03)
    old=np.asarray(original_states);n=baseline['modes']
    a0=old[...,:n]*baseline['modal_scale']
    v0=old[...,n:]*baseline['modal_scale']*baseline['omega']
    overlap=model['phi'].T@model['mass']@baseline['phi']
    anew=a0@overlap.T;vnew=v0@overlap.T
    new=np.concatenate([anew/model['modal_scale'],vnew/(model['modal_scale']*model['omega'])],axis=-1)
    qold=a0@baseline['phi'].T;vold=v0@baseline['phi'].T
    qerror=qold-anew@model['phi'].T;verror=vold-vnew@model['phi'].T
    def mass_relative(error, modal):
        numerator=np.maximum(np.einsum('...i,ij,...j->...',error,model['mass'],error),0)
        return np.sqrt(numerator/np.maximum(np.sum(modal**2,axis=-1),1e-30))
    return new,dict(displacement_mass_relative_max=float(mass_relative(qerror,a0).max()),
        velocity_mass_relative_max=float(mass_relative(verror,v0).max()),
        overlap=overlap.tolist(),canonical_component_abs_max=float(np.abs(new).max()))

def energy(states,model):
    x=np.asarray(states);n=model['modes'];theta=x[...,:n]@model['d']
    return .5*np.sum(model['nu']*(x[...,:n]**2+x[...,n:]**2),axis=-1)+2*np.sin(theta/2)**2-.5*theta**2

def geometry_checks(states,model):
    """Exact spatial extrema of FE slope/curvature at supplied time states."""
    x=np.asarray(states).reshape(-1,2*model['modes'])
    maxima=dict(max_abs_slope=0.,max_abs_tip_slope=0.,max_abs_curvature_per_m=0.,
                max_surface_bending_strain_proxy=0.,max_abs_tip_displacement_m=0.)
    for first in range(0,len(x),8192):
        q=x[first:first+8192,:model['modes']]
        a=q@model['slope_polynomial_a'].T;b=q@model['slope_polynomial_b'].T;c=q@model['slope_polynomial_c'].T
        safe=np.where(np.abs(a)>1e-30,a,1.)
        vertex=-b/(2*safe);valid=(np.abs(a)>1e-30)&(vertex>0)&(vertex<1)
        at_vertex=np.where(valid,a*vertex**2+b*vertex+c,0.)
        slope=max(np.abs(c).max(),np.abs(a+b+c).max(),np.abs(at_vertex).max())
        curvature=max(np.abs(b).max(),np.abs(2*a+b).max())/model['element_length']
        maxima['max_abs_slope']=max(maxima['max_abs_slope'],float(slope))
        maxima['max_abs_tip_slope']=max(maxima['max_abs_tip_slope'],float(np.abs(q@model['d']).max()))
        maxima['max_abs_curvature_per_m']=max(maxima['max_abs_curvature_per_m'],float(curvature))
        maxima['max_abs_tip_displacement_m']=max(maxima['max_abs_tip_displacement_m'],float(np.abs(q@model['tip_map_m']).max()))
    maxima['max_surface_bending_strain_proxy']=model['radius_m']*maxima['max_abs_curvature_per_m']
    maxima['slope_screen_passed']=maxima['max_abs_slope']<=SLOPE_GATE
    maxima['scope']='Spatial extrema exact for each supplied FE time state; temporal sampling is not a global certificate.'
    return maxima

def _max_support_quartic(vectors,inverse):
    """Max g(xi)^T inverse g(xi), g=c+b xi+a xi², on each element."""
    maximum=0.
    for a,b,c in vectors:
        v=[c,b,a];coeff=np.zeros(5)
        for i in range(3):
            for j in range(3):coeff[i+j]+=v[i]@inverse@v[j]
        derivative=np.array([i*coeff[i] for i in range(1,5)])
        derivative=np.trim_zeros(derivative,'b')
        candidates=[0.,1.]
        if len(derivative)>1:
            candidates.extend(float(z.real) for z in np.roots(derivative[::-1]) if abs(z.imag)<1e-9 and 0<z.real<1)
        maximum=max(maximum,max(np.polynomial.polynomial.polyval(z,coeff) for z in candidates))
    return max(maximum,0.)

def energy_geometry_bound(initial_states,model,gate=SLOPE_GATE):
    """Continuous conservative-flow bootstrap screen in the finite FE subspace.

    Within |theta_tip|<=gate, V>=X^T[diag(nu)-gate²dd^T/12]X/2.
    If the resulting full-beam slope bound is <gate and the initial states
    pass, a continuous energy/first-exit argument confines the exact model.
    Numerical roots/floating evaluation are NOT outward interval certification.
    """
    qlower=np.diag(model['nu'])-gate**2/12*np.outer(model['d'],model['d'])
    if np.linalg.eigvalsh(qlower).min()<=0:raise RuntimeError('Energy lower stiffness not positive')
    inverse=np.linalg.inv(qlower)
    support=_max_support_quartic(zip(model['slope_polynomial_a'],model['slope_polynomial_b'],model['slope_polynomial_c']),inverse)
    emax=float(np.max(energy(initial_states,model)))
    bound=float(np.sqrt(2*emax*support))
    initial=geometry_checks(initial_states,model)
    return dict(gate=float(gate),initial_energy_max=emax,lower_stiffness_min_eigenvalue=float(np.linalg.eigvalsh(qlower).min()),
        all_space_slope_energy_bound=bound,initial_screen=initial,
        continuous_model_energy_screen_passed=bool(bound<gate and initial['slope_screen_passed']),
        scope='Conservative exact finite-dimensional model, real-arithmetic bootstrap; numerical evaluation not an interval or physical-calibration certificate.')

def save_model(model,path):
    np.savez_compressed(path,**{k:v for k,v in model.items() if isinstance(v,(np.ndarray,float,int))})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(fields_T=list(FIELDS),elements=32,modes=2,slope_gate=SLOPE_GATE,
        pairing='Same original full-FE initial displacement/velocity, then mass-orthogonal projection into each bias-dependent modal subspace; report projection loss.',
        magnetic_model='Same original constant dipole and full cosine of approximate tip slope; no new exact-angle dynamics.',
        geometry='Elementwise quadratic Hermite-slope extrema; conservative energy bootstrap before training; curvature/strain proxies reported but no fabricated material limits.',
        physical_validity='Conditional model-space sweep, not calibrated proof of admissible magnetization/B range or physical nonlinear coefficients.',
        source_hashes={'reference_model_spec.json':sha(SPEC_PATH),'run_beam_validation.py':sha(RAW_SOURCE)})
    p=OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Physical design changed')
    if not p.exists():save(p,protocol)
    baseline=build_model(.03);old=json.loads(SPEC_PATH.read_text(encoding='utf-8-sig'))
    baseline_gap={key:float(np.max(np.abs(baseline[key]-np.asarray(old[source])))) for key,source in [('omega','omega_rad_s'),('nu','nu'),('d','slope_map_d'),('modal_scale','modal_scale'),('tip_map_m','tip_map_m')]}
    if max(baseline_gap.values())>1e-9:raise RuntimeError(baseline_gap)
    levels=[]
    for field in FIELDS:
        model=build_model(field);name=f'B{round(field*1000):03d}mT';save_model(model,OUT/(name+'.npz'))
        all_initials=[];projections={}
        for seed in (17,43,91,137,211):
            with np.load(ROOT/f'results/learning/data/seed_{seed}.npz') as archive:
                for split in ('train','val','test'):
                    x,diag=project_initials(archive[split+'_initials'],model,baseline)
                    all_initials.append(x);projections[f'{seed}/{split}']=diag
        all_initials=np.concatenate(all_initials)
        screen=energy_geometry_bound(all_initials,model)
        entry=dict(name=name,field_T=field,omega=model['omega'].tolist(),nu=model['nu'].tolist(),d=model['d'].tolist(),
            energy_scale_J=model['energy_scale_J'],tip_map_m=model['tip_map_m'].tolist(),
            initial_projection=projections,screen=screen,model_npz_sha256=sha(OUT/(name+'.npz')))
        levels.append(entry)
        if not screen['continuous_model_energy_screen_passed']:raise RuntimeError('Do not train: geometry gate failed '+name)
    result=dict(baseline_reconstruction_max_gaps=baseline_gap,levels=levels,
        physics_warning='Replacing theta≈s by theta=atan(s) changes the magnetic quartic coefficient from -muB/24 to -3muB/8 in an x-graph dipole geometry. This own kinematic comparison shows omitted geometry is of the same order as the retained magnetic residual, not that an exact nonlinear rod has been validated.')
    save(OUT/'design.json',result)
    print(json.dumps({'baseline_gaps':baseline_gap,'levels':[{'name':x['name'],'slope_energy_bound':x['screen']['all_space_slope_energy_bound'],'nu':x['nu'],'d':x['d']} for x in levels]},ensure_ascii=False))

if __name__=='__main__':main()
