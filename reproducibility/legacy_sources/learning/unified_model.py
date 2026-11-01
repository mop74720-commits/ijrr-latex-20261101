"""Single model definition: SI clamped-free Hermite beam, full magnetic cosine.

The consistent-mass FE assembler is reused without edits. Static-field modes
diagonalize K + mu*B0*e_theta*e_theta.T, with phi_tip_slope positive.
Dimensionless state order is (X_1,...,X_n,Y_1,...,Y_n), tau=omega0*t.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent


def find_beam_source():
    local_assembler=HERE/"beam_assembly.py"
    if local_assembler.is_file():return local_assembler
    for parent in HERE.parents:
        path=parent/"beam_validation"/"run_beam_validation.py"
        if path.is_file():return path
    raise FileNotFoundError("Place the unchanged beam_validation folder beside the unified model package.")


BEAM_SOURCE=find_beam_source()
_spec=importlib.util.spec_from_file_location("consistent_mass_beam",BEAM_SOURCE)
BEAM=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(BEAM)
PARAMETERS=dict(BEAM.P)


class MagneticBeam:
    def __init__(self, elements=32, modes=2, matrices=None):
        self.elements=elements;self.modes=modes
        if matrices is None:
            self.M,self.K,_,_=BEAM.assemble(elements)
            self.matrix_source="Python consistent-mass Hermite FE"
        else:
            saved=np.load(matrices)
            self.M=np.asarray(saved["M"],dtype=float)
            self.K=np.asarray(saved["K"],dtype=float)
            self.matrix_source=str(Path(matrices).resolve())
        self.energy_scale=PARAMETERS["dipole_A_m2"]*PARAMETERS["bias_field_T"]
        self.Kbias=self.K.copy();self.Kbias[-1,-1]+=self.energy_scale
        vals,phi=eigh(self.Kbias,self.M)
        if np.any(vals<=0):raise ValueError("Static magnetic stiffness is not positive definite")
        phi*=np.where(phi[-1]>=0.,1.,-1.)
        self.all_omega=np.sqrt(vals);self.all_phi=phi
        self.omega=self.all_omega[:modes].copy();self.phi=phi[:,:modes].copy()
        self.omega0=float(self.all_omega[0]);self.nu=self.omega/self.omega0
        self.modal_scale=np.sqrt(self.energy_scale/(self.omega0*self.omega))
        self.shape=self.phi*self.modal_scale[None,:]
        self.d=self.shape[-1].copy();self.tip_map_m=self.shape[-2].copy()
        self.gamma=PARAMETERS["drag_mass_factor_s_inv"]/self.omega0
        self.drive_nu=PARAMETERS["drive_angular_frequency_rad_s"]/self.omega0

    def energy(self,state):
        state=np.asarray(state);x=state[...,:self.modes];y=state[...,self.modes:]
        theta=x@self.d
        correction=2*np.sin(theta/2)**2-theta**2/2
        return .5*np.sum(self.nu*(x*x+y*y),axis=-1)+correction

    def gradient(self,state):
        state=np.asarray(state);x=state[...,:self.modes];y=state[...,self.modes:]
        theta=x@self.d
        return np.concatenate([self.nu*x+(np.sin(theta)-theta)[...,None]*self.d,self.nu*y],axis=-1)

    def input(self,time_s,amplitude_T=None,frequency_ratio=None,phase=0.,ramp_time_s=None):
        amp=PARAMETERS["transverse_field_amplitude_T"] if amplitude_T is None else amplitude_T
        frequency=PARAMETERS["drive_angular_frequency_rad_s"] if frequency_ratio is None else frequency_ratio*self.omega0
        ramp_time=PARAMETERS["ramp_time_s"] if ramp_time_s is None else ramp_time_s
        tt=np.asarray(time_s);ramp=1-np.exp(-(tt/ramp_time)**2)
        ramp_dt=2*tt/ramp_time**2*np.exp(-(tt/ramp_time)**2)
        by=amp*ramp*np.sin(frequency*tt+phase)
        by_dt=amp*(ramp_dt*np.sin(frequency*tt+phase)+ramp*frequency*np.cos(frequency*tt+phase))
        return by/PARAMETERS["bias_field_T"],by_dt/PARAMETERS["bias_field_T"]

    def field(self,tau,state,damping=0.,drive=None):
        grad=self.gradient(state);n=self.modes
        dx=grad[n:];dy=-grad[:n]-damping/self.nu*grad[n:]
        if drive is not None:
            u,_=self.input(tau/self.omega0,**drive)
            dy+=self.d*u*np.cos(self.d@state[:n])
        return np.r_[dx,dy]

    def observables(self,state,gradient=None):
        state=np.asarray(state);x=state[...,:self.modes]
        result={"tip_m":x@self.tip_map_m,"tip_slope_rad":x@self.d,
                "energy_J":self.energy(state)*self.energy_scale}
        if gradient is None:gradient=self.gradient(state)
        result["tip_velocity_m_s"]=gradient[...,self.modes:]@self.tip_map_m*self.omega0
        return result

    def simulate(self,x0,time_s,damping=0.,drive=None,rtol=2e-10,atol=2e-12,method=None):
        tau=np.asarray(time_s)*self.omega0
        method=("Radau" if self.modes>=8 else "DOP853") if method is None else method
        def jac(t,state):
            n=self.modes;theta=self.d@state[:n]
            result=np.zeros((2*n,2*n));result[:n,n:]=np.diag(self.nu)
            factor=np.cos(theta)-1
            if drive is not None:
                u,_=self.input(t/self.omega0,**drive);factor+=u*np.sin(theta)
            result[n:,:n]=-np.diag(self.nu)-factor*np.outer(self.d,self.d)
            result[n:,n:]=-damping*np.eye(n)
            return result
        options={"jac":jac} if method=="Radau" else {}
        sol=solve_ivp(lambda t,x:self.field(t,x,damping,drive),(tau[0],tau[-1]),x0,
                      t_eval=tau,method=method,rtol=rtol,atol=atol,**options)
        if not sol.success:raise RuntimeError(sol.message)
        return sol.y.T

    def export(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"model_matrices.npz",M=self.M,K=self.K,Kbias=self.Kbias,
                            all_phi=self.all_phi,all_omega=self.all_omega,phi=self.phi,omega=self.omega,
                            nu=self.nu,modal_scale=self.modal_scale,shape_map=self.shape,d=self.d,
                            tip_map_m=self.tip_map_m,energy_scale_J=self.energy_scale,omega0_rad_s=self.omega0)
        spec={"model":"Clamped-free small-slope Euler-Bernoulli Hermite beam with full tip magnetic cosine",
              "parameters_SI":PARAMETERS,"elements":self.elements,"modes":self.modes,
              "matrix_source":self.matrix_source,"source_path":str(BEAM_SOURCE),
              "source_sha256":hashlib.sha256(BEAM_SOURCE.read_bytes()).hexdigest(),
              "state_order":[f"X{j+1}" for j in range(self.modes)]+[f"Y{j+1}" for j in range(self.modes)],
              "omega_rad_s":self.omega.tolist(),"omega0_rad_s":self.omega0,"nu":self.nu.tolist(),
              "energy_scale_J":self.energy_scale,"modal_scale":self.modal_scale.tolist(),
              "tip_map_m":self.tip_map_m.tolist(),"slope_map_d":self.d.tolist(),"dimensionless_drag_gamma":self.gamma,
              "time_definition":"tau=omega0*t_SI",
              "canonical_scaling":"a_j=sqrt(E0/(omega0*omega_j))*X_j; p_j=sqrt(E0*omega_j/omega0)*Y_j",
              "hamiltonian":"h=0.5 sum nu_j*(X_j^2+Y_j^2)+1-cos(d.X)-0.5*(d.X)^2",
              "input_potential":"-u(tau)*sin(d.X), u=B_y/B0",
              "port_decoder":"X'=h_Y; Y'=-h_X-gamma*diag(1/nu)*h_Y+u*d*cos(d.X)",
              "scope":["Same SI material parameters and consistent mass as original beam validation",
                        "Static bias field is included in the eigenvectors and eigenfrequencies",
                        "No axial restraint or quartic stretch energy is introduced",
                        "Small-slope mechanical approximation; full cosine of approximate tip slope retained",
                        "No physical, clinical, or spaceflight calibration is claimed"]}
        (folder/"model_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
        return spec


if __name__=="__main__":
    print(json.dumps(MagneticBeam().export(HERE/"raw"),indent=2))
