"""Matched real/complex full-cosine Hamiltonian feature models."""
import itertools,math
import numpy as np
from scipy.linalg import null_space
from numba import njit

EXP=np.array([a for degree in [2,4] for a in itertools.product(range(degree+1),repeat=4)
              if sum(a)==degree],dtype=np.int64)
N2=10


def real_grad_features(x):
    x=np.atleast_2d(x);out=np.zeros((len(x),4,45))
    for k,a in enumerate(EXP):
        for j in range(4):
            if a[j]:
                e=a.copy();e[j]-=1
                out[:,j,k]=a[j]*np.prod(x**e,axis=1)
    return out


SPECS=[]
for a in EXP:
    if tuple(a)<tuple(a[[2,3,0,1]]):continue
    SPECS.append((a.copy(),"real"))
    if tuple(a)!=tuple(a[[2,3,0,1]]):SPECS.append((a.copy(),"imag"))


def complex_grad_features(x):
    x=np.atleast_2d(x);z=(x[:,:2]+1j*x[:,2:])/np.sqrt(2)
    u=np.c_[z,z.conj()]
    du=np.array([[1,0,1j,0],[0,1,0,1j],[1,0,-1j,0],[0,1,0,-1j]])/np.sqrt(2)
    out=np.zeros((len(x),4,45))
    for k,(a,part) in enumerate(SPECS):
        value=np.zeros((len(x),4),dtype=complex)
        for j in range(4):
            if a[j]:
                e=a.copy();e[j]-=1
                value+=a[j]*np.prod(u**e,axis=1)[:,None]*du[j]
        out[:,:,k]=value.real if part=="real" else value.imag
    return out


def exact_feature_transform():
    linear=[{(1,0,0,0):1/np.sqrt(2),(0,0,1,0):1j/np.sqrt(2)},
            {(0,1,0,0):1/np.sqrt(2),(0,0,0,1):1j/np.sqrt(2)},
            {(1,0,0,0):1/np.sqrt(2),(0,0,1,0):-1j/np.sqrt(2)},
            {(0,1,0,0):1/np.sqrt(2),(0,0,0,1):-1j/np.sqrt(2)}]
    result=np.zeros((45,45))
    for col,(a,part) in enumerate(SPECS):
        poly={(0,0,0,0):1+0j}
        for j in range(4):
            for unused in range(a[j]):
                nxt={}
                for b,value in poly.items():
                    for e,weight in linear[j].items():
                        key=tuple(np.array(b)+e);nxt[key]=nxt.get(key,0)+value*weight
                poly=nxt
        for e,value in poly.items():
            row=np.flatnonzero(np.all(EXP==e,axis=1))[0]
            result[row,col]=value.real if part=="real" else value.imag
    return result


TRANSFORM=exact_feature_transform()


def make_basis(beam):
    quartic=np.zeros(35)
    for j in range(5):
        exponent=(j,4-j,0,0);row=np.flatnonzero(np.all(EXP[N2:]==exponent,axis=1))[0]
        quartic[row]=-math.comb(4,j)*beam.d[0]**j*beam.d[1]**(4-j)/24
    norm=float(np.linalg.norm(quartic));complement=null_space(quartic[None,:])
    taylor=np.column_stack([quartic/norm,complement])
    return dict(quartic=quartic,normalizer=norm,complement=complement,taylor=taylor,
                complex_complement=np.linalg.solve(TRANSFORM[N2:,N2:],complement))


def gradients(x,beam,basis,representation):
    x=np.atleast_2d(x);theta=x[:,:2]@beam.d
    if representation=="complex":
        raw=complex_grad_features(x)
        quadratic=raw[:,:,:N2]
        quartic=raw[:,:,N2:]@basis["complex_complement"]
        sine=np.exp(1j*theta).imag
    else:
        raw=real_grad_features(x);quadratic=raw[:,:,:N2]
        quartic=raw[:,:,N2:]@basis["complement"]
        sine=np.sin(theta)
    trig=np.zeros((len(x),4,1))
    trig[:,:2,0]=(sine-theta)[:,None]*beam.d[None,:]/basis["normalizer"]
    return np.concatenate([quadratic,trig,quartic],axis=2)


def decode_weights(weights,prior,representation,beam,basis):
    offset=N2 if prior else 0
    coeff=np.zeros(45)
    if prior:
        for j in range(2):
            for s in [j,j+2]:
                a=np.zeros(4,dtype=int);a[s]=2
                coeff[np.flatnonzero(np.all(EXP==a,axis=1))[0]]=beam.nu[j]/2
    else:
        coeff[:N2]=(TRANSFORM[:N2,:N2]@weights[:N2] if representation=="complex" else weights[:N2])
    nonlinear=weights[N2-offset:]
    cosine=float(nonlinear[0]/basis["normalizer"])
    coeff[N2:]=basis["complement"]@nonlinear[1:]
    return coeff,cosine


@njit(cache=True)
def energy_gradient(x,coeff,cosine,d):
    h=0.;g=np.zeros(4)
    for k in range(45):
        if coeff[k]==0:continue
        a=EXP[k];v=coeff[k]
        for j in range(4):v*=x[j]**a[j]
        h+=v
        for j in range(4):
            if not a[j]:continue
            v=coeff[k]*a[j]
            for s in range(4):v*=x[s]**(a[s]-(s==j))
            g[j]+=v
    theta=x[0]*d[0]+x[1]*d[1]
    h+=cosine*(2*np.sin(theta/2)**2-theta*theta/2)
    v=cosine*(np.sin(theta)-theta)
    g[0]+=v*d[0];g[1]+=v*d[1]
    return h,g


@njit(cache=True)
def energy_hessian(x,coeff,cosine,d):
    """Analytic Hessian, retaining all learned momentum-dependent quartics."""
    result=np.zeros((4,4))
    for k in range(45):
        if coeff[k]==0:continue
        a=EXP[k]
        for i in range(4):
            if not a[i]:continue
            for j in range(4):
                aj=a[j]-(i==j)
                if aj<=0:continue
                value=coeff[k]*a[i]*aj
                for s in range(4):value*=x[s]**(a[s]-(s==i)-(s==j))
                result[i,j]+=value
    theta=x[0]*d[0]+x[1]*d[1]
    factor=cosine*(np.cos(theta)-1)
    for i in range(2):
        for j in range(2):result[i,j]+=factor*d[i]*d[j]
    return result


def twist(beam,basis,coeff,cosine):
    c=coeff.copy();c[N2:]+=cosine*basis["quartic"]
    def at(e):return c[np.flatnonzero(np.all(EXP==e,axis=1))[0]]
    a11=3*at((4,0,0,0))+at((2,0,2,0))+3*at((0,0,4,0))
    a22=3*at((0,4,0,0))+at((0,2,0,2))+3*at((0,0,0,4))
    a12=sum(at(e) for e in [(2,2,0,0),(2,0,0,2),(0,2,2,0),(0,0,2,2)])
    return np.array([[a11,a12],[a12,a22]])
