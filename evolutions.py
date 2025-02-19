import numpy as np
import scipy.linalg as la
from scipy.linalg import expm
from distances import assert_valid_density_matrix

#Exact evolution
def evolve_exact(Hdict,total_time,initial_state=None):
    if initial_state is None:
        initial_state=Hdict["initial_state"]
    H=Hdict["Hamiltonian"]

    return la.expm(-1.j*total_time*H)@initial_state

#First order Trotter evolution
def evolve_Trotter(Hdict,total_time,nsteps,initial_state=None):
    if initial_state is None:
        initial_state=Hdict["initial_state"]
    
    dt=total_time/nsteps
    evolved_state=initial_state.copy()
    gates=[]
    for weighted_Hterm in Hdict["weighted_terms"]:
        gates.append(expm(-1j * dt * weighted_Hterm))
    
    for _ in range(nsteps):
        for gate in gates:
            evolved_state= gate @ evolved_state

    return evolved_state

#Qdrift evolution
def evolve_Qdrift(Hdict,total_time,nsamples,seed=1,initial_state=None):
    # if too slow, could use Taylor approx U = 1 - iHt
    if initial_state is None:
        initial_state=Hdict["initial_state"]
    lambda_Qdrift=Hdict["lambda"]
    Hterms=Hdict["terms"]
    probabilities=Hdict["probabilities"]

    np.random.seed(int(seed))

    dt = total_time * lambda_Qdrift / nsamples
    sampled_terms = np.random.choice(len(Hterms), size=nsamples, p=probabilities)

    evolved_state=initial_state.copy()
    for i in range(nsamples):
        evolved_state=la.expm(-1.j*dt*Hterms[sampled_terms[i]])@evolved_state

    return evolved_state

def evolve_Qdrift_fast(Hdict, total_time, nsamples, seed=1, initial_state=None):
    # If no initial state is provided, use the default one
    if initial_state is None:
        initial_state = Hdict["initial_state"]
        
    lambda_Qdrift = Hdict["lambda"]
    Hterms = Hdict["terms"]
    probabilities = Hdict["probabilities"]

    # Initialize random seed
    np.random.seed(int(seed))

    # Time step calculation
    dt = total_time * lambda_Qdrift / nsamples

    # Precompute the evolution matrices (using Taylor approximation)
    c = np.cos(dt) * np.eye(len(initial_state))
    s = np.sin(dt)
    evol_mat = [(c - 1.j * s * Hterms[jj]) for jj in range(len(Hterms))]  # U = 1 - iHt (Taylor approx.)

    # Sample Hamiltonian terms
    sampled_terms = np.random.choice(len(Hterms), size=nsamples, p=probabilities)

    evolved_state = initial_state.copy()
    for i in range(nsamples):
        evolved_state = evol_mat[sampled_terms[i]].dot(evolved_state)

    return evolved_state

def evolve_Qdrift_trajectories(Hdict,total_time,nsamples,ntrajs,initial_state=None,idx=None,fast=True):
    evolved_trajs=np.zeros((ntrajs,Hdict["Hilbert_dim"],Hdict["Hilbert_dim"]),dtype=complex)
    for traj in range(ntrajs):
        evolved_traj=evolve_Qdrift_fast(Hdict,total_time,nsamples,int(traj*nsamples*total_time),initial_state)
        evolved_trajs[traj]=np.outer(evolved_traj,evolved_traj.conj())
        assert_valid_density_matrix(evolved_trajs[traj])

    return evolved_trajs,idx


