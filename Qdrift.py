import numpy as np
import scipy.linalg as la


#Qdrift evolution
def evolve_Qdrift(Hdict,total_time,nsamples,seed=1,initial_state=None):
    # if too slow, could use Taylor approx U = 1 - iHt
    if initial_state is None:
        initial_state=Hdict["initial_state"]
    lambda_Qdrift=Hdict["lambda"]
    Hterms=Hdict["terms"]
    probabilities=Hdict["probabilities"]

    np.random.seed(seed)

    dt = total_time * lambda_Qdrift / nsamples
    sampled_terms = np.random.choice(len(Hterms), size=nsamples, p=probabilities)

    evolved_state=initial_state.copy()
    for i in range(nsamples):
        evolved_state=la.expm(-1.j*dt*Hterms[sampled_terms[i]])@evolved_state

    return evolved_state

def evolve_Qdrift_trajectories(Hdict,total_time,nsamples,ntrajs,initial_state=None,idx=None):
    evolved_trajs=np.zeros((ntrajs,Hdict["hilbert_dim"]),dtype=complex)
    for traj in range(ntrajs):
       evolved_trajs[traj]=evolve_Qdrift(Hdict,total_time,nsamples,traj*nsamples*total_time,initial_state)
    return evolved_trajs,idx