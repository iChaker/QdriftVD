import numpy as np
import scipy.linalg as la

#1D Hamiltonian definitions
X = np.array([[0.,1.],[1.,0.]],dtype=complex)
Y = np.array([[0.,-1.j],[1.j,0.]],dtype=complex)
Z = np.array([[1.,0.],[0.,-1.]],dtype=complex)
I = np.array([[1.,0.],[0.,1.]],dtype=complex)
# density matrices and distances 
def assert_valid_density_matrix(rho,eigen=False):
    # Check if the matrix is square
    assert rho.shape[0] == rho.shape[1], "Matrix is not square!"
    # Check if the matrix is Hermitian (rho = rho^dagger)
    assert np.allclose(rho, rho.conj().T), "Matrix is not Hermitian!"
    
    # Check if the trace is 1
    assert np.isclose(np.trace(rho), 1), f"Trace is not 1, it's {np.trace(rho)}!"
    
    # Check if the matrix is positive semi-definite (all eigenvalues >= 0)
    if eigen:
        eigenvalues = np.linalg.eigvals(rho)
        assert np.all(eigenvalues >= -1e-10), f"Matrix has negative eigenvalues: {eigenvalues}"
    
    return True

def partial_trace(rho, site_index, local_dim=2):
    assert_valid_density_matrix(rho) 
    # Total dimension of the Hilbert space (size of rho)
    total_dim = rho.shape[0]
    
    # Calculate the number of sites (log base local_dim of dim_total)
    num_sites = int(np.log(total_dim) / np.log(local_dim))

    # Create a list of all subsystems to trace out (all except the site_index)
    subsystems = [i for i in range(num_sites) if i != site_index]
    
    # Reshape the density matrix into a multidimensional array
    reshaped_rho = rho.reshape([local_dim] * num_sites * 2)
    
    # Perform the partial trace over the subsystems that need to be traced out
    for idx in subsystems:
        reshaped_rho = reshaped_rho.trace(axis1=idx, axis2=idx + num_sites)

    assert_valid_density_matrix(reshaped_rho) 
    return reshaped_rho

def trace_distance(rho1,rho2,**kwargs):
    rho_diff = rho1 - rho2
    return 0.5*np.sum(np.abs(np.real(la.eigvals(rho_diff))))

def reduced_trace_distance(rho1,rho2,site_index=0, local_dim=2):
    rho1=partial_trace(rho1,site_index, local_dim)
    rho2=partial_trace(rho2,site_index, local_dim)
    return trace_distance(rho1,rho2)

def spin_distance(rho1, rho2, site_index=0,local_dim=2):
    rho1=partial_trace(rho1,site_index, local_dim)
    rho2=partial_trace(rho2,site_index, local_dim)
    # Calculate expectation values for Pauli matrices for both density matrices
    s1_x = np.real(np.trace(np.dot(rho1, X)))
    s1_y = np.real(np.trace(np.dot(rho1, Y)))
    s1_z = np.real(np.trace(np.dot(rho1, Z)))
    
    s2_x = np.real(np.trace(np.dot(rho2, X)))
    s2_y = np.real(np.trace(np.dot(rho2, Y)))
    s2_z = np.real(np.trace(np.dot(rho2, Z)))
    
    # Compute the Euclidean distance (L2 norm) between the vectors of expectation values
    distance = np.linalg.norm([s1_x - s2_x, s1_y - s2_y, s1_z - s2_z])
    
    return distance


# Purification
def eigenpurify(rho):
    # diagonalize density matrix
    eva,eve = la.eig(rho)
    # find dominant eigenvalue
    max_eval_idx = np.argmax(np.real(eva))
    # get corresponding eigenvector
    max_evec = eve.T[max_eval_idx]
    # form density matrix out of it
    pure_rho = np.outer(max_evec,max_evec.conj())
    return(pure_rho)

def mix_trajectories(Qtrajs, M):
    #Qtrajs shape: ntrajs x hdim x hdim
    ntrajs=Qtrajs.shape[0]
    state_idx = np.random.choice(np.arange(ntrajs),replace=False,size=M)
    rhoM = np.average([Qtrajs[idx] for idx in state_idx],axis=0)
    return rhoM

def mix_and_compare(Qtrajs,M,rhoE,purify=False,distance=trace_distance,**kwargs):
    rhoM=mix_trajectories(Qtrajs,M)
    if purify:
        rhoM=eigenpurify(rhoM)
    return distance(rhoM,rhoE,**kwargs)