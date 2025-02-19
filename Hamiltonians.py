import numpy as np

#1D Hamiltonian definitions
X = np.array([[0.,1.],[1.,0.]],dtype=complex)
Y = np.array([[0.,-1.j],[1.j,0.]],dtype=complex)
Z = np.array([[1.,0.],[0.,-1.]],dtype=complex)
I = np.array([[1.,0.],[0.,1.]],dtype=complex)

import numpy as np

def define1D_Hterm(nsites,site,pauli,twosites=False):
    #calculates matrix of a single Hamiltonian term with spectral norm=1
    #nsites: number of spin 1/2 sites
    #site: site index of the term Hi
    #pauli: can be X Y Z for site i
    #twosites: False=> site i, True=> site i and i+1 
    op_list = [I] * nsites
    if not twosites:
        op_list[site]= pauli
    else:
        op_list[site]= pauli
        op_list[(site+1)%nsites] = pauli

    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result

def define1D_Hamiltonian(nsites,terms,coefficients,d=2):
    #dictionary containing all hamiltonian information and initial state
    Hdict=dict()
    Hdict["nsites"]=nsites
    Hdict["local_dim"]=d
    Hdict["Hilbert_dim"]=d**nsites
    Hdict["terms"]=terms
    Hdict["coefficients"]=np.array(coefficients)
    Hdict["lambda"]=sum(coefficients)
    Hdict["probabilities"]=np.abs(Hdict["coefficients"]/Hdict["lambda"])
    Hdict["weighted_terms"]=[hi*Hi for hi,Hi in zip(coefficients,terms)]
    Hdict["Hamiltonian"]=sum(Hdict["weighted_terms"])
    
    # initial updownupdown state
    assert d==2 # if later we want to do qtrits ?
    binary_str = ''.join(['0' if i % 2 == 0 else '1' for i in range(nsites)])
    state_index = int(binary_str, 2)
    state_vector = np.zeros(2**nsites, dtype=complex)
    state_vector[state_index] = 1
    Hdict["initial_state"]=state_vector
    return Hdict

def Ising1D_Hamiltonian(nsites,J=1,h=1,periodic=False):

    if periodic: 
        p=0 
    else: 
        p=1

    Jterms= [define1D_Hterm(nsites,i,Z,twosites=True) for i in range(nsites-p)]
    hterms= [define1D_Hterm(nsites,i,X) for i in range(nsites)]
    Jcoefs= [-J]*len(Jterms)
    hcoefs= [-h]*len(hterms)
    Hdict=  define1D_Hamiltonian(nsites,Jterms+hterms,Jcoefs+hcoefs)
    Hdict["name"]= "Ising1D"

    return Hdict

