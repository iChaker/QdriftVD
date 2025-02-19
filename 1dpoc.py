import numpy as np
import scipy.linalg as la
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from evolutions import *
from distances import *
from Hamiltonians import *
from tqdm import tqdm
import os
from tqdm import tqdm
from threading import Lock
import pickle
from scipy.stats import bootstrap
import matplotlib.pyplot as plt


# Compute multiple evolutions

def equiv_nsamples_list(Hdict,nsteps_list):
    nterms=len(Hdict["terms"])
    return [nterms* nsteps for nsteps in nsteps_list]

def compute_exact_evolutions(Hdict,total_time_list,initial_state=None):
    exactstates=np.zeros((len(total_time_list),Hdict["Hilbert_dim"],Hdict["Hilbert_dim"]),dtype=complex)
    for total_time_idx,total_time in enumerate(total_time_list):
        exactstate = evolve_exact(Hdict,total_time,initial_state)
        exactstates[total_time_idx] = np.outer(exactstate,exactstate.conj())

    return exactstates

def compute_Trotter_evolutions(Hdict,total_time_list,nsteps_list,initial_state=None):
    # total times list x nsteps list x hdim x hdim
    Trstates=np.zeros((len(total_time_list),len(nsteps_list),Hdict["Hilbert_dim"],Hdict["Hilbert_dim"]),dtype=complex)
    for total_time_idx,total_time in enumerate(total_time_list):
        for nsteps_idx,nsteps in enumerate(nsteps_list):
            Trstate=evolve_Trotter(Hdict,total_time,nsteps,initial_state)
            Trstates[total_time_idx,nsteps_idx]=np.outer(Trstate,Trstate.conj())

    return Trstates


def compute_Qdrift_evolutions(Hdict, total_time_list, nsamples_list, ntrajs, initial_state=None, log_path="logfile.log", save_path="Qdrift_evolutions.pkl", save_progress_flag=True,save_freq=100):
    # total times list x nsamples list x trajectories x hdim x hdim
    Qstates = np.zeros((len(total_time_list), len(nsamples_list), ntrajs, Hdict["Hilbert_dim"],Hdict["Hilbert_dim"]), dtype=complex)
    
    # Lock for thread-safe saving/loading
    lock = Lock()
    
    # Helper function to verify Hdict with saved file:
    def compare_Hdict(dict1,dict2):
        return (dict1["name"]==dict2["name"] and dict1["Hilbert_dim"]==dict2["Hilbert_dim"] and np.array_equal(dict1["coefficients"],dict2["coefficients"]) and np.array_equal(dict1["initial_state"],dict2["initial_state"]))

    # Check if there is a saved file and resume
    if os.path.exists(save_path) and save_progress_flag:
        print("Resuming from saved file...")
        with lock:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
                Qstates = data['Qstates']
                processed_indices = set(data['processed_indices'])  # Convert back to set

                saved_Hdict = data['Hdict']
                saved_total_time_list = data['total_time_list']
                saved_nsamples_list = data['nsamples_list']
                saved_ntrajs = data['ntrajs']
                # Verify if parameters match
                assert compare_Hdict(Hdict,saved_Hdict), "Mismatch in Hdict"
                assert total_time_list == saved_total_time_list, "Mismatch in total_time_list"
                assert nsamples_list == saved_nsamples_list, "Mismatch in nsamples_list"
                assert ntrajs == saved_ntrajs, "Mismatch in ntrajs"
                
    else:
        processed_indices = set()

    # Helper function to save progress
    def save_progress():
        if save_progress_flag:  # Only save if the flag is True
            with lock:
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'Qstates': Qstates,
                        'processed_indices': list(processed_indices),
                        'Hdict': Hdict,
                        'total_time_list': total_time_list,
                        'nsamples_list': nsamples_list,
                        'ntrajs': ntrajs
                    }, f)
                print("Progress saved.")

    # Start logging the progress
    with open(log_path, "w") as f:
        with tqdm(total=len(total_time_list) * len(nsamples_list), file=f, desc="compute_Qdrift_evolutions") as progress_bar:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                
                for total_time_idx, total_time in enumerate(total_time_list):
                    for nsamples_idx, nsamples in enumerate(nsamples_list):
                        # Skip already processed (total_time_idx, nsamples_idx)
                        if (total_time_idx, nsamples_idx) in processed_indices:
                            with lock:
                                progress_bar.update(1)
                            continue
                        
                        # Submit the task for unprocessed indices
                        futures.append(executor.submit(
                            evolve_Qdrift_trajectories, Hdict, total_time, nsamples, ntrajs, initial_state,
                            (total_time_idx, nsamples_idx)
                        ))

                # Process the results as they finish
                for future in as_completed(futures):
                    evolved_trajs, idx = future.result()
                    Qstates[idx[0], idx[1]] = evolved_trajs
                    
                    # Update progress bar and save state periodically
                    progress_bar.update(1)
                    
                    # Update processed indices and save progress
                    processed_indices.add(idx)
                    if progress_bar.n % save_freq == 0 and save_progress_flag:  # Save progress every freq updates, only if saving is enabled
                        save_progress()
                    
            # Final save when the loop finishes
            save_progress()
    
    print("Done")
    return Qstates


def Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,seed=2,purify=False,distance=trace_distance,log_path="logfile.log",**kwargs):
    #Qdensities: times x nsteps x ntrajs x hdim x hdim
    distances=np.zeros((Qdensities.shape[:2])+(len(M_list),3)) # distances: times x steps x Ms x 3

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    with open(log_path, "w") as f:

        with tqdm(total= np.prod(distances.shape[:3]), desc="Qdrift_distance_stats",file=f) as progress_bar:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures=[]
                for idx in np.ndindex(distances.shape[:3]):
                    tmp_array= []

                    for _ in range(num_mix):
                        futures.append(executor.submit(mix_and_compare,Qdensities[idx[:2]],M_list[idx[2]],exactdensities[idx[0]],purify,distance,**kwargs))

                    for future in as_completed(futures):
                        tmp_array.append(future.result())
                        
                    tmp_array=np.array(tmp_array)
                    stats = bootstrap((tmp_array,), np.mean, confidence_level=0.9, random_state=rng)
                    distances[idx]=[np.mean(tmp_array), stats.confidence_interval.low, stats.confidence_interval.high]
                    progress_bar.update(1)

    return distances

def Trotter_distances(exactdensities,Trdensities,distance=trace_distance,**kwargs):
    distances=np.zeros((Trdensities.shape[:2]))
    for idx in np.ndindex(Trdensities.shape[:2]):
        distances[idx]=distance(Trdensities[idx],exactdensities[idx[0]],**kwargs)
    return distances




ising1D=Ising1D_Hamiltonian(2)

total_time_list = [0.1,0.5,1.0,1.5,2.0]
nsteps_list = [10,20,40,80,160,240,320,400,480,560,640]  #  total gates used = number of Hterms x nsteps
nsamples_list=equiv_nsamples_list(ising1D,nsteps_list)   #  total gates used = nsamples (plot with these values)
ntraj=1000000
num_mix=10000
M_list=[200,400,600,800,1000,2000,4000,6000,8000,10000,20000,40000,60000,80000,100000,200000,400000,600000,800000,1000000]

Qdensities=compute_Qdrift_evolutions(ising1D,total_time_list,nsamples_list,ntraj,save_path="1MQevolutions.pkl")
exactdensities=compute_exact_evolutions(ising1D,total_time_list)


if os.path.exists("QTD.pkl"):
    pass
else:
    QTD=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list)
    with open("QTD.pkl", 'wb') as f:
        pickle.dump(QTD, f)    

if os.path.exists("QTDpure.pkl"):
    pass
else:
    QTDpure=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,purify=True)
    with open("QTDpure.pkl", 'wb') as f:
        pickle.dump(QTD, f)

if os.path.exists("QRTD.pkl"):
    pass
else:
    QRTD1=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=0)
    QRTD2=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=1)
    with open("QRTD.pkl", 'wb') as f:
        pickle.dump((QRTD1,QRTD2), f)

if os.path.exists("QRTDpure.pkl"):
    pass
else:
    QRTDpure1=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=0,purify=True)
    QRTDpure2=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=1,purify=True)
    with open("QRTDpure.pkl", 'wb') as f:
        pickle.dump((QRTDpure1,QRTDpure2), f)

if os.path.exists("QS.pkl"):
    pass
else:
    QS1=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=0,purify=True)
    QS2=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=1,purify=True)
    with open("QS.pkl", 'wb') as f:
        pickle.dump((QS1,QS2), f)

if os.path.exists("QSpure.pkl"):
    pass
else:
    QSpure1=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=0,purify=True)
    QSpure2=Qdrift_distance_stats(exactdensities,Qdensities,num_mix,M_list,distance=reduced_trace_distance,site_index=1,purify=True)
    with open("QSpure.pkl", 'wb') as f:
        pickle.dump((QSpure1,QSpure2), f)