#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pyNN.spiNNaker as sim
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN import space
import neo
from quantities import ms
import networkx as netx
import scipy.sparse
import os.path
from matplotlib.lines import Line2D
import gc


# In[2]:


seed = 764756387
rng = NumpyRNG(seed=seed, parallel_safe=True)


# In[3]:


# In[4]:


pwd = os. getcwd()


# In[5]:


filename = '/EMG_dataset_with_spike_time_python3.npz'


# In[7]:


data=np.load(pwd+filename,allow_pickle=True )


# In[8]:


class WeightRecorder(object):
    """
    Recording of weights is not yet built in to PyNN, so therefore we need
    to construct a callback object, which reads the current weights from
    the projection at regular intervals.
    """

    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection
        self._weights = []

    def __call__(self, t):
        self._weights.append(self.projection.get('weight', format='list', with_address=False))
        return t + self.interval

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal


# In[9]:


data.files


# In[10]:




Y_EMG_Train = data['Y_EMG_Train']
Y_EMG_Test = data['Y_EMG_Test']
spike_times_train_up = data['spike_times_train_up']
spike_times_train_dn = data['spike_times_train_dn']
spike_times_test_up = data['spike_times_test_up']
spike_times_test_dn = data['spike_times_test_dn']


# In[11]:


n_exc = 256
n_inh = 64
n_stim = 8


cell_parameters = {
    'tau_m' : 20.0,# ms
    'tau_syn_E': 2.0, # ms
    'tau_syn_I': 5.0,# ms
    'v_rest': -65.0,  # mV
    'v_reset'  : -70.0,  # mV
    'v_thresh':  -50.0,# mV
    'cm':     1.0,    #nF
    'tau_refrac': 1.0,  # ms 
    'e_rev_E':   0.0,
    'e_rev_I': -70.0,
}


synaptic_parameters = {
    'excitatory_stdp': {
        'timing_dependence': {'tau_plus': 20.0, 'tau_minus': 20.0,
                              'A_plus': 0.01, 'A_minus': 0.01},
        'weight_dependence': {'w_min': 0, 'w_max': 0.5},
        'weight': 0.1,
        'delay': 0.1},
    'excitatory_simple': {'weight': RandomDistribution('uniform', low=0.025, high=0.05, rng=rng),'delay': 0.1},
    'inhibitory': {'weight': RandomDistribution('uniform', low= -0.05, high= -0.025, rng=rng), 'delay': 0.1},
    'input': {'weight': 0.05, 'delay': 0.1},
}

connectivity_parameters = {
    'gaussian': {'d_expression': 'exp(-d**2/1e4)'},
    'global_reservoir': {'p_connect': 0.0625},
    'global_input': {'p_connect': 0.125},
    'input_exc': {'n': 3},
    'input_inh': {'n': 3},
}
grid_parameters = {
    'aspect_ratio': 1, 'dx': 50.0, 'dy': 50.0, 'fill_order': 'random'
}



# In[12]:


N = n_exc+n_inh
graph = netx.watts_strogatz_graph(N, 8, 0.01, seed=None)

connection_matrix = netx.adjacency_matrix(graph)
connection_matrix= connection_matrix.todense()  # Convert to dense
connection_matrix_exc_exc = connection_matrix[:n_exc,:n_exc]
connection_matrix_exc_inh = connection_matrix[:n_exc,n_exc:]
connection_matrix_inh_exc = connection_matrix[n_exc:,:n_exc]

connection_matrix_small_world_exc_exc = np.array(connection_matrix_exc_exc, dtype = bool)
connection_matrix_small_world_exc_inh = np.array(connection_matrix_exc_inh, dtype = bool)
connection_matrix_small_world_inh_exc = np.array(connection_matrix_inh_exc, dtype = bool)


# Initiate the network

# In[13]:


nb_samples = len(Y_EMG_Test)
nb_neurons = n_exc + n_inh
output_spike_count_matrix = np.zeros((nb_samples, nb_neurons))


# In[ ]:


for i in range(nb_samples) :
    print(i)
    input_ex = spike_times_test_up[i]
    input_inh = spike_times_test_dn[i]
   
    sim.setup(timestep=1)

    all_cells = sim.Population(n_exc + n_inh, sim.IF_cond_exp(**cell_parameters),
                               structure=space.Grid2D(**grid_parameters),
                               label="All Cells")
    exc_cells = all_cells[:n_exc]; exc_cells.label = "Excitatory cells"

    inh_cells = all_cells[n_exc:]; inh_cells.label = "Inhibitory cells"

    exc_cells_input_subset = exc_cells[:n_exc/4]; exc_cells_input_subset.label = "Excitatory cells Input subset"

    inh_cells_input_subset = inh_cells[:n_inh]; inh_cells_input_subset.label = "Inhibitory cells Input subset"


    ext_stim_exc = sim.Population(n_stim, sim.SpikeSourceArray(spike_times= spike_times_train_up[i].tolist() ),
                          label="Input spike time array exitatory")
    ext_stim_inh = sim.Population(n_stim, sim.SpikeSourceArray(spike_times= spike_times_train_dn[i].tolist() ),
                          label="Input spike time array inhibitory")

    # stdp_mechanism = sim.STDPMechanism(
    #                     timing_dependence=sim.SpikePairRule(**synaptic_parameters['excitatory_stdp']['timing_dependence']),
    #                     weight_dependence=sim.AdditiveWeightDependence(**synaptic_parameters['excitatory_stdp']['weight_dependence']),
    #                     weight=synaptic_parameters['excitatory_stdp']['weight']
    #                     ,delay=synaptic_parameters['excitatory_stdp']['delay'], 
    #     dendritic_delay_fraction=0

    # )
    # Determine the connectivity among the population of neuron

    
    global_connectivity_reservoir = sim.FixedProbabilityConnector(
                                **connectivity_parameters['global_reservoir'])

    array_connectivity_small_world_exc_exc = sim.ArrayConnector(connection_matrix_small_world_exc_exc)
    array_connectivity_small_world_exc_inh = sim.ArrayConnector(connection_matrix_small_world_exc_inh)
    array_connectivity_small_world_inh_exc = sim.ArrayConnector(connection_matrix_small_world_inh_exc)



    input_connectivity_exc = sim.FixedNumberPostConnector(
                                **connectivity_parameters['input_exc'])
    input_connectivity_inh = sim.FixedNumberPostConnector(
                                **connectivity_parameters['input_inh'])


    input_exc_exc_connections = sim.Projection(ext_stim_exc, exc_cells_input_subset,
                                      input_connectivity_exc,
                                      receptor_type='excitatory',
                                      synapse_type=sim.StaticSynapse(**synaptic_parameters['input']),
                                      label='Input connections')
    
    input_inh_inh_connections = sim.Projection(ext_stim_inh, inh_cells_input_subset,
                                      input_connectivity_inh,
                                      receptor_type='excitatory',
                                      synapse_type=sim.StaticSynapse(**synaptic_parameters['input']),
                                      label='Input connections')


    exc_exc_connections = sim.Projection(exc_cells, exc_cells,
                                     array_connectivity_small_world_exc_exc,
                                     receptor_type='excitatory',
                                     synapse_type=sim.StaticSynapse(**synaptic_parameters['excitatory_simple']),
                                     label='Excitatory - Excitatory connections')

    exc_inh_connections = sim.Projection(exc_cells, inh_cells,
                                     array_connectivity_small_world_exc_inh,
                                     receptor_type='excitatory',
                                     synapse_type=sim.StaticSynapse(**synaptic_parameters['excitatory_simple']),
                                     label='Excitatory - Inhibitory connections')

    inh_inh_connections = sim.Projection(inh_cells, exc_cells,
                                     array_connectivity_small_world_inh_exc,
                                     receptor_type='excitatory',
                                     synapse_type=sim.StaticSynapse(**synaptic_parameters['inhibitory']),
                                     label='Inhibitory connections')



    # == Instrument the network =================================================
#     weight_recorder_exc_exc= WeightRecorder(sampling_interval=0.1, projection=exc_exc_connections )
#     weight_recorder_exc_inh= WeightRecorder(sampling_interval=0.1, projection=exc_inh_connections )

    ext_stim_exc.record('spikes')
    ext_stim_inh.record('spikes')
    all_cells.record('spikes')

    # === Run the simulation =====================================================
    #====Calculate the number of iterations required====#
    t_stop = 3000
    nb_iter = t_stop/25
    spike_count_vector_master = np.ones((nb_iter,320))
    sim.run(25)
    #===Run the simulation in incremental time steps of 25 ms
    count = 0
    while (count < nb_iter):
        spike_count_vector_pre = np.array(all_cells.get_spike_counts().values())
        sim.run(25)
        spike_count_vector_post = np.array(all_cells.get_spike_counts().values())
        spike_count_vector_master[count] = spike_count_vector_post-spike_count_vector_pre
        count = count+1
    label = Y_EMG_Test[i]
    file_name = '/simulation_output/'+'test_'+ str(i)+'_class_'+ str(label)+'_all_cells'+'.npz'
     
    path = pwd + file_name
    np.savez(path, spike_count_vector_master=spike_count_vector_master, label=label)
    sim.end()
    gc.collect()


# In[ ]:





# In[ ]:




