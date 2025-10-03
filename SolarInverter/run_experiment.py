'''################################################################################
#                                                                                 #
#-------------Code to perform anomaly detection on a solar inverter---------------#
#                                                                                 #
###################################################################################
'''

#Imports
import torch, pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre

'''
#Establish here the desired configuration:
    inh: only one forward connection with depresion
    exc_inh: Forward connection with potentiation, recurrent connection with depresion
'''
configuration='inh'#'exc_inh'

if configuration=='inh':
    nu1=(0.1,-0.1)
    recurrence=False
elif configuration=='exc_inh':
    nu1=(-0.1,0.1)
    recurrence=True

#nu2 is irrelevant if the configuration is 'inh'.
nu2=(0.1,-0.1)

#Set simulation time.
T = 100

#The rest of the parameters
n=1000 #Number of neurons
threshold=-55
decay=100

#Set the intervals for interval coding
intervals=torch.FloatTensor(np.arange(-10,170,1))

R=len(intervals)-1

#Functions required

def reset_voltages(network):
    """
    Reset the voltages of neurons in layer 'B' to -65.
    """
    network.layers['B'].v=torch.full(network.layers['B'].v.shape,-65)
    return network


#Funcioncilla para convertir a spikes las entradas:
def encode_spike(x,q1,q2):
    """
    Return 1 (spike) if x is in [q1, q2), else 0. Used for data encoding.
    """
    s=torch.zeros_like(x)
    s[(x>=q1) & (x<q2)]=1
    return s

def read_data(input_folder,T):
    """
    Read data and encode them.
    """
    data=pd.read_csv(input_folder)
    
    series=torch.FloatTensor(data['Power'])
    
    length=series.shape[0]
    #Clamping
    series[series<torch.min(intervals)]=torch.min(intervals)
    series[series>torch.max(intervals)]=torch.max(intervals)
    
    series2input=torch.cat([series.unsqueeze(0)] * R, dim=0)
    
    for i in range(R):
        series2input[i,:]=encode_spike(series2input[i,:],intervals[i],intervals[i+1])
    
    sequences = torch.split(series2input,T,dim=1)
    
    sequences=sequences[0:len(sequences)-1]
    
    return sequences



def create_network(R,T,n,threshold,decay,nu1,nu2,recurrence):
    """
    Create and configure a spiking neural network with input and LIF layers.
    Optionally adds a recurrent connection.
    Returns the network and spike/voltage monitors.
    """
    network = Network()
    source_layer = Input(n=R,traces=True)
    target_layer = LIFNodes(n=n,traces=True,thresh=umbral, tc_decay=decaimiento)
    
    network.add_layer(
        layer=source_layer, name="A"
    )
    network.add_layer(
        layer=target_layer, name="B"
    )
    
    #Create connections
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
        update_rule=PostPre, nu=nu1
    )
    
    network.add_connection(
        connection=forward_connection, source="A", target="B"
    )
    
    if recurrence:
        recurrent_connection = Connection(
            source=target_layer,
            target=target_layer,
            w=0.025 * (torch.eye(target_layer.n) - 1),
            update_rule=PostPre, nu=nu2#nu=(1e-4, 1e-2)
        )
        
        network.add_connection(
            connection=recurrent_connection, source="B", target="B"
        )
    
    #Create the monitors
    source_monitor = Monitor(
        obj=source_layer,
        state_vars=("s",),
        time=T,
    )
    target_monitor = Monitor(
        obj=target_layer,
        state_vars=("s", "v"),
        time=T,
    )
    
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    
    
    return [network,source_monitor,target_monitor]


def run_network(sequences,network,source_monitor,target_monitor,T):
    """
    Run the SNN on the provided sequences for training or evaluation.
    Returns spike counts and the network.
    """
    sp0=[]
    sp1=[]
    
    for idx, seq in enumerate(sequences, 1):
        print(f'Running sequence {idx}')
        inputs={'A':seq.T}
        network.run(inputs=inputs, time=T)
        spikes = {
            "X": source_monitor.get("s"), "B": target_monitor.get("s")
        }
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        voltages = {"Y": target_monitor.get("v")}
    sp0=torch.concatenate(sp0)
    sp0=sp0.detach().numpy()
    sp1=torch.concatenate(sp1)
    sp1=sp1.detach().numpy()
    return [sp0,sp1,network]


# Create the network.
network, source_monitor,target_monitor = create_network(R,T,n,threshold,decay,nu1,nu2,recurrence)
network=reset_voltajes(network)

#Read input data. It contains timestamp, power and labels.
sequences2train=read_data('data/train.csv',T)
sequences2test=read_data('data/test.csv',T)

#Train
network.learning=True
_,spikes_train,network=run_network(sequences2train,network,source_monitor,target_monitor,T)
network=reset_voltajes(network)

#Test
network.learning=False
_,spikes_test,network=run_network(sequences2test,network,source_monitor,target_monitor,T)

#Save results
np.savetxt('outputs/spikes_'+configuration,spikes_test,delimiter=',')
