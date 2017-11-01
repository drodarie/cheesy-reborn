
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates

@nrp.MapRobotSubscriber("muscle_states_msg", Topic("/gazebo_muscle_interface/robot/muscle_states", MuscleStates))
@nrp.MapRobotSubscriber("joint_state_msg", Topic('/joint_states', JointState))
@nrp.MapSpikeSource("neurons", nrp.brain.neurons, nrp.poisson)
@nrp.Robot2Neuron()
def transferfunction_Robot2Neuron( t, 
                      muscle_states_msg,
                      joint_state_msg,
                      neurons
                      ):
    import traceback
    import time
    neurons.rate = 200000.0
    #~ for i in range(len(neurons.rate)):
        #~ neurons.rate[i] = 3000.0
        
#~ neurons
        





