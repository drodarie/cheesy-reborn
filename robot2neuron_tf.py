
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates


@nrp.MapVariable("nestO", initial_value=nrp.config.brain_root.nest)
@nrp.MapVariable("circuitO", initial_value=nrp.config.brain_root.circuit)
@nrp.MapRobotSubscriber("muscle_states_msg", Topic("/gazebo_muscle_interface/robot/muscle_states", MuscleStates))
@nrp.MapRobotSubscriber("joint_state_msg", Topic('/joint_states', JointState))
@nrp.Robot2Neuron()
def transferfunction_Robot2Neuron( t, nestO, circuitO, 
                      muscle_states_msg,
                      joint_state_msg
                      ):
    nest = nestO.value
    circuit = circuitO.value
    
    nest.SetStatus( list(np.where(circuit["IO_ids"]==10)[0]+1), "I_e", 600.0)
    #~ nest.SetStatus( circuit["cells"], "I_e", 1000.0)






