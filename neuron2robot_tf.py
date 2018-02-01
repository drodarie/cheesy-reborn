#from hbp_nrp_excontrol.logs import clientLogger
#from std_msgs.msg import Header

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates
import nest
@nrp.MapVariable("nestO", initial_value=nrp.config.brain_root.nest)
@nrp.MapVariable("circuitO", initial_value=nrp.config.brain_root.circuit)
@nrp.MapRobotPublisher("activateFoot1", Topic("/gazebo_muscle_interface/robot/Foot1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateFoot2", Topic("/gazebo_muscle_interface/robot/Foot2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius1", Topic("/gazebo_muscle_interface/robot/Radius1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius2", Topic("/gazebo_muscle_interface/robot/Radius2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus1", Topic("/gazebo_muscle_interface/robot/Humerus1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus2", Topic("/gazebo_muscle_interface/robot/Humerus2/cmd_activation", Float64))
@nrp.Neuron2Robot()
def transferfunction_Neuron2Robot( t, nestO, circuitO, 
                      activateFoot1, activateFoot2,
                      activateRadius1, activateRadius2,
                      activateHumerus1, activateHumerus2
                      ):
    nest = nestO.value
    circuit = circuitO.value
    
    spikes = nest.GetStatus(circuit["rec"], "events")[0]
    spike_times   = list(spikes["times"])
    spike_senders = list(spikes["senders"]-1)
    
    clientLogger.info(str(len(spike_senders))+" spikes (t="+str(t)+"s)")
    
    nest.SetStatus(circuit["rec"], [{"n_events":0}])
    
    #~ clientLogger.info("("+str(numSpikes)+" spikes)")

    #~ for sender, value in [(activateHumerus1, nnData['LEFT_PMA']),(activateFoot1, nnData['LEFT_TA']),(activateRadius1, nnData['LEFT_RF']),(activateHumerus2, nnData['LEFT_CF']),(activateFoot2, nnData['LEFT_LG']),(activateRadius2, nnData['LEFT_POP'])]:
        #~ sender.send_message(value)                             

