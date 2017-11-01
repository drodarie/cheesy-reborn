#from hbp_nrp_excontrol.logs import clientLogger
#from std_msgs.msg import Header

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates
@nrp.MapRobotPublisher("activateFoot1", Topic("/gazebo_muscle_interface/robot/Foot1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateFoot2", Topic("/gazebo_muscle_interface/robot/Foot2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius1", Topic("/gazebo_muscle_interface/robot/Radius1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius2", Topic("/gazebo_muscle_interface/robot/Radius2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus1", Topic("/gazebo_muscle_interface/robot/Humerus1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus2", Topic("/gazebo_muscle_interface/robot/Humerus2/cmd_activation", Float64))
@nrp.MapSpikeSink("neurons", nrp.brain.neurons, nrp.spike_recorder)
@nrp.Neuron2Robot(Topic('/monitor/spike_recorder', cle_ros_msgs.msg.SpikeEvent))
def transferfunction_Neuron2Robot( t, 
                      activateFoot1, activateFoot2,
                      activateRadius1, activateRadius2,
                      activateHumerus1, activateHumerus2,
                      neurons
                      ):
    import traceback
    import time
    clientLogger.info("Writing spikes (t="+str(t)+"): ")
    allSpikes = ""
    for i in range(len(neurons.times)):
        allSpikes += str(int(neurons.times[i][0]))+","+str(neurons.times[i][1])+" / "
    clientLogger.info(allSpikes)

    #~ for sender, value in [(activateHumerus1, nnData['LEFT_PMA']),(activateFoot1, nnData['LEFT_TA']),(activateRadius1, nnData['LEFT_RF']),(activateHumerus2, nnData['LEFT_CF']),(activateFoot2, nnData['LEFT_LG']),(activateRadius2, nnData['LEFT_POP'])]:
        #~ sender.send_message(value)                             

