# Control for the PID controller and muscles in the Clostermann foreleg
#! /usr/bin/env python2.7

import sys
import rospy
import time
import numpy
import threading
from pprint import pprint
import traceback
import Queue

from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.srv import GetMuscleActivations, SetMuscleActivations, GetMuscleStates
from gazebo_msgs.srv import AdvanceSimulation
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from generic_controller_plugin.srv import SetPIDParameters


class SimControl(object):
  def __init__(self):
    ##############################
    ### clock topic listening ###
    ##############################
    self._clock_cb_mutex = threading.Lock()
    self._sim_time = None
    def _clock_callback(clock):
      with self._clock_cb_mutex:
        _sim_time = clock.clock.secs + clock.clock.nsecs * 1.e-9
    self.sub_clock = rospy.Subscriber('/clock', Clock, _clock_callback, queue_size=1)
    ##############################
    ##  Sim control             ##
    ##############################
    rospy.wait_for_service('/gazebo/pause_physics')
    self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    rospy.wait_for_service('/gazebo/unpause_physics')
    self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    rospy.wait_for_service('/gazebo/advance_simulation')
    self._advance_simulation = rospy.ServiceProxy('/gazebo/advance_simulation', AdvanceSimulation)
    # Figure out simulation time. So that the next calls to self.get_sim_time() return not None.
    if self.get_sim_time() is None:
      print "Advancing simulation one step in order to obtain simulation time"
      self._advance_simulation(1)
    print "Sim time = ", self.get_sim_time()

  def advance_simulation(self, dt):
    """Advance simulation by the given interval in seconds."""
    t = self.get_sim_time()
    tend = t + dt
    while t < tend:
      self._advance_simulation(1)
      t = self.get_sim_time()

  def get_sim_time(self):
    ''' Return last received simulation time as float in units of seconds.'''
    with self._clock_cb_mutex:
      return self._sim_time



class MuscleControl(object):
  '''
    To control the muscles via a simple to use interface.
  '''
  def __init__(self, model_name, joint_name_of_sled = None):
    self._model_name = model_name
    self._init_muscle_processing()

  def _init_muscle_processing(self):
    '''
        Register with ROS services.
    '''
    def _obtain_muscle_service_proxy(name, msgtype):
      '''Helper method to connect with the services.'''
      service_name = '/gazebo_muscle_interface/%s/%s' % (self._model_name, name)
      print 'Waiting for service %s' % service_name
      rospy.wait_for_service(service_name)
      cb = rospy.ServiceProxy(service_name, msgtype, persistent=True)
      return cb
    self._get_muscle_activations = _obtain_muscle_service_proxy('get_activations', GetMuscleActivations)
    self._set_muscle_activations = _obtain_muscle_service_proxy('set_activations', SetMuscleActivations)
    self._get_muscle_states      = _obtain_muscle_service_proxy('get_states', GetMuscleStates)
    self._muscle_states = self._get_muscle_states()
    self._muscle_name_to_idx = dict((k.name, i) for (i,k) in enumerate(self._muscle_states.muscles))

  def get_muscle_names(self):
    items = self._muscle_name_to_idx.items()
    items = sorted(items, key = lambda q: q[1])
    return map(lambda q: q[0], items)

  # "Public" API function
  def set_muscle_activations(self, kws):
    '''Set muscles activations via dict indexed by muscle name. Values are float type activations values.'''
    activations = list(self._get_muscle_activations().activations)
    for k, v in kws.items():
      activations[self._muscle_name_to_idx[k]] = v
    self._set_muscle_activations(activations = activations)

  def get_muscle_states(self):
    msg = self._get_muscle_states()
    return msg.muscles


class SledControl(object):
  def __init__(self, model_name, joint_name_of_sled):
    self._model_name = model_name
    self._joint_name_of_sled = joint_name_of_sled
    self._init_joint_states_processing()
    self._init_sled_controller()
    self.POS_FWD = 0.010
    self.POS_BWD = -0.01

  ##############################
  ### joint states listening ###
  ##############################
  def _init_joint_states_processing(self):
    # Here we use a queue because we are interested in the entire history because we want to do filtering to remove noise.
    # The Queue class is already thread safe so no external locking is required.
    self._jointstates_queue = Queue.Queue(100)
    def _joint_states_callback(jointstates):
      """This is the ROS callback"""
      try:
        self._jointstates_queue.put(jointstates, block = False)
      except Queue.Full:
        pass
        #print ("Ooops! Jointstate queue is full!")
    self.sub_jointstates = rospy.Subscriber("/joint_states", JointState, _joint_states_callback, queue_size=1)

  def _transform_joint_state_into_nice_repr(self, js):
    ret = {}
    for name, p, v, f in zip(js.name, js.position, js.velocity, js.effort):
      p = (p-self.POS_BWD)/(self.POS_FWD-self.POS_BWD)
      ret[name] = {
        'position' : p,
        'velocity' : v,
        'effort'   : f,
      }
    return ret

  def _get_joint_states(self):
    '''Return last received joint states. Returns list of dicts indexed by joint name.
       Dict values are dicts indexed by data name: position, velocity and effort.'''
    n = self._jointstates_queue.qsize()
    ret = []
    # Quickly fetch all the stuff from the queue
    for i in xrange(n):
      ret.append(self._jointstates_queue.get(block = False))
    # Then put it in a nice format
    ret = map(self._transform_joint_state_into_nice_repr, ret)
    return ret

  def get_last_received_sled_states(self):
    js_list = self._get_joint_states()
    return map(lambda states: states[self._joint_name_of_sled], js_list)

  def get_last_received_sled_joint_efforts(self):
    #js_list = self._get_joint_states()
    #if not js_list:
    #  return []
    # Note: The first item was received last!
    #efforts = [js[self._joint_name_of_sled]['effort'] for js in js_list]
    #return efforts
    sled_states = self.get_last_received_sled_states()
    return map(lambda state: state['effort'], sled_states)

  ##############################
  ##  generic controller      ##
  ##############################
  def _init_sled_controller(self):
    service_name = '/%s/set_pid_parameters' % self._model_name
    print 'Waiting for service %s' % service_name
    rospy.wait_for_service(service_name, timeout = 10)
    self._set_pid_parameters = rospy.ServiceProxy(service_name, SetPIDParameters)
    publisher_name = '/%s/%s/cmd_pos' % (self._model_name, self._joint_name_of_sled.replace('::','__'))
    print 'Publishing sled commands to %s' % publisher_name
    self._sled_pid_pub = rospy.Publisher(publisher_name, Float64, queue_size=1)

  def command_sled_position(self, fraction_fwd):
    raw_val = self.POS_FWD * fraction_fwd + (1.-fraction_fwd) * self.POS_BWD
    self.command_sled_position_raw(raw_val)

  def command_sled_position_raw(self, p):
    self._sled_pid_pub.publish(Float64(data=p))

  def set_sled_pid_parameters(self, kp, ki, kd):
    self._set_pid_parameters(joint = self._model_name+'::'+self._joint_name_of_sled, kp = kp, ki = ki, kd = kd)


if __name__ == '__main__':
  from PyQt4 import QtCore, QtGui, uic
  ##############################
  ##  QT Gui
  ##############################
  class MainWindow(QtGui.QMainWindow):
    def __init__(self, sim_control, sled_control, muscle_control):
      QtGui.QMainWindow.__init__(self)
      self.sim_control = sim_control
      self.muscle_control = muscle_control
      self.sled_control = sled_control
      self.ui = uic.loadUi('manualcontrol.ui', self)
      self.ui.runButton.clicked.connect(self.runSimulation)
      self.ui.sledButton.clicked.connect(lambda on: self.updateSledControl())
      for field in [
        self.ui.kpField,
        self.ui.kiField,
        self.ui.kdField,
      ]:
        field.editingFinished.connect(self.updateSledControl)
      self.ui.sledPosSlider.valueChanged.connect(lambda val: self.updateSledTarget())
      self.ui.musclesSlider.valueChanged.connect(lambda val: self.updateMuscles())
      # Default PID control parameters here!
      self.ui.kpField.setValue(40.)
      self.ui.kiField.setValue(5.)
      self.ui.kdField.setValue(10.)
      timer = QtCore.QTimer(self)
      timer.timeout.connect(self.updateEffortLabel)
      timer.start(100)
      self.lastEffort = None
      self.updateSledControl()
      self.updateSledTarget()
      self.updateMuscles()
      self.show()

    def runSimulation(self, on):
      try:
        if on:
          self.sim_control.unpause_physics()
        else:
          self.sim_control.pause_physics()
      except Exception, e:
        tb = traceback.format_exc()
        print ("=== %s ===" % str(e))
        print (tb)

    def updateSledControl(self):
      on = self.ui.sledButton.isChecked()
      if on:
        kp = self.ui.kpField.value()
        ki = self.ui.kiField.value()
        kd = self.ui.kdField.value()
      else:
        kp, ki, kd = 0., 0., 0.
      self.sled_control.set_sled_pid_parameters(kp, ki, kd)

    def updateSledTarget(self):
      val = self.ui.sledPosSlider.value()
      self.sled_control.command_sled_position(1. - val * 0.01)

    def updateMuscles(self):
      FULL_FWD = dict(
        (k, 0.) for k in self.muscle_control.get_muscle_names()
      )
      FULL_FWD.update({
        'Humerus1': 1.0,
        'Foot2': 1.0,
      })
      FULL_BWD = dict(
        (k, 0.) for k in self.muscle_control.get_muscle_names()
      )
      FULL_BWD.update({
        'Humerus2': 1.0,
        'Radius1': 1.0,
      })
      OFF = dict(
        (k, 0.) for k in self.muscle_control.get_muscle_names()
      )
      val = self.ui.musclesSlider.value()
      if val < 0:
        val += 100
        interp1 = FULL_FWD
        interp2 = OFF
      else:
        interp1 = OFF
        interp2 = FULL_BWD
      val *= 0.01
      cmd_dict = {}
      for (k, v1), v2 in zip(interp1.items(), interp2.values()):
        cmd_dict[k] = v1 + val * (v2 - v1)
      self.muscle_control.set_muscle_activations(cmd_dict)

    def updateEffortLabel(self):
      efforts = self.sled_control.get_last_received_sled_joint_efforts()
      if self.lastEffort is not None:
        effort = self.lastEffort
      elif efforts:
        effort = efforts[0]
        efforts = efforts[1:]
      else:
        effort = 0
      # Do low-pass filtering because the force generated by
      # the PID controller is very noisy.
      # We should make the mixing factors time dependent, though.
      # So the filtering is done with respect to frequency in
      # simulated time (or real time alternatively).
      for ef in efforts:
        effort = effort * 0.9 + ef * 0.1
      self.lastEffort = effort
      self.ui.effortLabel.setText("Effort: %f N" % effort)
      self.ui.effortLabel.update()


##############################
##  Main routine            ##
##############################
if __name__ == '__main__':
  model_name = sys.argv[1] if len(sys.argv)>1 else "mouse_and_sled"
  joint_name_of_sled = sys.argv[2] if len(sys.argv)>2 else "cdp1_msled::world_sled"
  print "Waiting for ROS ..."
  rospy.init_node('mouse_experiment', anonymous=False)
  print "Initializing mouse experiment control for model %s and sled joint %s" % (model_name, joint_name_of_sled)
  sim_control = SimControl()
  muscle_control = MuscleControl(model_name)
  sled_control = SledControl(model_name, joint_name_of_sled)
  print "Initialization done!"
  app = QtGui.QApplication(sys.argv)
  window = MainWindow(sim_control, sled_control, muscle_control)
  sys.exit(app.exec_())
