import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")


class GoToPositionEnv(GazeboEnv):
    def __init__(self, **kwargs): 
        self.max_sim_time = kwargs["max_sim_time"]
        self.obs = None
        self.observation_history = []
        self.memory_size = 1
        self.err_position = np.zeros(3)
        self.err_omega = np.zeros(3)
        self.err_velocity = np.zeros(3)
        self.rotateMatrix = np.zeros(9)
        super(GoToPositionEnv, self).__init__()
        self.sample_target()

    def sample_target(self):
        """ Sample a random angular velocity """
        #self.position_target = np.array([1, 0, 0])
        self.position_target = self.np_random.uniform(-1, 1, size=3)
        self.velocity_target = np.array([0, 0, 0])
        self.omega_target = np.array([0, 0, 0])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action)
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward(action)
        info = {"sim_time": self.sim_time, "current_pos":self.position_actual, "current_att": self.omega_actual, "current_vel": self.velocity_actual,"target_pos":self.position_target}
        return state, reward, done, info

### position control ->
    def GetRotationMatrix(self):
        w = self.obs.orientation_quat[0]
        x = -self.obs.orientation_quat[1]
        y = -self.obs.orientation_quat[2]
        z = -self.obs.orientation_quat[3]
        
        sqrSum = w*w + x*x + y*y + z*z
        if(abs(sqrSum-1) > 0.00001):  # need norm
            s =  np.sqrt(sqrSum)
            if(abs(s) <= 0.00001):
                w = 1.0
                x = 0.0
                y = 0.0
                z = 0.0
            else:
                w /= s
                x /= s
                y /= s
                z /= s
        
        wx = w*x
        wy = w*y
        wz = w*z
        xx = x*x
        xy = x*y
        xz = x*z
        yy = y*y
        yz = y*z
        zz = z*z        
        
        rm = np.zeros(9)
        rm[0] = 1.0 - 2.0 * (yy + zz)
        rm[1] = 2.0*(xy - wz)
        rm[2] = 2.0*(xz + wy)
        rm[3] = 2.0*(xy + wz)
        rm[4] = 1.0-2.0*(xx + zz)
        rm[5] = 2.0*(yz - wx)
        rm[6] = 2.0*(xz - wy)
        rm[7] = 2.0*(yz + wx)
        rm[8] = 1.0-2.0*(xx + yy)
        
        return rm

    def state(self):
        if self.obs is None:
            return np.zeros(22)

        self.err_position = self.position_target - self.obs.position_xyz
        self.err_velocity = self.velocity_target - self.obs.velocity_xyz
        self.err_omega = self.omega_target - self.obs.angular_velocity_rpy
        self.rotateMatrix = self.GetRotationMatrix()

        st = np.append(self.err_position, self.err_velocity)
        st = np.append(st, self.err_omega)
        st = np.append(st, self.rotateMatrix)
        st = np.append(st,np.array(self.obs.motor_velocity)/838.0)
        
        return st
        
    def compute_reward(self, action):
        rwd = np.sqrt(np.sum(np.square(self.err_position)))
        rwd += 0.01*np.sqrt(np.sum(np.square(self.err_omega)))
        #rwd += 0.05*np.sqrt(np.sum(np.square(action)))

        return -rwd
    ### position control <-

    def reset(self):
        self.observation_history = []
        self.sample_target()   ### position control
        return super(GoToPositionEnv, self).reset()


class AttitudeFlightControlEnv(GazeboEnv):
    def __init__(self, **kwargs): 
        self.max_sim_time = kwargs["max_sim_time"]
        super(AttitudeFlightControlEnv, self).__init__()

    def compute_reward(self):
        """ Compute the reward """
        return -np.clip(np.sum(np.abs(self.error))/(self.omega_bounds[1]*3), 0, 1)

    def sample_target(self):
        """ Sample a random angular velocity """
        return  self.np_random.uniform(self.omega_bounds[0], self.omega_bounds[1], size=3)
    
class GyroErrorFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      ( (3 * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        self.observation_history = []
        return super(GyroErrorFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorESCVelocityFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()
        self.logData = []  # test logData
        self.numEpsd = 1   # test logData

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error, self.obs.motor_velocity]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}

        # test logData
        logLine = np.append(self.omega_target, self.omega_actual)
        #logLine = np.append(logLine, state)
        #logLine = np.append(logLine, action)
        #logLine = np.append(logLine, reward)
        logLine = np.append(logLine,self.obs.position_xyz)
        logLine = np.append(logLine,self.obs.velocity_xyz)
        logLine = np.append(logLine,self.obs.timestamp)
        self.logData.append(logLine)
        # test logData

        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      (( (3+self.motor_count) * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        # test logData
        if(len(self.logData) > 0):
            log_name = 'LogData%d.csv' % (self.numEpsd)
            np.savetxt(log_name, self.logData, delimiter=',')
            self.logData = []
            
            self.numEpsd = self.numEpsd + 1
        # test logData
        
        self.observation_history = []
        return super(GyroErrorESCVelocityFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackContinuousEnv(GyroErrorESCVelocityFeedbackEnv):
    def __init__(self, **kwargs): 
        self.command_time_off = kwargs["command_time_off"]
        self.command_time_on = kwargs["command_time_on"]
        self.command_off_time = None
        super(GyroErrorESCVelocityFeedbackContinuousEnv, self).__init__(**kwargs)

    def step(self, action):
        """ Sample a random angular velocity """
        ret = super(GyroErrorESCVelocityFeedbackContinuousEnv, self).step(action) 

        # Update the target angular velocity 
        if not self.command_off_time:
            self.command_off_time = self.np_random.uniform(*self.command_time_on)
        elif self.sim_time >= self.command_off_time: # Issue new command
            # Commands are executed as pulses, always returning to center
            if (self.omega_target == np.zeros(3)).all():
                self.omega_target = self.sample_target() 
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_on)
            else:
                self.omega_target = np.zeros(3)
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_off) 

        return ret 

