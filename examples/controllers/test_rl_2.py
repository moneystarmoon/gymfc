import argparse
import gym
import gymfc
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import math
import os
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import PPO1


def plot_step_response(desired, actual,
                 end=1., title=None,
                 step_size=0.002, threshold_percent=0.1):
    """
        Args:
            threshold (float): Percent of the start error
    """

    #actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    #desired = desired[:end]
    threshold = threshold_percent * desired

    plot_min = -math.radians(350)
    plot_max = math.radians(350)

    subplot_index = 3
    num_subplots = 3

    f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
    f.set_size_inches(10, 5)
    if title:
        plt.suptitle(title)
    ax[0].set_xlim([0, end_time])
    res_linewidth = 2
    linestyles = ["c", "m", "b", "g"]
    reflinestyle = "k--"
    error_linestyle = "r--"

    # Always
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")

    ax[-1].set_xlabel("Time (s)")


    """ ROLL """
    # Highlight the starting x axis
    ax[0].axhline(0, color="#AAAAAA")
    ax[0].plot(t, desired[:,0], reflinestyle)
    ax[0].plot(t, desired[:,0] -  threshold[:,0] , error_linestyle, alpha=0.5)
    ax[0].plot(t, desired[:,0] +  threshold[:,0] , error_linestyle, alpha=0.5)
 
    r = actual[:,0]
    ax[0].plot(t[:len(r)], r, linewidth=res_linewidth)

    ax[0].grid(True)



    """ PITCH """

    ax[1].axhline(0, color="#AAAAAA")
    ax[1].plot(t, desired[:,1], reflinestyle)
    ax[1].plot(t, desired[:,1] -  threshold[:,1] , error_linestyle, alpha=0.5)
    ax[1].plot(t, desired[:,1] +  threshold[:,1] , error_linestyle, alpha=0.5)
    p = actual[:,1]
    ax[1].plot(t[:len(p)],p, linewidth=res_linewidth)
    ax[1].grid(True)


    """ YAW """
    ax[2].axhline(0, color="#AAAAAA")
    ax[2].plot(t, desired[:,2], reflinestyle)
    ax[2].plot(t, desired[:,2] -  threshold[:,2] , error_linestyle, alpha=0.5)
    ax[2].plot(t, desired[:,2] +  threshold[:,2] , error_linestyle, alpha=0.5)
    y = actual[:,2]
    ax[2].plot(t[:len(y)],y , linewidth=res_linewidth)
    ax[2].grid(True)

    plt.show()

	
best_mean_reward = -np.inf
numMean = 30    # mean reward of last numMean episodes
best_reward = -np.inf
n_steps = 0
loadSpan = 1    # 8 callbacks(n_steps) per episode(1000 timestep)
numEpisodes = 0 # number of episodes

log_dir = "./log_ppo2_fixed_pos1/"
os.makedirs(log_dir, exist_ok=True)

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, loadSpan, numEpisodes, best_reward, numMean, best_mean_reward

  n_steps += 1
  if(n_steps % loadSpan) == 0:
      print("callback") 
      x, y = ts2xy(load_results(log_dir), 'episodes')
      numEp = len(x)
      if(numEp > numEpisodes):
          print(numEp, "episodes")
          numEpisodes = numEp

          lastReward = y[-1]
          if(lastReward > best_reward):
              best_reward = lastReward
              print(x[-1], "Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
		  
          mean_reward = np.mean(y[-numMean:])
          if(mean_reward > best_mean_reward):
              best_mean_reward = mean_reward
              print(x[-1], "Saving new best mean model")
              _locals['self'].save(log_dir + 'best_mean_model.pkl')
  return True	

def main(env_id, seed):
    desired_pos=[]
    actual_pos=[]
    desired_att=[]
    actual_att=[]
    desired_vel=[]
    actual_vel=[]	
    # Create log dir
    #log_dir = "./test_attitude/"
    #os.makedirs(log_dir, exist_ok=True)
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)
    print("Environment Ready!!!")
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    #env = SubprocVecEnv([lambda: env for i in range(1)])
    #model = PPO2(MlpPolicy, env,gamma=0.99, n_steps=16384, ent_coef=0.0, learning_rate=1e-4, vf_coef=0.5,
    #             max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=50, cliprange=0.2, verbose=0,
    #             _init_setup_model=True, policy_kwargs=None,
    #             full_tensorboard_log=False)
    model=PPO2(MlpPolicy,env,verbose=0,n_steps=2048,learning_rate=1e-4,noptepochs=10,nminibatches=64,ent_coef=0.0)
    #model.learn(total_timesteps=3000,callback=callback)
    #model.save(log_dir + 'final_model.pkl')
    
    model=PPO2.load('./best_model.pkl')
    obs = env.reset()
    for i in range(4000):
        #action=np.array([0.5,0.5,0.5,0.5])
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
        desired_pos.append(info[0]["target_pos"])
        actual_pos.append(info[0]["current_pos"])
        desired_att.append([0,0,0])
        actual_att.append(info[0]["current_att"])
        desired_vel.append([0,0,0])
        actual_vel.append(info[0]["current_vel"])
        #desired_att.append(info[0]["sp"])
        #actual_att.append(info[0]["current_rpy"])
    #print(actual_att)
    print(len(np.array(desired_pos)))
    plot_step_response(np.array(desired_pos), np.array(actual_pos))
    plot_step_response(np.array(desired_att), np.array(actual_att))
    plot_step_response(np.array(desired_vel), np.array(actual_vel))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', help="The Gym environement ID", type=str,
                        default="AttFC_GyroErr-MotorVel_M4_Pos-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=17)

    args = parser.parse_args()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir,
                               "../configs/iris.config")
    print ("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path

    main(args.env_id, args.seed)
