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
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

RewardArr=[]
n_steps=0

def plot_step_response(desired, actual, choice,
                 end=1., title=None,
                 step_size=0.001, threshold_percent=0.1):
    """
        Args:
            threshold (float): Percent of the start error
    """

    #actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    #desired = desired[:end]
    threshold = threshold_percent * desired

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

    if choice==0:
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
    else:
        ax[0].set_ylabel("Roll (rad/s)")
        ax[1].set_ylabel("Pitch (rad/s)")
        ax[2].set_ylabel("Yaw (rad/s)")

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

def callback(_locals, _globals):
    global n_steps, RewardArr
    if (n_steps + 1) % 100 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print("Reward=",mean_reward)
            RewardArr.append(mean_reward)
    n_steps+=1
    return True


def main(env_id, seed):
    log_dir = "./trytry/"
    os.makedirs(log_dir, exist_ok=True)
    global RewardArr
    desired_pos=[]
    actual_pos=[]
    desired_att=[]
    actual_att=[]
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)
    print("Environment Ready!!!")
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000,callback=callback)
    model.save("ppo")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        desired_pos.append([0,0,2])
        actual_pos.append(info[0]["current_pos"])
        desired_att.append([0,0,0])
        actual_att.append(info[0]["current_att"])
    plot_step_response(np.array(desired_pos), np.array(actual_pos),choice=0)
    plot_step_response(np.array(desired_att), np.array(actual_att),choice=1)
    print(RewardArr)
    plt.plot(RewardArr)
    plt.show()
        

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
