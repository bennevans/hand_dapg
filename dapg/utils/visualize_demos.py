#import mj_envs
import robohive
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    demos_new = pickle.load(open('./demonstrations/'+env_name+'_demos_updated.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos_new)

def demo_playback(env_name, demo_paths):
    e = GymEnv(env_name, env_kwargs=dict(obs_keys=['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err']))
    e.reset()
    for path in demo_paths:
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            e.env.mj_render()

if __name__ == '__main__':
    main()
