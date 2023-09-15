import robohive
import numpy as np
from mjrl.utils.gym_env import GymEnv
import pickle
from tqdm import tqdm
from copy import deepcopy

RELOCATE_OBS_KEYS = ['hand_jnt', 'palm_obj_err', 'palm_tar_err', 'obj_tar_err']
HAMMER_OBS_KEYS = ['hand_jnt', 'obj_vel', 'palm_pos', 'obj_pos', 'obj_rot', 'target_pos', 'nail_impact']

def get_custom_obs(env, keys):
    raw_env = env.env.env
    obs = []
    for k in keys:
        val = raw_env.obs_dict[k]
        if len(val.shape) == 0:
            obs.append(np.expand_dims(val, axis=0))
        else:
            obs.append(raw_env.obs_dict[k])
    return np.concatenate(obs)

def get_obs_dict(env):
    raw_env = env.env.env
    return raw_env.obs_dict

def convert_demos(env, keys_v1):
    o1 = get_custom_obs(env, keys_v1)
    new_dim = o1.shape[0]

    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))

    new_demos = []

    for path in tqdm(demos):
        env.reset()
        orig_sd = path['init_state_dict']
        env.set_env_state(orig_sd)

        new_obs = np.zeros((path['observations'].shape[0], new_dim))

        init_obs = {}
        for k in orig_sd:
            init_obs[k] = orig_sd[k].copy()

        new_rew = np.zeros((path['observations'].shape[0],))
        actions = path['actions']
        for t in range(actions.shape[0]):
            o, rew, done, info = env.step(actions[t])
            new_obs[t] = get_custom_obs(env, keys_v1)
            new_rew[t] = rew
            # e_v0.env.mj_render()
        new_demo = dict(observations=new_obs, init_state_dict = init_obs, actions=actions)
        new_demo['rewards'] = new_rew
        new_demos.append(new_demo)

    pickle.dump(new_demos, open('./demonstrations/'+env_name+'_demos_updated.pickle', 'wb'))

env_name = 'hammer-v1'
env_v1 = GymEnv(env_name)
env = GymEnv(env_name, env_kwargs=dict(obs_keys=HAMMER_OBS_KEYS))
keys_v1 = env_v1.env.env.obs_keys
convert_demos(env, keys_v1)