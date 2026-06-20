import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import logging

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2_model import TDMPC2

torch.backends.cudnn.benchmark = True

LOG = logging.getLogger(__name__)


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	LOG.info(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	LOG.info(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	LOG.info(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		LOG.warning(colored('single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		LOG.info(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		LOG.info(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		LOG.info(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	
	LOG.info("Starting evaluation...")
	for _, task in enumerate(tasks):
		if cfg.multitask:
			raise NotImplementedError
		ep_rewards, ep_successes, ep_lengths = [], [], []
		n_eval_episodes = cfg.test_episodes // cfg.num_envs
		for i in range(n_eval_episodes):
			LOG.info(f"Eval iterations {i}/{n_eval_episodes}")
			obs, done, ep_reward, t = env.reset(), torch.tensor(False), 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done.any():
				torch.compiler.cudagraph_mark_step_begin()
				action = agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			
			LOG.info(ep_reward)
			frames = np.stack(frames)
			frames = frames.transpose(1, 0, 2, 3, 4)
			T, N, H, W, C = frames.shape
			frames = frames.reshape(N * T, H, W, C)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if cfg.save_video:
				video_path = os.path.join(video_dir, f'{task}-{i}.mp4')
				imageio.mimsave(
					os.path.join(video_path), frames, fps=15)
				LOG.info(f"Video saved to the path {video_path}")
		
		ep_rewards_cat = torch.cat(ep_rewards) 
		ep_rewards_mean = ep_rewards_cat.mean()
		ep_rewards_median = ep_rewards_cat.median()
		ep_successes = np.mean(ep_successes)
		episode_length = torch.tensor(ep_lengths, dtype=torch.float32).mean()
		
		LOG.info(
			f"eval_reward_mean={ep_rewards_mean:.2f} "
			f"eval_reward_std={ep_rewards_cat.std():.2f} "
			f"eval_reward_median={ep_rewards_median:.2f} "
			f"eval_episode_length_mean={episode_length:.1f}"
		)
  
	if cfg.multitask:
		LOG.info(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	import multiprocessing as mp
	mp.set_start_method('spawn', force=True)
	evaluate()
