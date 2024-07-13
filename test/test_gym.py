import isaacgym
import isaacgymenvs
import torch

num_envs = 1024

envs = isaacgymenvs.make(
    seed=0, 
	# task="Ant", 
	task="MultiRoboMaster", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	headless=False,
	graphics_device_id=0,
	force_render=True
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()

actions = torch.zeros((num_envs * envs.num_robots , envs.cfg["env"]["numActions"]), device = 'cuda:0')
actions[:, 0] = 1.0

for _ in range(10000):
	random_actions = 2.0 * torch.rand((num_envs * envs.num_robots, envs.cfg["env"]["numActions"]), device = 'cuda:0') - 1.0
	# random_actions = 2.0 * torch.rand((num_envs, ) + envs.action_space.shape, device = 'cuda:0') - 1.0
	# envs.step(random_actions)
	envs.step(actions)

breakpoint()