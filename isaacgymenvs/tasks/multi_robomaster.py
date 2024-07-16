import os
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.ma_vec_task import MA_VecTask
from isaacgymenvs.utils.torch_jit_utils import to_torch

from isaacgymenvs.utils.rm_utils import *
from gym.spaces import Box
from torch import Tensor
from typing import Tuple, Dict

from colorama import Fore


class MultiRoboMaster(MA_VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]

        # num_robots=4, num_agents=6, num_agent=2
        self.num_robots = self.cfg["env"]["numRobots"]
        self.cfg["env"]["numAgents"] = self.cfg["env"]["numRobots"] + 2
        self.num_agent = int(self.cfg["env"]["numRobots"] / 2)

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.wheel_stiffness = self.cfg["env"]["control"]["stiffness"]
        self.wheel_damping = self.cfg["env"]["control"]["damping"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.field_width = self.cfg["env"]["field"]["width"]
        self.field_length = self.cfg["env"]["field"]["length"]

        # reward scales
        self.rew_scales = {}
        for k, v in self.cfg["env"]["reward"].items():
            self.rew_scales[k] = v
        
        self.cfg["env"]["numActions"] = 3
        self.cfg["env"]["numStates"] = self.num_robots * 12 + 6
        self.cfg["env"]["numObservations"] = 54

        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(5.0, 0.0, 3.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # Shape: (num_env * num_actor, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # Shape: (num_dofs, 2)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # State
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print(f'root_states: {self.root_states.shape}')   # (num_envs * num_agents, 13) 
                                                         # position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
        self.robot_states = self.root_states.view(self.num_envs, self.num_agents, -1)[:, 2:, :]     # (num_envs, num_robots, 13)
        self.ball_states = self.root_states.view(self.num_envs, self.num_agents, -1)[:, 1, :].squeeze(1)    # (num_envs, 13)

        # DoF
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print(f'dof: {self.dof_state.shape}')   # (num_envs * num_agents * num_dof, 2), 2: pos, vel
        dof_state_shaped = self.dof_state.view(self.num_envs, -1, 2)  # dof_state_shaped: (num_envs, num_agents * num_dof, 2)
        self.default_dof_states = self.dof_state.clone()


    def allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs * self.num_robots, self.num_obs), device=self.device,dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs * self.num_robots, device=self.device, dtype=torch.long)
        self.extras = {'win': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                       'lose': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                       'draw': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}
        

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # self.sim_params.dt = 1 / 60.0  # default: 1/60.0
        # self.sim_params.substeps = 2  # default: 2
        # self.sim_params.use_gpu_pipeline = True
        # self.sim_params.physx.use_gpu = True
        # self.sim_params.physx.num_position_iterations = 8  # default: 4
        # self.sim_params.physx.num_velocity_iterations = 1  # default: 1
        # self.sim_params.physx.rest_offset = 0.001
        # self.sim_params.physx.contact_offset = 0.02
        # self.sim_params.physx.max_gpu_contact_pairs = 2920898
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        self._create_ground_plane()
        print(f'num_envs: {self.num_envs}, env_spacing: {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # # If randomizing, apply once immediately on startup before the fist sim step
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["static_friction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamic_friction"]
        plane_params.restitution = self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file_robot = "urdf/robomaster/robot/robomaster.urdf"
        asset_file_ball = "urdf/robomaster/ball.urdf"
        asset_file_field = "urdf/robomaster/field.urdf"
        
        # load robomaster asset
        robot_options = gymapi.AssetOptions()   # TODO: 控制碰撞数量
        robot_options.fix_base_link = False
        robot_options.collapse_fixed_joints = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file_robot, robot_options)
        
        # load ball asset
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file_ball, ball_options)

        # load field asset
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(self.sim, asset_root, asset_file_field, field_options)

        # set robomaster dof properties
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)    # 66 dofs
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)    # ['hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature']
        rm_gripper_limits = []      # lower, upper, max effort

        for i in range(self.num_dof):
            if robot_dof_props[i]['hasLimits']:    # 2个, 'left_gripper_joint_1', 'right_gripper_joint_1',
                robot_dof_props[i]['driveMode'] = gymapi.DOF_MODE_EFFORT
                robot_dof_props[i]['stiffness'] = 0.0
                robot_dof_props[i]['damping'] = 0.0
                rm_gripper_limits.append([robot_dof_props[i]['lower'], robot_dof_props[i]['upper'], robot_dof_props[i]['effort']])
            elif robot_dof_props[i]['velocity'] == 10:   # 4个, 'front_left_wheel_joint','front_right_wheel_joint','rear_left_wheel_joint','rear_right_wheel_joint',
                robot_dof_props[i]['driveMode'] = gymapi.DOF_MODE_VEL
                robot_dof_props[i]['stiffness'] = self.wheel_stiffness
                robot_dof_props[i]['damping'] = self.wheel_damping
                self.wheel_limits_lower = robot_dof_props[i]['lower']
                self.wheel_limits_upper = robot_dof_props[i]['upper']
            else:   # rollers, 'front_left_roller1_joint'...
                robot_dof_props[i]['driveMode'] = gymapi.DOF_MODE_NONE
                robot_dof_props[i]['stiffness'] = 0.0
                robot_dof_props[i]['damping'] = 0.0

        # define start pose for robomaster, 偶数在左边，奇数在右边            
        #                  ⬆ y
        #  -------------------------------    3
        # |                               |   
        # |                               |   
        # |                               |   0   -> x
        # |                               |   
        # |                               |           
        #  -------------------------------    -3
        # -4.5            0              4.5
        #
        robot_pose = [gymapi.Transform() for i in range(self.num_robots)]
        for i in range(self.num_robots):
            delta = 2*self.field_width / (self.num_agent +1)
            if i % 2 == 0:
                robot_pose[i].p = gymapi.Vec3(-2, -self.field_width+(i/2+1)*delta, 0.01)
            else:
                robot_pose[i].p = gymapi.Vec3(2, -self.field_width+((i+1)/2)*delta, 0.01)
            if i % 2 == 1: robot_pose[i].r = gymapi.Quat(0, 0, 1, 0)

        # TODO: create force sensors

        # define start pose for ball
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.1)

        # create environments
        self.envs = []
        self.actor_indices = []    # Only robots

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            field_handle = self.gym.create_actor(env_ptr, field_asset, gymapi.Transform(), "field", i, 0, 0)
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_init_pose, "ball", i, 1, 0)
            # print("ball handle: ", ball_handle, "actor index: ", actor_index, "env ptr: ", env_ptr)
            actor_index = self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM)
            self.actor_indices.append(actor_index)

            # Create robomaster actors
            for j in range(self.num_robots):
                rm_handle = self.gym.create_actor(env_ptr, robot_asset, robot_pose[j], "rm" + "_" + str(j), i, 2**(j+1), 0)
                actor_index = self.gym.get_actor_index(env_ptr, rm_handle, gymapi.DOMAIN_SIM)
                # print("robot", j, "handle: ", rm_handle, "actor index: ", actor_index, "env ptr: ", env_ptr)
                self.gym.set_actor_dof_properties(env_ptr, rm_handle, robot_dof_props)
                self.actor_indices.append(actor_index)

                # self.gym.enable_actor_dof_force_sensors(env_ptr, rm_handle)

        self.actor_indices = to_torch(self.actor_indices, device=self.device).to(dtype=torch.int32)
        
        rm_handle = 2
        self.num_dof_per_env = self.gym.get_env_dof_count(env_ptr)
        self.num_dof_per_robot = self.gym.get_actor_dof_count(env_ptr, rm_handle)

        self.wheel_dof_indices_in_env = []   # 4个轮子的dof index, length = 4 * num_robots
        front_left_wheel_dof_index = self.gym.find_actor_dof_handle(env_ptr, rm_handle, "front_left_wheel_joint")
        front_right_wheel_dof_index = self.gym.find_actor_dof_handle(env_ptr, rm_handle, "front_right_wheel_joint")
        rear_left_wheel_dof_index = self.gym.find_actor_dof_handle(env_ptr, rm_handle, "rear_left_wheel_joint")
        rear_right_wheel_dof_index = self.gym.find_actor_dof_handle(env_ptr, rm_handle, "rear_right_wheel_joint")
        for k in range(self.num_robots):
            self.wheel_dof_indices_in_env.extend([x + k * self.num_dof_per_robot for x in [front_left_wheel_dof_index, front_right_wheel_dof_index, rear_left_wheel_dof_index, rear_right_wheel_dof_index]])
        
        # print(self.gym.get_actor_dof_names(env_ptr, rm_handle))

    def compute_reward(self, actions):
        '''
        FIXME: 出现两个进球奖励，代码逻辑需要调整
        rew_buf = (num_env, )
        reset_buf = (num_env, ) 
        '''
        self.rew_buf[:], self.reset_buf[:], self.extras['win'], self.extras['lose'], self.extras['draw'] = compute_robot_reward(
            self.states_buf,
            self.num_envs,
            self.rew_scales,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.dt,
            self.field_length,
            self.field_width,
            self.num_robots,
            self.num_agent,
            actions,
            self.obs_buf,
        )
        # breakpoint()

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # states_buf: (num_envs, num_states)
        # num_states: num_robots * 12[Pos(3), Vel(3), Ori(3 rpy), AngularVel(3)] + 6[BallPos(3), BallVel(3)]
        self.states_buf[:] = compute_states(    
            self.robot_states,
            self.ball_states,
            self.num_envs,
        )
        # obs_buf: (num_envs*num_robots, num_obs)
        # num_obs:     
        #       - Pos(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
        #       - Vel(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
        #       - Ori(self[3 sin,cos,tan(yaw)])
        #       - Ori2Others(teammates[3*(num_agent-1)], opponents[3*num_agent], ball[3], GoalCenter[2*3])
        #       - Dist2Others(teammates[1*(num_agent-1)], opponents[1*num_agent], ball[1], GoalCenter[2])
        #       - DistBall&Goal[2]
        #       - TimeLeft[1]
        for robot_idx in range(self.num_robots):
            self.obs_buf[robot_idx*self.num_envs:(robot_idx+1)*self.num_envs, :] = compute_robot_observations(self.states_buf,
                                                                                                              self.num_envs,
                                                                                                              self.num_robots,
                                                                                                              self.num_agent,
                                                                                                              self.max_episode_length,
                                                                                                              self.progress_buf,
                                                                                                              self.field_length,
                                                                                                              self.field_width,
                                                                                                              robot_idx,
                                                                                                              self.num_obs,)

    def reset_idx(self, env_ids):
        """
        Reset environment with indices in env_idx. 
        Only used in post_physics_step() function.
        """
        # # Randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)
        env_ids_int32 = self.actor_indices.view(self.num_envs, self.num_robots+1)[env_ids].flatten().to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self.initial_root_states), 
                                                     gymtorch.unwrap_tensor(env_ids_int32), 
                                                     len(env_ids_int32))
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                         gymtorch.unwrap_tensor(self.default_dof_states),
        #                                         gymtorch.unwrap_tensor(env_ids_int32),
        #                                         len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        # actions.shape = [num_envs * num_robots, num_actions], stacked as followed:
        # {[(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env0),
        #  [(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env1),
        #  ... }

        self.actions = actions.clone().to(self.device) * self.action_scale
        wheel_vel = mecanum_tranform(self.actions, self.num_envs, self.device)   # (num_envs, 4*num_robots)
        wheel_vel = torch.clamp(wheel_vel, self.wheel_limits_lower, self.wheel_limits_upper)
        actions_target_tensor = torch.zeros((self.num_envs, self.num_dof_per_env), device=self.device)   # (num_envs, num_dof_per_env)
        actions_target_tensor[:, self.wheel_dof_indices_in_env] = wheel_vel
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_target_tensor))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.compute_observations()
        self.compute_reward(self.actions)

        # print("rpy: 0: ", self.states_buf[0, 6:9], "1: ", self.states_buf[0, 6+12:9+12], "2: ", self.states_buf[0, 6+2*12:9+2*12], "3: ", self.states_buf[0, 6+3*12:9+3*12])

        if self.viewer is not None and self.cfg['debug_viz']:
            self.gym.clear_lines(self.viewer)
            for env_idx, env_ptr in enumerate(self.envs):
                for robot_idx in range(self.num_robots):
                    robot_coordinate = self.states_buf[env_idx, robot_idx * 12: robot_idx * 12 + 3].cpu().numpy()
                    self._add_robot_lines(env_ptr, robot_coordinate, robot_idx)

    def _add_robot_lines(self, env_ptr, coordinate, idx, radius=0.2):
        lines = []
        for height in range(3):
            for angle in range(360):
                theta1 = np.radians(angle)
                theta2 = np.radians(angle + 1)
                begin_point = [coordinate[0] + radius * np.cos(theta1),
                               coordinate[1] + radius * np.sin(theta1),
                               coordinate[2] + height * 0.01]
                end_point = [coordinate[0] + radius * np.cos(theta2),
                             coordinate[1] + radius * np.sin(theta2),
                             coordinate[2] + height * 0.01]   
                lines.append(begin_point)
                lines.append(end_point)
        lines = np.array(lines, dtype=np.float32)

        if idx % 2 == 0:
            colors = np.array([[1/(idx+1), 0, 0]] * (len(lines) // 2), dtype=np.float32)
        else:
            colors = np.array([[0, 0, 1/idx]] * (len(lines) // 2), dtype=np.float32)

        self.gym.add_lines(self.viewer, env_ptr, len(lines) // 2, lines, colors)


    def _record_traj(self, record_path, record_freq):
        pass


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_states(
        robot_states,    # (num_envs, num_robots, 13)
        ball_states,     # (num_envs, 13)
        num_envs,
):
    # type: (Tensor, Tensor, int)->Tensor
    # input: position([0:3]), rotation([3:7], xyzw), linear velocity([7:10]), angular velocity([10:13])
    # return: (num_envs, num_states)
    #   num_states: num_robots * 12[Pos(3), Vel(3), Ori(3 rpy), AngularVel(3)] + 6[BallPos(3), BallVel(3)]
    rpy = get_euler_xyz(robot_states.reshape(-1, 13)[:, 3:7])
    states_buf = torch.cat((robot_states[..., 0:3],
                            robot_states[..., 7:10],
                            rpy[0].reshape(num_envs, -1).unsqueeze(-1),
                            rpy[1].reshape(num_envs, -1).unsqueeze(-1),
                            rpy[2].reshape(num_envs, -1).unsqueeze(-1),
                            robot_states[..., 10:]),
                            dim=-1).reshape(num_envs, -1)
    states_buf = torch.cat((states_buf, 
                            ball_states[:, 0:3], 
                            ball_states[:, 7:10]), 
                            dim=-1)
    return states_buf

@torch.jit.script
def compute_robot_reward(
        states_buf,
        num_envs,
        rew_scales,
        reset_buf,
        progress_buf,
        max_episode_length,
        dt,
        field_width,
        field_length,
        num_robots,
        num_agent,
        actions,
        obs_buf,
):
    # type: (Tensor, int, Dict[str, float], Tensor, Tensor, int, float, float, float, int, int, Tensor, Tensor)->Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    # Input: 
    # - states_buf: (num_envs, num_states)
    # -              num_states: num_robots * 12[Pos(3), Vel(3), Ori(3 rpy), AngularVel(3)] + 6[BallPos(3), BallVel(3)]
    # - num_envs: int
    # - rew_scales: Dict[str, float]
    # - reset_buf: (num_envs, )
    # - progress_buf: (num_envs, )
    # - max_episode_length: int
    # - dt: float
    # - field_width: float
    # - field_length: float
    # - num_robots: int
    # - num_agent: int
    # - actions: (num_envs * num_robots, num_actions)
    # - obs_buf: (num_envs*num_robots, num_obs)
    #       - Pos(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
    #       - Vel(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
    #       - Ori(self[3 sin,cos,tan(yaw)])
    #       - Ori2Others(teammates[3*(num_agent-1)], opponents[3*num_agent], ball[3], GoalCenter[2*3])
    #       - Dist2Others(teammates[1*(num_agent-1)], opponents[1*num_agent], ball[1], GoalCenter[2])
    #       - DistBall&Goal[2]
    #       - TimeLeft[1]   
    # Return: 
    #   rew_buf[num_envs], reset_buf[num_envs], win[num_envs], lose[num_envs], draw[num_envs]
    # TODO: 待完善
    device = states_buf.device

    num_robot_info = 12
    out_of_bounds_tolerance = 0.15
    tmp_ones = torch.ones_like(reset_buf).to(device)

    # reward
    win_reward = torch.where(states_buf[:, -6] > field_length, tmp_ones * rew_scales['scoring'], torch.zeros_like(reset_buf, dtype=torch.float))
    lose_penalty = torch.where(states_buf[:, -6] < -field_length, tmp_ones * rew_scales['conceding'], torch.zeros_like(reset_buf, dtype=torch.float))
    draw_penalty = torch.where(progress_buf >= max_episode_length - 1, tmp_ones * rew_scales['draw_penalty_scale'], torch.zeros_like(reset_buf,
                                                                                                                                     dtype=torch.float))
    rew_out_of_bounds = torch.zeros(num_envs, dtype=torch.float).to(device)
    for i in range(num_agent):
        rew_out_of_bounds += torch.where(states_buf[:, 2*i * num_robot_info] < -(field_width + out_of_bounds_tolerance), tmp_ones, torch.zeros_like(reset_buf, dtype=torch.float))
        rew_out_of_bounds += torch.where(states_buf[:, 2*i * num_robot_info] > (field_width + out_of_bounds_tolerance), tmp_ones, torch.zeros_like(reset_buf, dtype=torch.float))
    
    rew_vel_forward = torch.zeros(num_envs, dtype=torch.float).to(device)
    for i in range(num_agent):
        rew_vel_forward += states_buf[:, 2*i * num_robot_info + 3]

    rew_vel_to_ball = torch.zeros(num_envs, dtype=torch.float).to(device)
    # for i in range(num_agent): 
        

    rew_dist_to_ball = torch.zeros(num_envs, dtype=torch.float).to(device)
    # for i in range(num_agent):

    


    dense_reward = rew_out_of_bounds * rew_scales['out_of_bounds'] + \
                   rew_vel_forward * rew_scales["vel_forward"] + \
                   rew_vel_to_ball * rew_scales["vel_to_ball"] + \
                   rew_dist_to_ball * rew_scales["dist_to_ball"]

    total_reward = win_reward + lose_penalty + draw_penalty + dense_reward * rew_scales['dense_reward_scale']

    # reset
    out_of_bounds = torch.zeros(num_envs, dtype=torch.bool).to(device)
    for i in range(num_robots):
        out_of_bounds |= states_buf[:, i * num_robot_info] < -(field_width + out_of_bounds_tolerance)
        out_of_bounds |= states_buf[:, i * num_robot_info] > (field_width + out_of_bounds_tolerance)
    # reset = torch.where(out_of_bounds, tmp_ones, reset_buf)
    # reset = torch.where((win_reward > 0) | (lose_penalty < 0), tmp_ones, reset)
    # reset = torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset)
    # # 防止顶翻，可以不考虑
    # # for i in range(num_robots):
    # #     reset = torch.where((states_buf[:, 7+i*num_robot_info] < 5.8) | (states_buf[:, 7+i*num_robot_info] > 0.4), tmp_ones, reset)


    reset = []
    reset_info = ['out_of_bounds', 'win', 'lose', 'draw']
    reset.append(torch.where(out_of_bounds, tmp_ones, reset_buf))
    reset.append(torch.where(win_reward > 0, tmp_ones, reset[-1]))
    reset.append(torch.where(lose_penalty < 0, tmp_ones, reset[-1]))
    reset.append(torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset[-1]))
    # 防止顶翻，可以不考虑
    # for i in range(num_robots):
    #     reset.append(torch.where((states_buf[:, 7+i*num_robot_info] < 5.8) | (states_buf[:, 7+i*num_robot_info] > 0.4), tmp_ones, reset[-1]))
    for i in range(len(reset)):
        if i == 0 and torch.sum(reset[0] - reset_buf) > 0:
            print(f"{reset_info[0]}: {torch.sum(reset[0] - reset_buf)}") 
        elif torch.sum(reset[i] - reset[i-1]) > 0:
            print(f"{reset_info[i]}: {torch.sum(reset[i] - reset[i-1])}")

    return total_reward, reset[-1], win_reward>0, lose_penalty<0, progress_buf>=max_episode_length-1

@torch.jit.script
def compute_robot_observations(
    states_buf, 
    num_envs,
    num_robots,
    num_agent,
    max_episode_length,
    progress_buf,
    field_length,
    field_width,
    robot_idx,
    num_obs,
):
    # type: (Tensor, int, int, int, int, Tensor, float, float, int, int)->Tensor
    # Input:
    # - states_buf: (num_envs, num_states)
    #     num_states: num_robots * 12[Pos(3), Vel(3), Ori(3 rpy), AngularVel(3)] + 6[BallPos(3), BallVel(3)]
    # - num_envs: int
    # - num_robots: int
    # - num_agent: int
    # - max_episode_length: int
    # - progress_buf: (num_envs, )
    # - field_length: float
    # - field_width: float
    # - robot_idx: int
    # Return:
    # - obs_buf: (num_envs, num_robots, num_obs)
    #       - Pos(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
    #       - Vel(self[2], teammates[2*(num_agent-1)], opponents[2*num_agent], ball[2])
    #       - Ori(self[3 sin,cos,tan(yaw)])
    #       - Ori2Others(teammates[3*(num_agent-1)], opponents[3*num_agent], ball[3], GoalCenter[2*3])
    #       - Dist2Others(teammates[1*(num_agent-1)], opponents[1*num_agent], ball[1], GoalCenter[2])
    #       - DistBall&Goal[2]
    #       - TimeLeft[1]
    device = states_buf.device
    num_robot_info = 12
    self_goal_center = torch.tensor([-field_length, 0.0], device=device, dtype=torch.float).repeat(num_envs, 1) 
    oppo_goal_center = torch.tensor([field_length, 0.0], device=device, dtype=torch.float).repeat(num_envs, 1) 
    if robot_idx % 2 == 0:
        teammate_idx = torch.arange(num_agent, device=device, dtype=torch.int) * 2
        teammate_idx = teammate_idx[teammate_idx != robot_idx]
        opponent_idx = torch.arange(num_agent, device=device, dtype=torch.int) * 2 + 1
    else: 
        teammate_idx = torch.arange(num_agent, device=device, dtype=torch.int) * 2 + 1
        teammate_idx = teammate_idx[teammate_idx != robot_idx]
        opponent_idx = torch.arange(num_agent, device=device, dtype=torch.int) * 2
    self_state = states_buf[:, robot_idx * num_robot_info: (robot_idx + 1) * num_robot_info]   # (num_envs, num_robot_info)
    teammate_state = [states_buf[:, idx * num_robot_info: (idx + 1) * num_robot_info] for idx in teammate_idx]
    teammate_state = torch.stack(teammate_state, dim=1)   # (num_envs, (num_agent-1), num_robot_info)
    opponent_state = [states_buf[:, idx * num_robot_info: (idx + 1) * num_robot_info] for idx in opponent_idx]
    opponent_state = torch.stack(opponent_state, dim=1)   # (num_envs, num_agent, num_robot_info)
    ball_state = states_buf[:, -6:]    # (num_envs, 6)

    obs_pos = torch.cat((self_state[:, 0:2], teammate_state[..., 0:2].reshape(num_envs, -1), opponent_state[..., 0:2].reshape(num_envs, -1), ball_state[:, 0:2], self_goal_center, oppo_goal_center), dim=-1)   # num_envs, 14
    obs_vel = torch.cat((self_state[:, 3:5], teammate_state[..., 3:5].reshape(num_envs, -1), opponent_state[..., 3:5].reshape(num_envs, -1), ball_state[:, 3:5]), dim=-1)  # num_envs, 10
    obs_ori = torch.cat((torch.sin(self_state[:, 8]).unsqueeze(-1), torch.cos(self_state[:, 8]).unsqueeze(-1), torch.tan(self_state[:, 8]).unsqueeze(-1)), dim=-1)      # num_envs, 3
    vec2others = obs_pos[:, 2:] - self_state[:, 0:2].repeat(1, obs_pos[:, 2:].shape[1] // 2)   # num_envs, 12
    angles = torch.atan2(vec2others[:, 1::2], vec2others[:, 0::2])
    obs_ori2others = torch.cat((torch.sin(angles), torch.cos(angles), torch.tan(angles)), dim=1)   # num_envs, 18
    obs_dist2others = torch.norm(vec2others.view(num_envs, -1, 2), dim=-1)  # num_envs, 6
    obs_distball2goal = torch.cat((torch.norm(ball_state[:, 0:2] - self_goal_center, dim=-1).unsqueeze(-1), 
                               torch.norm(ball_state[:, 0:2] - oppo_goal_center, dim=-1).unsqueeze(-1)), dim=-1)   # num_envs, 2
    obs_time_left = (max_episode_length - progress_buf).unsqueeze(-1)   # num_envs, 1

    # print('obs_pos: ', obs_pos.shape)
    # print('obs_vel: ', obs_vel.shape)
    # print('obs_ori: ', obs_ori.shape)
    # print('obs_ori2others: ', obs_ori2others.shape)
    # print('obs_dist2others: ', obs_dist2others.shape)
    # print('obs_distball2goal: ', obs_distball2goal.shape)
    # print('obs_time_left: ', obs_time_left.shape)

    obs = torch.cat((obs_pos, obs_vel, obs_ori, obs_ori2others, obs_dist2others, obs_distball2goal, obs_time_left), dim=-1)
    return obs