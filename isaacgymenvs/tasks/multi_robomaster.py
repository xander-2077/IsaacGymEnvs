import numpy as np
import os, sys, time
import torch
from gym.spaces import Box

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.rm_utils import *

from pprint import pprint
import pdb

class MultiRoboMaster(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.num_agent = self.cfg["env"]["numAgent"]  # 单边的agent数量



        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        
    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.0  # default: 1/60.0
        sim_params.substeps = 2  # default: 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_position_iterations = 8  # default: 4
        sim_params.physx.num_velocity_iterations = 1  # default: 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.max_gpu_contact_pairs = 2920898
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # # If randomizing, apply once immediately on startup before the fist sim step
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file_rm = "urdf/robomaster/robot/robomaster.urdf"
        asset_file_ball = "urdf/robomaster/ball.urdf"
        asset_file_field = "urdf/robomaster/field.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file_rm = self.cfg["env"]["asset"].get("robot", asset_file_rm)
            asset_file_ball = self.cfg["env"]["asset"].get("ball", asset_file_ball)
            asset_file_field = self.cfg["env"]["asset"].get("field", asset_file_field)
        
        # load robomaster asset
        rm_options = gymapi.AssetOptions()
        rm_options.fix_base_link = False
        rm_options.collapse_fixed_joints = False
        rm_asset = self.gym.load_asset(self.sim, asset_root, asset_file_rm, rm_options)
        
        # load ball asset
        ball_options = gymapi.AssetOptions()   # need to tuned
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file_ball, ball_options)

        # load field asset
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(self.sim, asset_root, asset_file_field, field_options)

        # set robomaster dof properties
        self.num_rm_dof = self.gym.get_asset_dof_count(rm_asset)

        rm_dof_props = self.gym.get_asset_dof_properties(rm_asset)
        rm_gripper_limits = []

        for i in range(rm_dof_props.shape[0]):
            if rm_dof_props[i]['hasLimits']:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_EFFORT
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0
                rm_gripper_limits.append([rm_dof_props[i]['lower'], rm_dof_props[i]['upper'], rm_dof_props[i]['effort']])

            elif rm_dof_props[i]['velocity'] < 1e2:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_VEL
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 500.0  # need to tuned
                self.rm_wheel_vel_limit = rm_dof_props[i]['velocity']  # max velocity
            else:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_NONE
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0

        self.rm_gripper_limits = torch.tensor(rm_gripper_limits, device=self.device)   # lower, upper, max effort

        # define start pose for robomaster
        rm_pose = [gymapi.Transform() for i in range(4)]
        rm_pose[0].p = gymapi.Vec3(-2, -2, 0.01)   # TODO: change initial position
        rm_pose[1].p = gymapi.Vec3(-2, 2, 0.01)
        rm_pose[2].p = gymapi.Vec3(2, -2, 0.01)
        rm_pose[3].p = gymapi.Vec3(2, 2, 0.01)
        rm_pose[2].r = gymapi.Quat(0, 0, 1, 0)
        rm_pose[3].r = gymapi.Quat(0, 0, 1, 0)

        # define start pose for ball
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.1)

        # create environments
        self.envs = []
        self.rm_handles = {}    # {env_ptr: [rm_handle1, rm_handle2, rm_handle3, rm_handle4]}
        self.ball_handles = {}  # {env_ptr: ball_handle}
        self.actor_index_in_sim = {}    # {env_ptr: [rm1_index, rm2_index, rm3_index, rm4_index, ball_index]}
        self.wheel_dof_handles = {}    # {env_ptr: [[front_left_wheel_dof, front_right_wheel_dof, rear_left_wheel_dof, rear_right_wheel_dof], ...]}

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.create_actor(
                env_ptr, field_asset, gymapi.Transform(), "field", i, 0, 0
            )

            # Create robomaster actors
            self.rm_handles[env_ptr] = []
            self.actor_index_in_sim[env_ptr] = []
            self.wheel_dof_handles[env_ptr] = []

            for j in range(self.num_agent * 2):
                rm_handle = self.gym.create_actor(
                    env_ptr, rm_asset, rm_pose[j], "rm" + "_" + str(j), i, 2**(j+1), 0
                )
                self.gym.set_actor_dof_properties(env_ptr, rm_handle, rm_dof_props)
                self.rm_handles[env_ptr].append(rm_handle)

                self.actor_index_in_sim[env_ptr].append(self.gym.get_actor_index(env_ptr, rm_handle, gymapi.DOMAIN_SIM))

                front_left_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "front_left_wheel_joint"
                )
                front_right_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "front_right_wheel_joint"
                )
                rear_left_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "rear_left_wheel_joint"
                )
                rear_right_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "rear_right_wheel_joint"
                )

                self.wheel_dof_handles[env_ptr].append(
                    [
                        front_left_wheel_dof,
                        front_right_wheel_dof,
                        rear_left_wheel_dof,
                        rear_right_wheel_dof,
                    ]
                )

            ball_handle = self.gym.create_actor(
                env_ptr, ball_asset, ball_init_pose, "ball", i, 1, 0
            )
            self.ball_handles[env_ptr] = ball_handle

            self.actor_index_in_sim[env_ptr].append(self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM))

            self.envs.append(env_ptr)

            self.wheel_dof_handles_per_env = torch.tensor(self.wheel_dof_handles[self.envs[0]]).reshape(self.num_agent * 2 * 4, )   # tensor([  0,  16,  33,  49,  66,  82,  99, 115, 132, 148, 165, 181, 198, 214, 231, 247])

            self.num_dof_per_env = self.gym.get_env_dof_count(self.envs[0])


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def refresh_state_dict(self):
        self.state_dict['r0'] = self.state_buf[..., :self.num_info_per_robot]
        self.state_dict['r1'] = self.state_buf[..., self.num_info_per_robot:2*self.num_info_per_robot]
        self.state_dict['b0'] = self.state_buf[..., 2*self.num_info_per_robot:3*self.num_info_per_robot]
        self.state_dict['b1'] = self.state_buf[..., 3*self.num_info_per_robot:4*self.num_info_per_robot]
        self.state_dict['ball_pos'] = self.state_buf[..., 4*self.num_info_per_robot:4*self.num_info_per_robot+2]
        self.state_dict['ball_vel'] = self.state_buf[..., 4*self.num_info_per_robot+2:4*self.num_info_per_robot+4]
        self.state_dict['goal_r'] = self.state_buf[..., 4*self.num_info_per_robot+4:4*self.num_info_per_robot+6]
        self.state_dict['goal_b'] = self.state_buf[..., 4*self.num_info_per_robot+6:4*self.num_info_per_robot+8]

    def compute_reward(self, actions):
        '''
        Get reward_buf, reset_buf 
        # FIXME: 出现两个进球奖励，代码逻辑需要调整
        self.reward_buf = torch.zeros((self.num_env, 2))
        self.reset_buf = torch.zeros(self.num_env) 
        '''
        self.reward_buf.zero_()
        for env_idx in range(self.num_env):
            # Scoring or Conceding
            if self.state_dict['ball_pos'][env_idx][0] < self.state_dict['goal_r'][env_idx][0]:
                self.reward_buf[env_idx][0] += -1 * self.args.reward_conceding
                self.reward_buf[env_idx][1] += 1 * self.args.reward_scoring
                self.reset_buf[env_idx] = 1
            elif self.state_dict['ball_pos'][env_idx][0] > self.state_dict['goal_b'][env_idx][0]:
                self.reward_buf[env_idx][0] += 1 * self.args.reward_scoring
                self.reward_buf[env_idx][1] += -1 * self.args.reward_conceding
                self.reset_buf[env_idx] = 1
            
            # Out of boundary
            for robot in ['r0', 'r1']:
                if self.state_dict[robot][env_idx][1] < self.state_dict['goal_r'][env_idx][0]-1 or self.state_dict[robot][env_idx][1] > self.state_dict['goal_b'][env_idx][0]+1:
                    self.reward_buf[env_idx][0] += -1 * self.args.reward_out_of_boundary
                    self.reset_buf[env_idx] = 1

            for robot in ['b0', 'b1']:
                if self.state_dict[robot][env_idx][1] < self.state_dict['goal_r'][env_idx][0]-1 or self.state_dict[robot][env_idx][1] > self.state_dict['goal_b'][env_idx][0]+1:
                    self.reward_buf[env_idx][1] += 1 * self.args.reward_out_of_boundary
                    self.reset_buf[env_idx] = 1

            # Velocity to ball
            for robot in ['r0', 'r1', 'b0', 'b1']:
                dir_vec = self.state_dict['ball_pos'][env_idx] - self.state_dict[robot][env_idx][1:3]
                norm_dir_vec = dir_vec / dir_vec.norm()
                vel_towards_ball = torch.dot(self.state_dict[robot][env_idx][3:5], norm_dir_vec).item()
                if robot in ['r0', 'r1']:
                    self.reward_buf[env_idx][0] += vel_towards_ball * self.args.reward_vel_to_ball
                else:
                    self.reward_buf[env_idx][1] += vel_towards_ball * self.args.reward_vel_to_ball

            # Velocity forward
            self.reward_buf[env_idx][0] += (self.state_dict['r0'][env_idx][3] + self.state_dict['r1'][env_idx][3]) * self.args.reward_vel
            self.reward_buf[env_idx][1] += - (self.state_dict['b0'][env_idx][3] + self.state_dict['b1'][env_idx][3]) * self.args.reward_vel

            # Close to ball
            for robot in ['r0', 'r1', 'b0', 'b1']:
                dist_to_ball = torch.norm(self.state_dict['ball_pos'][env_idx] - self.state_dict[robot][env_idx][1:3]).item()
                if dist_to_ball < 0.3:
                    if robot in ['r0', 'r1']:
                        self.reward_buf[env_idx][0] += (0.5 - dist_to_ball) * self.args.reward_dist_to_ball
                    else:
                        self.reward_buf[env_idx][1] += (0.5 - dist_to_ball) * self.args.reward_dist_to_ball



            # TODO: add more rewards

        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # pprint(self.reward_buf)


    def compute_observations(self):
        self._refresh()

        num_state = self.num_state  # 36
        num_info_per_robot = self.num_info_per_robot  # 7

        # 每个env的观测值顺序为: Robot1, Robot2, Robot3, Robot4, Ball, GoalRed, GoalBlue
        
        # self.root_tensor: (num_env * num_actor, 13)
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_orientations = self.root_tensor[:, 3:7]   # xyzw
        self.root_angvels = self.root_tensor[:, 10:13]

        for i, env_ptr in enumerate(self.envs):
            for j, actor_index in enumerate(self.actor_index_in_sim[env_ptr]):
                if j < len(self.actor_index_in_sim[env_ptr]) - 1:
                    # Robot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1)
                    self.state_buf[i, j * self.num_info_per_robot] = j
                    self.state_buf[i, j * self.num_info_per_robot + 1 : j * self.num_info_per_robot + 3] = self.root_positions[actor_index][:-1]
                    self.state_buf[i, j * self.num_info_per_robot + 3 : j * self.num_info_per_robot + 5] = self.root_linvels[actor_index][:-1]
                    self.state_buf[i, j * self.num_info_per_robot + 5] = quaternion_to_yaw(self.root_orientations[actor_index], self.device)
                    self.state_buf[i, j * self.num_info_per_robot + 6] = self.root_angvels[actor_index][-1]
                else:
                    # Ball: BallPos(2), BallVel(2)
                    self.state_buf[i, j * self.num_info_per_robot : j * self.num_info_per_robot + 2] = self.root_positions[actor_index][:-1]
                    self.state_buf[i, j * self.num_info_per_robot + 2 : j * self.num_info_per_robot + 4] = self.root_linvels[actor_index][:-1]

            # Goal: GoalRed(2), GoalBlue(2)
            self.state_buf[i, j * self.num_info_per_robot + 4 : j * self.num_info_per_robot + 6] = torch.tensor([-4.5, 0.0])
            self.state_buf[i, j * self.num_info_per_robot + 6 : j * self.num_info_per_robot + 8] = torch.tensor([4.5, 0.0])


    def reset_idx(self, env_ids):
        """
        Reset environment with indices in env_idx. 
        Only used in post_physics_step() function.
        """
        indices_list = env_ids.tolist()
        envs = [self.envs[i] for i in indices_list]
        actor_indices = torch.tensor([self.actor_index_in_sim[env_ptr] for env_ptr in envs], dtype=torch.int32, device=self.device).flatten()
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self.saved_root_tensor), 
                                                     gymtorch.unwrap_tensor(actor_indices), 
                                                     len(actor_indices))
        
        # Clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device) * self.rm_wheel_vel_limit
        wheel_vel = mecanum_tranform(self.actions, self.num_envs, self.device)

        actions_target_tensor = torch.zeros((self.num_envs, self.num_dof_per_env), device=self.device)

        actions_target_tensor[:, self.wheel_dof_handles_per_env] = wheel_vel
        
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_target_tensor))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.refresh_state_dict()
        self.compute_reward(self.actions)

        # debug viz
        pass



#####################################################################
###=========================jit functions=========================###
#####################################################################