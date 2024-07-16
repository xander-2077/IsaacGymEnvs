import torch
import numpy as np


def local_pos(global_pos, robot_xy, rotation_matrix):
  
    rotated_translation = torch.matmul(
        rotation_matrix, (global_pos - robot_xy).unsqueeze(-1)
    )
    return rotated_translation

def mecanum_tranform(vel, num_env, device):
    '''
    Velocity limit:
    1. -3.5 <= v_x <= 3.5 m/s
    2. -3.5 <= v_y <= 3.5 m/s
    3. -5π/3 <= w <= 5π/3 rad/s

    RoboMaster EP Parameters:
    1. wheel radius: r = 0.05 m
    2. robot length: L = 0.30 m
    3. robot width: W = 0.30 m
    '''
    r = 0.05
    L = 0.30
    W = 0.30

    vel[..., 0] = torch.clamp(vel[..., 0], -3.5, 3.5)
    vel[..., 1] = torch.clamp(vel[..., 1], -3.5, 3.5)
    vel[..., 2] = torch.clamp(vel[..., 2], -5 * 3.1415926 / 3, 5 * 3.1415926 / 3)

    mecanum_vel = torch.zeros((*vel.shape[:-1], 4), dtype=torch.float32, device=device)

    mecanum_vel[..., 0] = vel[..., 0] - vel[..., 1] - vel[..., 2] * (L + W) / 2
    mecanum_vel[..., 1] = vel[..., 0] + vel[..., 1] + vel[..., 2] * (L + W) / 2
    mecanum_vel[..., 2] = vel[..., 0] + vel[..., 1] - vel[..., 2] * (L + W) / 2
    mecanum_vel[..., 3] = vel[..., 0] - vel[..., 1] + vel[..., 2] * (L + W) / 2
    
    return (mecanum_vel / r).reshape(num_env, -1)

def _t2n(x):
    return x.detach().cpu().numpy()

def transform_to_local(robot_state, other_state):
    '''
    将其他机器人的全局状态转换为当前机器人的局部坐标系
    '''
    # 获取当前机器人和其他机器人的位置和速度
    robot_pos = robot_state[1:3]
    robot_ori = robot_state[5]
    other_pos = other_state[1:3]
    other_vel = other_state[3:5]
    other_ori = other_state[5]
    other_angvel = other_state[6]

    # 计算相对位置和速度
    relative_pos = other_pos - robot_pos
    relative_vel = other_vel - robot_state[3:5]

    # 旋转到局部坐标系
    relative_pos_local = rotate_to_local(relative_pos, robot_ori)
    relative_vel_local = rotate_to_local(relative_vel, robot_ori)

    # 返回局部状态
    return torch.cat((other_state[0:1], relative_pos_local, relative_vel_local, torch.tensor([other_ori - robot_ori]), torch.tensor([other_angvel])))

def transform_position_to_local(robot_state, global_pos):
    '''
    将全局位置转换为当前机器人的局部坐标系
    '''
    robot_pos = robot_state[1:3]
    robot_ori = robot_state[5]
    relative_pos = global_pos - robot_pos
    return rotate_to_local(relative_pos, robot_ori)

def transform_velocity_to_local(robot_state, global_vel):
    '''
    将全局速度转换为当前机器人的局部坐标系
    '''
    robot_vel = robot_state[3:5]
    robot_ori = robot_state[5]
    relative_vel = global_vel - robot_vel
    return rotate_to_local(relative_vel, robot_ori)

def rotate_to_local(vector, orientation):
    '''
    将向量旋转到局部坐标系
    '''
    cos_ori = torch.cos(orientation)
    sin_ori = torch.sin(orientation)
    local_x = cos_ori * vector[0] + sin_ori * vector[1]
    local_y = -sin_ori * vector[0] + cos_ori * vector[1]
    return torch.tensor([local_x, local_y])