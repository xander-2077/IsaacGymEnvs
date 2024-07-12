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