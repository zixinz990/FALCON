import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
from functools import partial

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

import pinocchio as pin
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy

from termcolor import colored
from sim2real.utils.arm_ik.robot_arm_ik_g1_23dof import G1_29_ArmIK_NoWrists


class JointControlGUI:
    def __init__(self, policy):
        self.policy = policy
        self.root = tk.Tk()
        self.root.title("FALCON Joint Control")
        self.root.geometry("500x700")

        # Create a scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Waist control
        tk.Label(
            self.scrollable_frame,
            text="--- Waist Control ---",
            font=("Helvetica", 12, "bold"),
        ).pack(pady=10)

        # Create 3 sliders for the 3 waist DOFs
        waist_labels = ["Waist Yaw", "Waist Roll", "Waist Pitch"]
        self.waist_sliders = []
        for i in range(3):
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(fill="x", padx=10, pady=2)
            tk.Label(frame, text=waist_labels[i], width=25, anchor="w").pack(
                side="left"
            )

            # Range: +/- 1.5 radians (~85 degrees)
            slider = tk.Scale(
                frame,
                from_=-1.5,
                to=1.5,
                resolution=0.01,
                orient="horizontal",
                length=250,
                command=partial(self.update_waist, i),
            )
            # Set initial value from policy
            slider.set(self.policy.waist_dofs_command[0, i])
            slider.pack(side="right")
            self.waist_sliders.append(slider)

        # Upper body control
        upper_body_labels = [
            "Left Shoulder Pitch",
            "Left Shoulder Roll",
            "Left Shoulder Yaw",
            "Left Elbow",
            "Left Wrist Roll",
            "Left Wrist Pitch",
            "Left Wrist Yaw",
            "Right Shoulder Pitch",
            "Right Shoulder Roll",
            "Right Shoulder Yaw",
            "Right Elbow",
            "Right Wrist Roll",
            "Right Wrist Pitch",
            "Right Wrist Yaw",
        ]
        tk.Label(
            self.scrollable_frame,
            text="--- Upper Body Control ---",
            font=("Helvetica", 12, "bold"),
        ).pack(pady=10)

        self.upper_sliders = []
        # Create a slider for each upper body joint
        for i in range(self.policy.num_upper_dofs):
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(fill="x", padx=10, pady=2)
            tk.Label(frame, text=upper_body_labels[i], width=25, anchor="w").pack(
                side="left"
            )

            # Range: +/- 2.5 radians (~140 degrees) - adjusted for wider arm range
            slider = tk.Scale(
                frame,
                from_=-2.5,
                to=2.5,
                resolution=0.01,
                orient="horizontal",
                length=250,
                command=partial(self.update_upper, i),
            )

            # Set initial value from current policy state
            initial_val = self.policy.ref_upper_dof_pos[0, i]
            slider.set(initial_val)
            slider.pack(side="right")
            self.upper_sliders.append(slider)

        tk.Label(
            self.scrollable_frame,
            text="Note: Disable 'use_upper_body_controller' in YAML\nto prevent IK from overwriting these values.",
            fg="red",
        ).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_waist(self, index, value):
        # Update the waist command in the policy
        self.policy.waist_dofs_command[0, index] = float(value)

    def update_upper(self, index, value):
        # Update the upper body reference position in the policy
        self.policy.ref_upper_dof_pos[0, index] = float(value)

    def on_closing(self):
        self.root.destroy()
        os._exit(0)  # Force exit to kill the policy thread

    def run(self):
        self.root.mainloop()


class LocoManipPolicy(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=0.25):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        # Initialize with default angles
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = self.config.get(
            "residual_upper_body_action", False
        )

        self.upper_body_controller = None
        if self.config.get("use_upper_body_controller", False):
            self.init_upper_body_controller()

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command

        return current_obs_dict

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] == "g1_29dof":
            self.upper_body_controller = G1_29_ArmIK_NoWrists(
                Unit_Test=False, Visualization=False, robot_config=self.config
            )
        else:
            self.logger.error("Unsupported robot type: %s", self.config["ROBOT_TYPE"])
        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        # Initialize waypoints
        self.degrees = -0
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array(
            [
                [np.cos(-self.theta), -np.sin(-self.theta), 0],
                [np.sin(-self.theta), np.cos(-self.theta), 0],
                [0, 0, 1],
            ]
        )
        self.EE_right_R = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), 0],
                [np.sin(self.theta), np.cos(self.theta), 0],
                [0, 0, 1],
            ]
        )
        self.EE_left_x = 0.30
        self.EE_right_x = 0.30
        self.EE_left_y = 0.13
        self.EE_right_y = -0.13
        self.EE_left_z = 0.08
        self.EE_right_z = 0.08
        self.update_waypoints()
        # Initialize external force
        self.EE_efrc_L = np.array([0, 0, 0, 0, 0, 0])
        self.EE_efrc_R = np.array([0, 0, 0, 0, 0, 0])
        # Initialize interpolated positions and orientations
        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation,
        )

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        # import ipdb; ipdb.set_trace()
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        # WBC actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data

        # Apply upper body controller (IK)
        # NOTE: If you are using the GUI sliders, this block might overwrite your slider values!
        # You should set "use_upper_body_controller: False" in your YAML config to use the GUI fully.
        if self.upper_body_controller:
            # Control upper qpos and tau
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
            # Zero out wrist joints
            wrist_joint_indices = [19, 20, 21, 26, 27, 28]
            for idx in wrist_joint_indices:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = scaled_policy_action + self.default_dof_angles
        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(
                q_target[0],
                self.motor_pos_lower_limit_list,
                self.motor_pos_upper_limit_list,
            )

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs]
        )

    def update_waypoints(self):
        self.waypoints_left = [
            pin.SE3(
                self.EE_left_R.astype(np.float64),
                np.array([self.EE_left_x, self.EE_left_y, self.EE_left_z]),
            )
        ]
        self.waypoints_right = [
            pin.SE3(
                self.EE_right_R.astype(np.float64),
                np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z]),
            )
        ]

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green")
            )
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green")
            )
        elif keycode == "m":
            self.lin_vel_command[:, 0] = -1.0
            self.logger.info(
                colored(f"lin_vel_command: {self.lin_vel_command}", "green")
            )
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        if cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green")
            )
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green")
            )
        elif cur_key == "R1+up":
            self.EE_left_x += 0.05
            self.EE_right_x += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+down":
            self.EE_left_x -= 0.05
            self.EE_right_x -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+left":
            self.EE_left_y += 0.02
            self.EE_right_y -= 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "R1+right":
            self.EE_left_y -= 0.02
            self.EE_right_y += 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "X+up":
            self.EE_left_z += 0.05
            self.EE_right_z += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+down":
            self.EE_left_z -= 0.05
            self.EE_right_z -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+left":
            self.degrees -= 5
            self.theta = np.radians(self.degrees)
            self.EE_left_R = np.array(
                [
                    [np.cos(-self.theta), -np.sin(-self.theta), 0],
                    [np.sin(-self.theta), np.cos(-self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.EE_right_R = np.array(
                [
                    [np.cos(self.theta), -np.sin(self.theta), 0],
                    [np.sin(self.theta), np.cos(self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "X+right":
            self.degrees += 5
            self.theta = np.radians(self.degrees)
            self.EE_left_R = np.array(
                [
                    [np.cos(-self.theta), -np.sin(-self.theta), 0],
                    [np.sin(-self.theta), np.cos(-self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.EE_right_R = np.array(
                [
                    [np.cos(self.theta), -np.sin(self.theta), 0],
                    [np.sin(self.theta), np.cos(self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green")
            )
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green")
            )
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green")
            )
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green")
            )
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(
                colored(f"Debug kp level: {self.command_sender.kp_level}", "green")
            )

    def _handle_base_height_control(self, keycode):
        """Handle base height control."""
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        """Handle joystick base height control."""
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()
        # You can inspect these values changing as you move sliders
        # print(f"Base height command: {self.base_height_command}")
        # print(f"Waist dofs command: {self.waist_dofs_command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--config", type=str, default="config/g1/g1_29dof.yaml", help="config file"
    )
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError(
            "model_path must be provided either via --model_path argument or in config file"
        )

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )

    # --- GUI Setup ---
    # We initialize the GUI in the main thread and run the policy in a daemon thread.
    gui = JointControlGUI(policy)

    # Start the policy loop in a separate thread so it doesn't block the GUI
    policy_thread = threading.Thread(target=policy.run, daemon=True)
    policy_thread.start()

    # Start the GUI main loop
    gui.run()
