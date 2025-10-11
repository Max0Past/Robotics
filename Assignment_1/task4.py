from typing import List
import time

import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN)

from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color, ColorGreen


# Increase for debugging!
MAX_SPEED = np.deg2rad(2.0)

############################## YOUR CODE GOES HERE ####################################
#######################################################################################
# UA: Код для обчислення кінематики має бути тут                                        #
# EN: Your code for IK and drawing goes here.                                           #


def forward_kinematics(arm: RoboticArm, angles=None):
    """
    Calculate the end-effector position based on joint angles.
    """
    base_position = np.array(arm.joints[0].position)
    link_lengths = np.array(arm.link_lengths)
    angles = np.array(angles if angles is not None else arm.get_angles())

    cumulative_angles = np.cumsum(angles)
    deltas = np.stack([link_lengths * np.cos(cumulative_angles),
                       link_lengths * np.sin(cumulative_angles)], axis=-1)

    return base_position + deltas.sum(axis=0)


def jacobian(arm: RoboticArm, angles=None):
    """
    Compute the Jacobian matrix for the robotic arm.
    """
    angles = np.array(angles if angles is not None else arm.get_angles())
    link_lengths = np.array(arm.link_lengths)
    base_position = np.array(arm.joints[0].position)

    cumulative_angles = np.cumsum(angles)
    # The positions of the end of each link (which is the start of the next joint)
    joint_positions_vectors = np.cumsum(
        np.stack([link_lengths * np.cos(cumulative_angles),
                  link_lengths * np.sin(cumulative_angles)], axis=-1), axis=0)
    joint_positions = base_position + joint_positions_vectors

    # We need positions of joint origins, which includes the base
    joint_origins = np.vstack([base_position, joint_positions[:-1]])

    end_effector_pos = joint_positions[-1]
    vecs = end_effector_pos - joint_origins

    jacobian_matrix = np.zeros((2, len(link_lengths)))
    # The i-th column is the cross-product of (p_ee - p_i) with the z-axis
    jacobian_matrix[0, :] = -vecs[:, 1]
    jacobian_matrix[1, :] = vecs[:, 0]

    return jacobian_matrix


def null_space_projection(J):
    """
    Compute the projection matrix onto the null space of the Jacobian.
    """
    # Use pseudo-inverse for numerical stability
    J_pinv = np.linalg.pinv(J)
    return np.eye(J.shape[1]) - J_pinv @ J


def inverse_kinematics(target_position, arm: RoboticArm, initial_angles_guess=None, secondary_objective=None):
    """
    Perform inverse kinematics to find joint angles for a target position.
    Uses damped least squares for stability and allows a null-space secondary objective.
    NOTE: This is a full solver, good for testing but not for real-time control.
    """
    max_iterations = 200
    tolerance = 0.01
    damping = 0.1 # Reduced damping for the full solver
    alpha_primary = 0.5 # Learning rate
    alpha_secondary = 0.1

    current_angles = np.array(initial_angles_guess if initial_angles_guess is not None else arm.get_angles())

    for _ in range(max_iterations):
        current_pos = forward_kinematics(arm, current_angles)
        error = target_position - current_pos

        if np.linalg.norm(error) < tolerance:
            break

        J = jacobian(arm, current_angles)
        J_T = J.T
        
        # Damped Least Squares
        inv_term = np.linalg.inv(J @ J_T + np.eye(2) * (damping**2))
        delta_theta_primary = J_T @ inv_term @ error

        delta_theta = alpha_primary * delta_theta_primary

        if secondary_objective is not None:
            delta_theta_secondary = alpha_secondary * null_space_projection(J) @ secondary_objective
            delta_theta += delta_theta_secondary
        
        current_angles += delta_theta

    return current_angles


def manipulability_gradient(arm: RoboticArm, angles):
    """
    Compute the gradient of manipulability for the robotic arm.
    Numerical finite-difference approximation.
    """
    epsilon = 1e-6
    J = jacobian(arm, angles)
    # Ensure determinant is non-negative before sqrt
    det_val = np.linalg.det(J @ J.T)
    manipulability = np.sqrt(max(0.0, det_val))

    grad = np.zeros(len(angles))
    for i in range(len(angles)):
        angles_plus = angles.copy()
        angles_plus[i] += epsilon
        J_plus = jacobian(arm, angles_plus)
        det_val_plus = np.linalg.det(J_plus @ J_plus.T)
        manipulability_plus = np.sqrt(max(0.0, det_val_plus))
        grad[i] = (manipulability_plus - manipulability) / epsilon

    return grad


def joint_limit_avoidance_gradient(angles, joint_limits, weight=1.0):
    """
    Compute the gradient to avoid joint limits.
    joint_limits expected shape: (n_joints, 2) with [min, max] per joint.
    """
    joint_limits = np.array(joint_limits)
    mid = np.mean(joint_limits, axis=1)
    span = joint_limits[:, 1] - joint_limits[:, 0]
    # Avoid division by zero for fixed joints
    span[span == 0] = 1.0
    grad = -weight * (angles - mid) / (span ** 2)
    return grad


#######################################################################################
#######################################################################################


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 4 - visualization")
    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[2, 1, 1, 2, 1, 2],
        link_angles=[np.deg2rad(160), np.deg2rad(-80), np.deg2rad(130),
                     np.deg2rad(0), np.deg2rad(90), np.deg2rad(200)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    # The controller is not used in the fixed version, we set angles directly.
    # We keep it for the `step` method which can be useful.
    controller = ArmController(arm, max_velocity=MAX_SPEED)
    arm.start_drawing()

    pygame.font.init() 
    my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

    running = True
    target_point = np.array([5., 5.])

    # --- quick IK unit-test (prints to console) ---
    try:
        cur_angles = np.array(arm.get_angles())
        ee = forward_kinematics(arm, cur_angles)
        test_target = ee + np.array([0.5, -0.2])
        sol = inverse_kinematics(test_target, arm, initial_angles_guess=cur_angles)
        err = np.linalg.norm(forward_kinematics(arm, sol) - test_target)
        print(f"IK quick-test error: {err:.4f}")
    except Exception as e:
        print("IK quick-test failed:", e)

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    target_point = core.from_pix(np.array(event.dict['pos']))
                elif event.type == MOUSEBUTTONDOWN:
                    target_point = core.from_pix(np.array(event.dict['pos']))

            ctx.screen.fill((0, 0, 0, 0))

            text_surface = my_font.render(
                f"Target: {target_point[0]:.2f}, {target_point[1]:.2f}",
                True, (255, 255, 255), (0,0,0))
            ctx.screen.blit(text_surface, (40, 40))
            pygame.draw.circle(ctx.screen, ColorGreen, center=core.to_pix(target_point), radius=15)
            arm.render()

            #######################################################################################
            #            NEW, STABLE, VELOCITY-BASED IK CONTROL LOOP                              #
            #######################################################################################

            # 1) Get current state of the arm
            current_angles = np.array(arm.get_angles())
            current_pos = forward_kinematics(arm, current_angles)

            # 2) Calculate the error vector (desired end-effector velocity)
            error = target_point - current_pos
            distance_to_target = np.linalg.norm(error)
            
            # Only move if we are not already at the target
            if distance_to_target > 0.05: # tolerance
                
                # 3) Clamp target to reachable workspace
                link_lengths = np.array(arm.link_lengths)
                max_reach = link_lengths.sum()
                base_pos = np.array(arm.joints[0].position)
                dist_from_base = np.linalg.norm(target_point - base_pos)
                
                if dist_from_base > max_reach:
                    # Clamp the target to the edge of the reachable circle
                    vec_to_target = target_point - base_pos
                    target_point = base_pos + vec_to_target * (max_reach / dist_from_base)
                    # Recalculate error with clamped target
                    error = target_point - current_pos

                # 4) Compute Jacobian and Damped Least Squares for the primary task
                J = jacobian(arm, current_angles)
                damping = 0.2 # Damping factor for stability
                inv_term = np.linalg.inv(J @ J.T + np.eye(2) * (damping**2))
                delta_theta_primary = J.T @ inv_term @ error

                # 5) Compute secondary objective (null-space motion)
                # We want to maximize manipulability
                manipulability_grad = manipulability_gradient(arm, current_angles)

                # We could add other objectives like joint limit avoidance here
                # joint_limit_grad = joint_limit_avoidance_gradient(...)
                secondary_objective = manipulability_grad 
                
                # Project secondary objective onto the null space and scale it
                alpha_secondary = 0.5 # How strongly we pursue the secondary objective
                delta_theta_secondary = alpha_secondary * null_space_projection(J) @ secondary_objective

                # 6) Combine primary and secondary objectives
                delta_theta = delta_theta_primary + delta_theta_secondary
                
                # 7) Regulate speed: scale the angle change to not exceed MAX_SPEED
                delta_norm = np.linalg.norm(delta_theta)
                if delta_norm > MAX_SPEED:
                    delta_theta = delta_theta * (MAX_SPEED / delta_norm)

                # 8) Apply the calculated change in angles
                new_angles = current_angles + delta_theta
                arm.set_angles(new_angles.tolist())


            # Visualization helpers
            pygame.draw.line(ctx.screen, Color(200, 200, 0, 255), 
                             core.to_pix(current_pos), 
                             core.to_pix(target_point), 2)

            try:
                J = jacobian(arm, arm.get_angles())
                manip = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
                txt = my_font.render(f"Manipulability: {manip:.3f}", True, (255,255,255))
                ctx.screen.blit(txt, (40, 80))
            except Exception:
                pass

            #######################################################################################
            #######################################################################################

            ctx.world.Step(TIME_STEP, 10, 10)
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()