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
MAX_SPEED = np.deg2rad(1.5)

############################## YOUR CODE GOES HERE ####################################
#######################################################################################
# UA: Код для обчислення кінематики має бути тут               #
# EN: Your code for IK and drawing goes here.                                         #

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
    joint_positions = np.vstack([base_position, base_position + np.cumsum(
        np.stack([link_lengths * np.cos(cumulative_angles),
                  link_lengths * np.sin(cumulative_angles)], axis=-1), axis=0)])

    end_effector_pos = joint_positions[-1]
    vecs = end_effector_pos - joint_positions[:-1]

    jacobian_matrix = np.zeros((2, len(link_lengths)))
    jacobian_matrix[0, :] = -vecs[:, 1]
    jacobian_matrix[1, :] = vecs[:, 0]

    return jacobian_matrix

def null_space_projection(J):
    """
    Compute the projection matrix onto the null space of the Jacobian.
    """
    J_pinv = np.linalg.pinv(J)
    return np.eye(J.shape[1]) - J_pinv @ J

def inverse_kinematics(target_position, arm: RoboticArm, initial_angles_guess=None, secondary_objective=None):
    """
    Perform inverse kinematics to find joint angles for a target position.
    """
    max_iterations = 200
    tolerance = 0.01
    damping = 0.8
    alpha_secondary = 0.1

    current_angles = np.array(initial_angles_guess if initial_angles_guess is not None else arm.get_angles())

    for _ in range(max_iterations):
        current_pos = forward_kinematics(arm, current_angles)
        error = target_position - current_pos

        if np.linalg.norm(error) < tolerance:
            break

        J = jacobian(arm, current_angles)
        J_T = J.T
        inv_term = np.linalg.inv(J @ J_T + np.eye(2) * (damping**2))
        delta_theta_primary = J_T @ inv_term @ error

        if secondary_objective is not None:
            delta_theta_secondary = alpha_secondary * null_space_projection(J) @ secondary_objective
            current_angles += delta_theta_primary + delta_theta_secondary
        else:
            current_angles += delta_theta_primary

    return current_angles

def manipulability_gradient(arm: RoboticArm, angles):
    """
    Compute the gradient of manipulability for the robotic arm.
    """
    epsilon = 1e-6
    J = jacobian(arm, angles)
    manipulability = np.sqrt(np.linalg.det(J @ J.T))

    grad = np.zeros(len(angles))
    for i in range(len(angles)):
        angles_plus = angles.copy()
        angles_plus[i] += epsilon
        manipulability_plus = np.sqrt(np.linalg.det(jacobian(arm, angles_plus) @ jacobian(arm, angles_plus).T))
        grad[i] = (manipulability_plus - manipulability) / epsilon

    return grad

def joint_limit_avoidance_gradient(angles, joint_limits, weight=1.0):
    """
    Compute the gradient to avoid joint limits.
    """
    grad = -weight * (angles - np.mean(joint_limits, axis=1)) / (np.ptp(joint_limits, axis=1) ** 2)
    return grad

def interpolate_points(point1, point2, num_steps):
    """
    Generate intermediate points between two points for smoother transitions.
    """
    return np.linspace(point1, point2, num_steps, endpoint=False)[1:]

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
    controller = ArmController(arm, max_velocity=MAX_SPEED)
    arm.start_drawing()

    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

    running = True
    target_point = np.array([5., 5.])
    start = 0
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # The user closed the window or pressed escape
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

            ############################## YOUR CODE GOES HERE ####################################
            #######################################################################################
            # UA: Код для обчислення зворотної кінематики та малювання має бути тут               #
            # EN: Your code for IK and drawing goes here.                                         #
            #######################################################################################
            #######################################################################################
            # controller.move_to_angles([your predicted angles])

            pass

            #######################################################################################
            #######################################################################################

            # Make Box2D simulate the physics of our world for one step.
            ctx.world.Step(TIME_STEP, 10, 10)
            # Flip the screen and try to keep at the target FPS
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()