from typing import List
import time

import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.plotter_graphs import get_drawing1
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color


# Increase for debugging!
MAX_SPEED = np.deg2rad(25.0)
            
            
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


# ------- Helper: safe draw to avoid buffer overflow -------
def safe_draw(arm_plotter: RoboticArmPlotter):
    """
    Call arm_plotter.draw() only if it won't overflow the internal buffer.
    This guards against AssertionError: Too many points.
    """
    try:
        num = getattr(arm_plotter, "num_points", None)
        mx = getattr(arm_plotter, "max_num_points", None)
        if num is None or mx is None:
            # Can't check, call normally but catch assertion
            arm_plotter.draw()
        else:
            # reserve 2 slots margin
            if num < mx - 2:
                arm_plotter.draw()
            else:
                # skip drawing to avoid assertion; render will still show existing strokes
                pass
    except AssertionError:
        # If library asserts, ignore to avoid crash — final image will still be rendered
        pass


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 1 - visualization")

    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[5, 4, 3],
        link_angles=[np.deg2rad(-20), np.deg2rad(140), np.deg2rad(180)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )

    controller = ArmController(arm, max_velocity=MAX_SPEED)
    drawing1: List[np.ndarray] = get_drawing1()

    # State flags and guards
    pen_down = False
    final_image_rendered = False

    drawing = drawing1

    # Move arm to initial start point before main loop
    start_point = np.copy(drawing[0][0])
    if start_point[1] > 13.0:
        start_point[1] = 13.0

    target_angles = inverse_kinematics(start_point, arm)
    controller.move_to_angles(target_angles)

    # Wait for controller to finish initial move while rendering
    while not controller.is_idle():
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                raise SystemExit
        # Use opaque fill for reliable rendering
        ctx.screen.fill((0, 0, 0))
        arm.render()
        controller.step()
        ctx.world.Step(TIME_STEP, 10, 10)
        pygame.display.flip()
        ctx.clock.tick(TARGET_FPS)

    # main loop variables
    running = True
    i_shape = 0
    i_point = 0
    state = 'MOVE_TO_START'

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            # Opaque clear to avoid alpha drawing issues
            ctx.screen.fill((0, 0, 0))

            # Render scene
            arm.render()
            controller.step()

            # Only call safe_draw while pen_down to avoid buffer growth each frame
            if pen_down:
                safe_draw(arm)

            if controller.is_idle():
                # If already finished, keep rendering final image
                if state == 'FINISHED':
                    ctx.world.Step(TIME_STEP, 10, 10)
                    pygame.display.flip()
                    ctx.clock.tick(TARGET_FPS)
                    continue

                # Completed all shapes
                if i_shape >= len(drawing):
                    arm.stop_drawing()
                    pen_down = False
                    if not final_image_rendered:
                        safe_draw(arm)  # final attempt to capture last strokes
                        final_image_rendered = True
                    state = 'FINISHED'
                    print("Drawing complete! Window will remain open showing final image. Press ESC or close window to exit.")
                    continue

                if state == 'MOVE_TO_START':
                    arm.stop_drawing()
                    pen_down = False

                    target_point = drawing[i_shape][i_point].copy()
                    if target_point[1] > 13.0:
                        target_point[1] = 13.0

                    target_angles = inverse_kinematics(target_point, arm)
                    controller.move_to_angles(target_angles)
                    state = 'DRAWING'

                elif state == 'DRAWING':
                    arm.start_drawing()
                    pen_down = True
                    safe_draw(arm)  # capture current point immediately

                    i_point += 1
                    if i_point >= len(drawing[i_shape]):
                        i_shape += 1
                        i_point = 0
                        state = 'MOVE_TO_START'
                    else:
                        target_point = drawing[i_shape][i_point].copy()
                        if target_point[1] > 13.0:
                            target_point[1] = 13.0
                        target_angles = inverse_kinematics(target_point, arm)
                        controller.move_to_angles(target_angles)

            # physics / display tick
            ctx.world.Step(TIME_STEP, 10, 10)
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    finally:
        pygame.quit()