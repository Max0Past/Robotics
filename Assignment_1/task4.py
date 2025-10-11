from typing import List, Optional, Sequence
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


def forward_kinematics(arm, angles: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Повертає позицію кінцевого ефектора (2D) для заданих кутів
    arm: об'єкт з атрибутами joints (list з позицією бази), link_lengths, get_angles()
    """
    base_position = np.array(arm.joints[0].position, dtype=float)
    link_lengths = np.array(arm.link_lengths, dtype=float)
    angles = np.array(angles if angles is not None else arm.get_angles(), dtype=float)

    cumulative_angles = np.cumsum(angles)
    deltas = np.stack([
        link_lengths * np.cos(cumulative_angles),
        link_lengths * np.sin(cumulative_angles)
    ], axis=-1)

    return base_position + deltas.sum(axis=0)


def jacobian(arm, angles: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Обчислює 2 x n Якобіан для planar manipulator
    """
    angles = np.array(angles if angles is not None else arm.get_angles(), dtype=float)
    link_lengths = np.array(arm.link_lengths, dtype=float)
    base_position = np.array(arm.joints[0].position, dtype=float)

    cumulative_angles = np.cumsum(angles)
    joint_positions_vectors = np.cumsum(
        np.stack([link_lengths * np.cos(cumulative_angles),
                  link_lengths * np.sin(cumulative_angles)], axis=-1), axis=0)
    joint_positions = base_position + joint_positions_vectors

    joint_origins = np.vstack([base_position, joint_positions[:-1]])
    end_effector_pos = joint_positions[-1]
    vecs = end_effector_pos - joint_origins  # shape (n, 2)

    J = np.zeros((2, len(link_lengths)), dtype=float)
    # колонка i: крос-продукт (p_ee - p_i) x z_hat -> результат у площині: [-y, x]
    J[0, :] = -vecs[:, 1]
    J[1, :] = vecs[:, 0]
    return J


def null_space_projection_from_pinv(J: np.ndarray) -> np.ndarray:
    """
    Проєкція у нуль-простір використовуючи псевдоінверсу (повертає n x n матрицю)
    """
    J_pinv = np.linalg.pinv(J)
    return np.eye(J.shape[1]) - J_pinv @ J


def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """
    SVD-damped pseudoinverse (повертає n x m матрицю)
    Стійкіше біля сингулярностей ніж np.linalg.pinv з малим дампером
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S**2 + damping**2)
    return Vt.T @ np.diag(S_damped) @ U.T


def manipulability_value(J: np.ndarray) -> float:
    """Маніпулябільність: sqrt(det(J J^T)) (>=0)"""
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(0.0, det)))


def joint_limit_avoidance_gradient(angles: np.ndarray, joint_limits: Sequence[Sequence[float]], weight: float = 1.0) -> np.ndarray:
    """
    Градієнт для уникнення лімітів (симетрична функція, що штовхає кути до середини інтервалу)
    joint_limits: список [min, max] на кожен суглоб (радіани)
    """
    joint_limits = np.array(joint_limits, dtype=float)
    mid = np.mean(joint_limits, axis=1)
    span = joint_limits[:, 1] - joint_limits[:, 0]
    span[span == 0] = 1.0
    grad = -weight * (angles - mid) / (span ** 2)
    return grad


def clamp_per_joint(delta: np.ndarray, max_per_joint: np.ndarray) -> np.ndarray:
    """Обмежує зміни кутів по-суглобово для плавності руху"""
    return np.clip(delta, -max_per_joint, max_per_joint)


# -------------------------
# Головна інверсна кінематика
# -------------------------

def inverse_kinematics(target_position: Sequence[float],
                         arm,
                         joint_limits: Optional[Sequence[Sequence[float]]] = None,
                         initial_angles_guess: Optional[Sequence[float]] = None,
                         *,
                         max_iterations: int = 200,
                         tolerance: float = 5e-3,
                         base_damping: float = 1e-2,
                         alpha_primary: float = 0.35,
                         alpha_secondary: float = 0.08,
                         max_delta_deg_per_iter: float = 2.0) -> np.ndarray:
    """
    Інверсна кінематика з обмеженнями:
    - SVD-damped pseudoinverse
    - адаптивний дампер залежно від маніпулябільності
    - другорядна ціль: уникнення лімітів (проектується в нуль-простір)
    - per-joint clamp для плавності руху

    Повертає масив кутів (в радіанах)
    """
    target = np.array(target_position, dtype=float)
    current_angles = np.array(initial_angles_guess if initial_angles_guess is not None else arm.get_angles(), dtype=float)
    n = len(current_angles)
    joint_limits = np.array(joint_limits) if joint_limits is not None else np.vstack([[-np.pi, np.pi]] * n)
    max_delta_per_joint = np.full(n, np.deg2rad(max_delta_deg_per_iter))

    for it in range(max_iterations):
        current_pos = forward_kinematics(arm, current_angles)
        error = target - current_pos
        err_norm = np.linalg.norm(error)
        if err_norm < tolerance:
            break

        J = jacobian(arm, current_angles)  # (2, n)
        manip = manipulability_value(J)
        adapt_damping = base_damping + (1.0 / (manip + 1e-6)) * 0.01  # +eps для стабільності

        J_pinv = damped_pinv(J, adapt_damping)  # n x 2
        delta_theta_primary = J_pinv @ error  # n-vector

        delta = alpha_primary * delta_theta_primary

        # другорядне завдання: уникнення лімітів — тільки коли маніпулябільність не занадто мала
        if manip > 1e-6 and joint_limits is not None:
            sec_obj = joint_limit_avoidance_gradient(current_angles, joint_limits, weight=1.0)
            null_proj = np.eye(n) - J_pinv @ J
            delta_theta_secondary = null_proj @ sec_obj
            delta += alpha_secondary * delta_theta_secondary

        # обмежуємо по-суглобово для гладкості
        delta = clamp_per_joint(delta, max_delta_per_joint)
        current_angles = current_angles + delta

    return current_angles

#######################################################################################
#######################################################################################


if __name__ == "__main__":
    # --- 1. Налаштування сцени та руки ---
    ctx = core.RenderingContext("Task 4 - Simplified IK Solver")
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

    pygame.font.init()
    my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

    # --- 2. Основний цикл програми ---
    running = True
    try:
        while running:
            # Обробка подій (мишка, закриття вікна)
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            # Отримуємо поточне положення курсора
            mouse_pos = pygame.mouse.get_pos()
            target_point = core.from_pix(np.array(mouse_pos))

            # Використовуємо контролер для плавного руху до цілі
            if controller.is_idle():
                controller.move_to_angles(
                    inverse_kinematics(
                        target_position=target_point,
                        arm=arm,
                        joint_limits=np.deg2rad([[-180, 180]] * len(arm.link_lengths)),
                        initial_angles_guess=arm.get_angles()
                    )
                )

            controller.step()

            # --- 3. Рендеринг ---
            ctx.screen.fill((0, 0, 0, 0))

            # Відображення тексту та цілі
            text_surface = my_font.render(f"Target: {target_point[0]:.2f}, {target_point[1]:.2f}", 
                                          True, (255, 255, 255), (0, 0, 0))
            ctx.screen.blit(text_surface, (40, 40))
            pygame.draw.circle(ctx.screen, ColorGreen, center=core.to_pix(target_point), radius=15)

            # Рендеринг руки
            arm.render()

            # Оновлення екрану
            ctx.world.Step(TIME_STEP, 10, 10)
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)

    except KeyboardInterrupt:
        print("Програму зупинено.")
    pygame.quit()