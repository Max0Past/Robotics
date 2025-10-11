from typing import List, Optional, Sequence
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


# Збільшіть для налагодження!
MAX_SPEED = np.deg2rad(25.0)
            
            
############################## ВАШ КОД ТУТ ####################################
#######################################################################################
# UA: Код для обчислення кінематики має бути тут               #
# EN: Your code for IK and drawing goes here.                                         #

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
    Обчислює 2 x n Якобіан для планарного маніпулятора
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
    vecs = end_effector_pos - joint_origins  # форма (n, 2)

    J = np.zeros((2, len(link_lengths)), dtype=float)
    # колонка i: векторний добуток (p_ee - p_i) x z_hat -> результат у площині: [-y, x]
    J[0, :] = -vecs[:, 1]
    J[1, :] = vecs[:, 0]
    return J


def null_space_projection_from_pinv(J: np.ndarray) -> np.ndarray:
    """
    Проєкція у нуль-простір використовуючи псевдоінверсію
    """
    J_pinv = np.linalg.pinv(J)
    return np.eye(J.shape[1]) - J_pinv @ J


def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """
    SVD-заглушена псевдоінверсія (повертає n x m матрицю)
    Стійкіше біля сингулярностей ніж np.linalg.pinv з малим дампером
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S**2 + damping**2)
    return Vt.T @ np.diag(S_damped) @ U.T


def manipulability_value(J: np.ndarray) -> float:
    """Маніпулябельність: sqrt(det(J J^T)) (>=0)"""
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(0.0, det)))


def joint_limit_avoidance_gradient(angles: np.ndarray, joint_limits: Sequence[Sequence[float]], weight: float = 1.0) -> np.ndarray:
    """
    Градієнт для уникнення обмежень суглобів (симетрична функція, що штовхає кути до середини інтервалу)
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
    - SVD-заглушена псевдоінверсія
    - адаптивний дампер залежно від маніпулябельності
    - другорядна ціль: уникнення обмежень (проектується в нуль-простір)
    - по-суглобове обмеження для плавності руху

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
        delta_theta_primary = J_pinv @ error  # n-вектор

        delta = alpha_primary * delta_theta_primary

        # другорядне завдання: уникнення обмежень — тільки коли маніпулябельність не занадто мала
        if manip > 1e-6 and joint_limits is not None:
            sec_obj = joint_limit_avoidance_gradient(current_angles, joint_limits, weight=1.0)
            null_proj = np.eye(n) - J_pinv @ J
            delta_theta_secondary = null_proj @ sec_obj
            delta += alpha_secondary * delta_theta_secondary

        # обмежуємо по-суглобово для гладкості
        delta = clamp_per_joint(delta, max_delta_per_joint)
        current_angles = current_angles + delta

    return current_angles

def manipulability_gradient(arm: RoboticArm, angles):
    """
    Обчислює градієнт маніпулябельності для роботизованої руки
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
    Обчислює градієнт для уникнення обмежень суглобів
    """
    grad = -weight * (angles - np.mean(joint_limits, axis=1)) / (np.ptp(joint_limits, axis=1) ** 2)
    return grad

def interpolate_points(point1, point2, num_steps):
    """
    Генерує проміжні точки між двома точками для плавніших переходів
    """
    return np.linspace(point1, point2, num_steps, endpoint=False)[1:]

#######################################################################################
#######################################################################################


# ------- Допоміжна функція: безпечне малювання -------
def safe_draw(arm_plotter: RoboticArmPlotter):
    """
    Викликає arm_plotter.draw() лише якщо це не перевантажить внутрішній буфер
    Це захищає від AssertionError: Too many points
    """
    try:
        num = getattr(arm_plotter, "num_points", None)
        mx = getattr(arm_plotter, "max_num_points", None)
        if num is None or mx is None:
            # Не можемо перевірити, викликаємо нормально, але ловимо AssertionError
            arm_plotter.draw()
        else:
            # резервуємо 2 слоти запасу
            if num < mx - 2:
                arm_plotter.draw()
            else:
                # пропускаємо малювання, щоб уникнути помилки; рендер все одно покаже існуючі штрихи
                pass
    except AssertionError:
        # Якщо бібліотека видає помилку, ігноруємо, щоб уникнути краху — фінальне зображення все одно буде відображено
        pass


if __name__ == "__main__":
    ctx = core.RenderingContext("Завдання 1 - візуалізація")

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

    # Прапори стану та захисти
    pen_down = False
    final_image_rendered = False

    drawing = drawing1

    # Переміщуємо руку до початкової точки перед основним циклом
    start_point = np.copy(drawing[0][0])
    if start_point[1] > 13.0:
        start_point[1] = 13.0

    target_angles = inverse_kinematics(start_point, arm)
    controller.move_to_angles(target_angles)

    # Чекаємо, поки контролер завершить початковий рух, одночасно рендерячи
    while not controller.is_idle():
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                raise SystemExit
        # Використовуємо непрозоре заповнення для надійного рендерингу
        ctx.screen.fill((0, 0, 0))
        arm.render()
        controller.step()
        ctx.world.Step(TIME_STEP, 10, 10)
        pygame.display.flip()
        ctx.clock.tick(TARGET_FPS)

    # змінні основного циклу
    running = True
    i_shape = 0
    i_point = 0
    state = 'MOVE_TO_START'

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            # Непрозоре очищення для уникнення проблем з альфа-каналом
            ctx.screen.fill((0, 0, 0))

            # Рендер сцени
            arm.render()
            controller.step()

            # Викликаємо safe_draw лише якщо pen_down, щоб уникнути зростання буфера кожного кадру
            if pen_down:
                safe_draw(arm)

            if controller.is_idle():
                # Якщо вже завершено, продовжуємо рендерити фінальне зображення
                if state == 'FINISHED':
                    ctx.world.Step(TIME_STEP, 10, 10)
                    pygame.display.flip()
                    ctx.clock.tick(TARGET_FPS)
                    continue

                # Завершено всі фігури
                if i_shape >= len(drawing):
                    arm.stop_drawing()
                    pen_down = False
                    if not final_image_rendered:
                        safe_draw(arm)  # фінальна спроба захопити останні штрихи
                        final_image_rendered = True
                    state = 'FINISHED'
                    print("Малювання завершено! Вікно залишиться відкритим, показуючи фінальне зображення. Натисніть ESC або закрийте вікно, щоб вийти.")
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
                    safe_draw(arm)  # захоплюємо поточну точку негайно

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

            # фізика / оновлення дисплея
            ctx.world.Step(TIME_STEP, 10, 10)
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)

    except KeyboardInterrupt:
        print("Переривання з клавіатури. Завершення...")
    finally:
        pygame.quit()