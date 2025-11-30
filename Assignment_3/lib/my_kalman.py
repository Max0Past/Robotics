import numpy as np

# Для використання референсної імплементації поставте True
REFERENCE_IMPLEMENTATION = False


if REFERENCE_IMPLEMENTATION:
    def predict(x, P, F, Q):
        """
        x - середнє стану, numpy array of shape (N,)
        P - коваріація стану, numpy array of shape (N, N)
        F - матриця динаміки стану, numpy array of shape (N, N)
        Q - матриця коваріації похибки моделі динаміки, numpy array of shape (N, N)
        """
        # Імплементуйте рівняння для кроку прогнозу КФ (predict step)
        pass

    def update(x, P, z, R, H):
        """
        Треба викликати після кроку прогнозу (predict)
        x - середнє стану, numpy array of shape (N,)
        P - коваріація стану, numpy array of shape (N, N)
        z - спостереження, numpy array of shape (M,)
        R - матриця коваріації похибки моделі спостереження, numpy array of shape (M, M).
            По факту параметри шуму сенсорів.
        H - матриця спостереження, numpy array of shape (N, M)
        """
        # Імплементуйте рівняння для кроку оновлення КФ (update step)
        pass
else:
    from filterpy.kalman import update, predict