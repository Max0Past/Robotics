import numpy as np

# Для використання референсної імплементації поставте True
REFERENCE_IMPLEMENTATION = True


def predict(x, P, F, Q):
    """
    x - середнє стану, numpy array of shape (N,)
    P - коваріація стану, numpy array of shape (N, N)
    F - матриця динаміки стану, numpy array of shape (N, N)
    Q - матриця коваріації похибки моделі динаміки, numpy array of shape (N, N)
    """
    # x = F @ x
    x_pred = F @ x
    
    # P = F @ P @ F.T + Q
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

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
    # y = z - H @ x
    y = z - H @ x
    
    # S = H @ P @ H.T + R
    S = H @ P @ H.T + R
    
    # K = P @ H.T @ inv(S)
    K = P @ H.T @ np.linalg.inv(S)
    
    # x = x + K @ y
    x_new = x + K @ y
    
    # P = (I - K @ H) @ P
    I = np.eye(len(x))
    P_new = (I - K @ H) @ P
    
    return x_new, P_new