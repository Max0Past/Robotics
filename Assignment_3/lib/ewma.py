import numpy as np
from scipy import signal

def ewma(data, alpha, offset=None, dtype=None):
    data = np.asarray(data, dtype=dtype if dtype else data.dtype)
    
    if data.size < 1:
        return data
    
    # Використання рекурсивного фільтра
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    if offset is None:
        zi = signal.lfilter_zi(b, a) * data[0]
    else:
        zi = signal.lfilter_zi(b, a) * offset
    
    out, _ = signal.lfilter(b, a, data, zi=zi)
    
    return out