import torch
import numpy as np

def signal2pytorch(x):
    """ Convert a signal vector x into a 3D PyTorch tensor for Conv1D input. """
    X = np.expand_dims(x, axis=0)  # Add channel dimension
    X = np.expand_dims(X, axis=0)  # Add batch dimension
    X = torch.from_numpy(X).float()
    return X
