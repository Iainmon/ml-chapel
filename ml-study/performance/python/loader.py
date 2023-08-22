import numpy as np
import struct
import os

def load_chapel_tensor(file):
    rank, = struct.unpack('<q', file.read(8))
    shape = [struct.unpack('<q', file.read(8))[0] for _ in range(rank)]
    a = np.zeros(shape)
    for x in np.nditer(a):
        x, = struct.unpack('<d', file.read(8))
    return a

def load_chapel_dataset(folder):
    # get all names of binary files
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.bin')]
    dataset = {}
    for f in files:
        path = os.path.join(folder, f)
        category = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as file:
            dataset[category] = load_chapel_tensor(file)
    return dataset
        

