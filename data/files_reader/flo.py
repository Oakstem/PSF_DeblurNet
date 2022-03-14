import numpy as np


def read_flo(path):
    with open(path, 'rb') as f:

        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'

        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]

        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))

    return data2D