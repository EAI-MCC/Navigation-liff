import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
with h5py.File('data/bathroom_02.h5','r') as f:
    all_image = np.array(f['observation'])
    print(np.shape(all_image))
    for i in range(180):
        print(i)
        plt.imshow(all_image[i])
        plt.ion()
        plt.show()
        plt.pause(0.001)
        time.sleep(1)
