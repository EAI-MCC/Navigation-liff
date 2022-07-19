# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import random

LOCAL_T_MAX = 5 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 100 # parallel thread size
ACTION_SIZE = 4 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 100.0 * 10**6 # 10 million frames
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
VERBOSE = True

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4

NUM_EVAL_EPISODES = 100 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)
TASK_LIST = {
  'bathroom_01'    : ['26', '37', '43', '53', '69'],
  'bedroom_01'     : ['134', '264', '320', '384', '387'],
  'kitchen_01'     : ['90', '136', '157', '207', '329'],
  'living_room_01' : ['92', '135', '193', '228', '254']  
}
TASK_LIST_EVA = {
  'bathroom_01'    : ['26', '37', '43', '53', '69'],
  'bedroom_01'     : ['134', '264', '320', '384', '387'],
  'kitchen_01'     : ['90', '136', '157', '207', '329'],
  'living_room_01' : ['92', '135', '193', '228', '254']  
}

class autoTaskList:
  def __init__(self,datapath) -> None:
    self.TaskList = {}
    self.TaskListEva = {}
    file_names = os.listdir(datapath)
    for name in file_names:
        if name.find('h5')<0:
            continue
        with h5py.File(datapath+name) as f:
            data_np_array = np.array(f['location'])
            num_of_data = np.shape(data_np_array)[0]
            list_name = name[:-3]
            self.TaskList[list_name] = []
            self.TaskListEva[list_name] = []
            for ii in range(5):
              # automatically generate tasklist dict from ii*num_of_data to (ii+1)*num_of_data
              self.TaskList[list_name].append(str(random.randint(int(ii*num_of_data/5)
                ,int((ii+1)*num_of_data/5-1))))
              self.TaskListEva[list_name].append(str(random.randint(int(ii*num_of_data/5)
                ,int((ii+1)*num_of_data/5))))
  @property
  def tasklist(self):
    return self.TaskList
  @property
  def tasklisteva(self):
    return self.TaskListEva

                  
      


# number of each dataset
'''
bedroom_06.h5 --- 308
bedroom_04.h5 --- 408
living_room_01.h5 --- 776
bedroom_05.h5 --- 1028
living_room_07.h5 --- 532
bedroom_03.h5 --- 496
living_room_04.h5 --- 876
bathroom_03.h5 --- 204
living_room_06.h5 --- 876
living_room_03.h5 --- 644
living_room_05.h5 --- 788
bedroom_08.h5 --- 496
bathroom_02.h5 --- 180
living_room_08.h5 --- 468
kitchen_02.h5 --- 676
bedroom_07.h5 --- 284
bathroom_04.h5 --- 168
bathroom_01.h5 --- 132
bedroom_02.h5 --- 336
living_room_02.h5 --- 700
'''
if __name__ == '__main__':
  a = autoTaskList()
  print(a.TaskList)
  print(a.TaskListEva)
