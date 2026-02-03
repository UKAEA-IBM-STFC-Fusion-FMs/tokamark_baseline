from typing import Dict, Any
import numpy as np

# ======================================================================================================================
class TimeCNNTransform:

    def __init__(self, dict_metadata, verbose=False):
        # dictionary that persists across calls
        self.dict_metadata = dict_metadata
        self.verbose = verbose
        self.var_groups = {"input": None, 
                           "actuator": None, 
                           "output": None}
        
        max_input_sec = 0.050
        t_to_cut_sec = max(0, round(self.dict_metadata['task_window_segmenter']['input_length'] - max_input_sec, 3))
        print(t_to_cut_sec)

        self.list_id_start_time_input = []
        for var in self.dict_metadata['input'].keys():
            if self.dict_metadata["task_name"] in ["task_3-3", 
                                                   "task_4-1", "task_4-2", 
                                                   "task_4-3", "task_4-4", "task_4-5"]:              
                dt_var = self.dict_metadata['input'][var]['dt']
                id_start_time = int(t_to_cut_sec/dt_var)
                self.list_id_start_time_input.append(id_start_time)
            else:
                self.list_id_start_time_input.append(0)
        print(self.list_id_start_time_input)
        
        self.list_id_start_time_actuator = []
        for var in self.dict_metadata['actuator'].keys():
            if self.dict_metadata["task_name"] in ["task_3-3", 
                                                   "task_4-1", "task_4-2", 
                                                   "task_4-3", "task_4-4", "task_4-5"]:
                dt_var = self.dict_metadata['actuator'][var]['dt']
                id_start_time = int(t_to_cut_sec/dt_var)
                self.list_id_start_time_actuator.append(id_start_time)
            else:
                self.list_id_start_time_actuator.append(0)
        print(self.list_id_start_time_actuator)
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        shot.update({
            "input": [ np.expand_dims(arr, axis=0) if arr.shape[0] != 1 else arr
                      for arr in (np.moveaxis(data["values"][..., self.list_id_start_time_input[i]:], -1, 0)
                                  for i, (var, data) in enumerate(shot["input"].items()))
                                  ],
            "actuator": [ np.expand_dims(arr, axis=0) if arr.shape[0] != 1 else arr
                      for arr in (np.moveaxis(data["values"][..., self.list_id_start_time_actuator[i]:], -1, 0)
                                  for i, (var, data) in enumerate(shot["actuator"].items()))
                                  ],
            "output": [ np.expand_dims(arr, axis=0) if arr.shape[0] != 1 else arr
                      for arr in (np.moveaxis(data["values"], -1, 0)
                                  for var, data in shot["output"].items())
                                  ]
            })
        
        return {
            'x': shot["input"] + shot["actuator"],
            'y':  shot["output"] 
            }
