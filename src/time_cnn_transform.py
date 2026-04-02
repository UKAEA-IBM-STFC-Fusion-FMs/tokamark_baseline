from typing import Dict, Any
import numpy as np


# ======================================================================================================================
class TimeCNNTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dict_metadata, verbose=False):
        # dictionary that persists across calls
        self.dict_metadata = dict_metadata
        
        output_length = self.dict_metadata['task_window_segmenter']['output_length']
        delta = self.dict_metadata['task_window_segmenter']['delta']

        self.verbose = verbose
        self.var_groups = {"input": None, 
                           "actuator": None, 
                           "output": None}
        
        max_input_length = 0.050

        self.list_id_start_time_input = []
        for var in self.dict_metadata['input'].keys():
            if self.dict_metadata["task_type"] == "non_markovian":     
                dt_var = self.dict_metadata['input'][var]['dt']
                id_start_time = int(max_input_length/dt_var)
                self.list_id_start_time_input.append(id_start_time)
            else:
                self.list_id_start_time_input.append(0)
        
        self.list_id_start_time_actuator = []
        for var in self.dict_metadata['actuator'].keys():
            if self.dict_metadata["task_type"] == "non_markovian":
                dt_var = self.dict_metadata['actuator'][var]['dt']
                id_start_time = int(max_input_length/dt_var) + int(delta/dt_var) + int(output_length/dt_var)
                self.list_id_start_time_actuator.append(id_start_time)
            else:
                self.list_id_start_time_actuator.append(0)
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        shot.update({
            "input": [ 
                np.expand_dims(arr, axis=0) 
                for arr in (
                    np.moveaxis(data["values"][..., -self.list_id_start_time_input[i]:], -1, 0)
                    for i, (var, data) in enumerate(shot["input"].items())
                )
            ],
            "actuator": [ 
                np.expand_dims(arr, axis=0)
                for arr in (
                    np.moveaxis(data["values"][..., -self.list_id_start_time_actuator[i]:], -1, 0)
                    for i, (var, data) in enumerate(shot["actuator"].items())
                )
            ],
            "output": [
                np.expand_dims(arr, axis=0)
                for arr in (
                    np.moveaxis(data["values"], -1, 0)
                    for var, data in shot["output"].items()
                )
            ]
        })

        return {
            'x': shot["input"] + shot["actuator"],
            'y':  shot["output"] 
            }
