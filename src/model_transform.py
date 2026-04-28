from typing import Dict, Any
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
max_input_length = 0.050

# ------------------------------------------------------------------------------------------------------------------
def _spaced_windows_tensor(arr, n, L):
    N = arr.shape[0]
    if L > N:
        raise ValueError("L is larger than first dimension")

    max_start = N - L

    if n == 1:
        starts = np.array([0])
    else:
        starts = np.linspace(0, max_start, n)
        starts = np.round(starts).astype(int)

    windows = np.stack([arr[s:s+L] for s in starts], axis=0)
    return windows

# ------------------------------------------------------------------------------------------------------------------
def _resample(shot_section, n_window):
    resampled = []

    for var, shot in shot_section.items():
        time = shot["time"]
        values = shot["values"]

        w_size = int(np.ceil(len(time) / n_window))
        # print('n_window =', n_window)

        if w_size <= 0:
            raise ValueError(f"Invalid resampling factor for {var}")

        # Reshape windows
        # print(f' {var} before :', values.shape)
        resampled_values = _spaced_windows_tensor(values, n_window, w_size)
        # print(f' {var} after :', resampled_values.shape)

        resampled.append(resampled_values)

    return resampled


# ======================================================================================================================
class ModelTransform_1:

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

        self.list_id_start_time_input = []
        for var in self.dict_metadata['input'].keys():
            if self.dict_metadata["task_type"] == "non_markovian":     
                dt_var = self.dict_metadata['input'][var]['dt']
                id_start_time = int(max_input_length/dt_var)
                self.list_id_start_time_input.append(id_start_time)
            else:
                self.list_id_start_time_input.append(0)
        
        self.list_id_start_time_actuator = []
        self.list_id_t_cut_time_actuator = []
        for var in self.dict_metadata['actuator'].keys():
            # print('var ', var)
            dt_var = self.dict_metadata['actuator'][var]['dt']
            
            if self.dict_metadata["task_type"] == "non_markovian":
                id_start_time = int(max_input_length/dt_var) + int(delta/dt_var) + int(output_length/dt_var)
                id_t_cut_time = ( id_start_time
                    - int( min(max_input_length, self.dict_metadata['task_window_segmenter']['input_length']) / dt_var ) )
            else:
                id_start_time = 0
                id_t_cut_time = int(delta/dt_var) + int(output_length/dt_var)

            self.list_id_start_time_actuator.append(id_start_time)
            self.list_id_t_cut_time_actuator.append(id_t_cut_time)

            # print('id_t_cut_time ', id_t_cut_time)
            # print('id_start_time ', id_start_time)
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        
        # print("\nshot_id ", shot['shot_id'])
        # print('input')
        # print([ shot['input'][var]['values'].shape for var in shot['input'].keys() ])
        # print('actuator')
        # print([ shot['actuator'][var]['values'].shape for var in shot['actuator'].keys() ])
        # print('output')
        # print([ shot['output'][var]['values'].shape for var in shot['output'].keys() ])

        shot.update({
            "input": {
                var: {
                    "values": np.moveaxis(
                        data["values"][..., -self.list_id_start_time_input[i]:],
                        -1,
                        0
                    ),
                    "time": data["time"][-self.list_id_start_time_input[i]:]
                }
                for i, (var, data) in enumerate(shot["input"].items())
            },

            # "actuator": {
            #     var: {
            #         "values": np.moveaxis(
            #             data["values"][..., -self.list_id_start_time_actuator[i]:],
            #             -1,
            #             0
            #         ),
            #         "time": data["time"][-self.list_id_start_time_actuator[i]:]
            #     }
            #     for i, (var, data) in enumerate(shot["actuator"].items())
            # },

            "actuator_past": {
                var: {
                    "values": np.moveaxis(
                        data["values"][..., -self.list_id_start_time_actuator[i]:-self.list_id_t_cut_time_actuator[i]],
                        -1,
                        0
                    ),
                    "time": data["time"][-self.list_id_start_time_actuator[i]:-self.list_id_t_cut_time_actuator[i]]
                }
                for i, (var, data) in enumerate(shot["actuator"].items())
            },

            "actuator_future": {
                var: {
                    "values": np.moveaxis(
                        data["values"][..., -self.list_id_t_cut_time_actuator[i]:],
                        -1,
                        0
                    ),
                    "time": data["time"][-self.list_id_t_cut_time_actuator[i]:]
                }
                for i, (var, data) in enumerate(shot["actuator"].items())
            },

            "output": {
                var: {
                    "values": np.moveaxis(
                        data["values"],
                        -1,
                        0
                    ),
                    "time": data["time"]
                }
                for var, data in shot["output"].items()
            }
        })

        # print("\nshot_id ", shot['shot_id'])
        # print('input')
        # print([ shot['input'][var]['values'].shape for var in shot['input'].keys() ])
        # print('actuator past')
        # print([ shot['actuator_past'][var]['values'].shape for var in shot['actuator_past'].keys() ])
        # print('actuator_future')
        # print([ shot['actuator_future'][var]['values'].shape for var in shot['actuator_future'].keys() ])
        # print('output')
        # print([ shot['output'][var]['values'].shape for var in shot['output'].keys() ])

        return shot


# ======================================================================================================================
class ModelTransform_2:

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

        # self.model_dt = max(
        #     self.dict_metadata[section][var]['dt']
        #     for section in ['input', 'actuator', 'output']
        #     for var in self.dict_metadata[section]
        # )

        self.model_dt = 0.005
        self.list_n_windows = {
            'input': int(min(max_input_length, self.dict_metadata['task_window_segmenter']['input_length']) / self.model_dt),  
            'output': int(self.dict_metadata['task_window_segmenter']['output_length'] / self.model_dt),
            'actuator': int( min(max_input_length, self.dict_metadata['task_window_segmenter']['input_length']) / self.model_dt 
                        + self.dict_metadata['task_window_segmenter']['output_length'] / self.model_dt 
                        + self.dict_metadata['task_window_segmenter']['delta'] / self.model_dt )
                        }
        # print(self.list_n_windows)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        t_cut = shot['t_cut']

        # print('\n input')
        x_input = _resample(
            shot["input"], self.list_n_windows["input"]
        )
        # print([arr.shape for arr in x_input])
        # print([np.expand_dims(arr, axis=1).shape for arr in x_input])

        # # print('\n actuator')
        # x_actuator = self._resample(
        #     shot["actuator"], self.dict_metadata["actuator"], t_cut
        # )

        # print('\n actuator past')
        x_actuator_past = _resample(
            shot["actuator_past"], self.list_n_windows["input"]
        )
        # print([arr.shape for arr in x_actuator_past])
        # print([np.expand_dims(arr, axis=1).shape for arr in x_actuator_past])

        # print('\n actuator future')
        x_actuator_future = _resample(
            shot["actuator_future"], self.list_n_windows["actuator"] - self.list_n_windows["input"]
        )

        # # print('\n y')
        y_output = [data["values"] for var, data in shot["output"].items()]


        return {
            'input': [np.expand_dims(arr, axis=1) for arr in x_input + x_actuator_past],
            'exogenous': [np.expand_dims(arr, axis=1) for arr in x_actuator_future],
            # 'y': [np.expand_dims(arr, axis=1) for arr in y_output]
            'y': y_output
        }