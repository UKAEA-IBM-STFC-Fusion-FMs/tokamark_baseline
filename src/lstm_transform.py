from typing import Dict, Any
import numpy as np


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
def _resample(shot_section, metadata_section, lstm_dt):
    resampled = []

    for var, shot in shot_section.items():
        time = shot["time"]
        values = shot["values"]

        dt_var = metadata_section[var]['dt']
        w_size = int(round(lstm_dt / dt_var))
        n_window = int( len(time) / w_size )

        if w_size <= 0:
            raise ValueError(f"Invalid resampling factor for {var}")

        # Reshape windows
        # print(f' {var} before :', values.shape)
        resampled_values = _spaced_windows_tensor(values, n_window, w_size)
        # print(f' {var} after :', resampled_values.shape)

        resampled.append(resampled_values)

    return resampled


# ======================================================================================================================
class LstmTransform_1:

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
        self.list_id_t_cut_time_actuator = []
        for var in self.dict_metadata['actuator'].keys():
            dt_var = self.dict_metadata['actuator'][var]['dt']
            
            if self.dict_metadata["task_type"] == "non_markovian":
                id_start_time = int(max_input_length/dt_var) + int(delta/dt_var) + int(output_length/dt_var)
                self.list_id_start_time_actuator.append(id_start_time)
            else:
                self.list_id_start_time_actuator.append(0)
            
            id_t_cut_time = ( self.list_id_start_time_actuator[-1] 
                + int( min(max_input_length, self.dict_metadata['task_window_segmenter']['input_length']) / dt_var ) )
            
            self.list_id_t_cut_time_actuator.append(id_t_cut_time)
        
        # print(self.list_id_t_cut_time_actuator)
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

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
                        data["values"][..., -self.list_id_start_time_actuator[i]:self.list_id_t_cut_time_actuator[i]],
                        -1,
                        0
                    ),
                    "time": data["time"][-self.list_id_start_time_actuator[i]:self.list_id_t_cut_time_actuator[i]]
                }
                for i, (var, data) in enumerate(shot["actuator"].items())
            },

            "actuator_future": {
                var: {
                    "values": np.moveaxis(
                        data["values"][..., self.list_id_t_cut_time_actuator[i]:],
                        -1,
                        0
                    ),
                    "time": data["time"][self.list_id_t_cut_time_actuator[i]:]
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

        return shot


# ======================================================================================================================
class LstmTransform_2:

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

        self.lstm_dt = max(
            self.dict_metadata[section][var]['dt']
            for section in ['input', 'actuator', 'output']
            for var in self.dict_metadata[section]
        )

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        t_cut = shot['t_cut']

        # print('\n input')
        x_input = _resample(
            shot["input"], self.dict_metadata["input"], self.lstm_dt
        )
        # print([arr.shape for arr in x_input])
        # print([np.expand_dims(arr, axis=1).shape for arr in x_input])

        # # print('\n actuator')
        # x_actuator = self._resample(
        #     shot["actuator"], self.dict_metadata["actuator"], t_cut
        # )

        # print('\n actuator past')
        x_actuator_past = _resample(
            shot["actuator_past"], self.dict_metadata["actuator"], self.lstm_dt
        )

        # print('\n actuator future')
        x_actuator_future = _resample(
            shot["actuator_future"], self.dict_metadata["actuator"], self.lstm_dt
        )

        # # print('\n y')
        y_output = [data["values"] for var, data in shot["output"].items()]
        # _resample(
        #     shot["output"], self.dict_metadata["output"], self.lstm_dt
        # )

        return {
            'input': [np.expand_dims(arr, axis=1) for arr in x_input + x_actuator_past],
            'exogenous': [np.expand_dims(arr, axis=1) for arr in x_actuator_future],
            # 'y': [np.expand_dims(arr, axis=1) for arr in y_output]
            'y': y_output
        }