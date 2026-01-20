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
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        # Copy non-window metadata
        # new_shot = {k: v for k, v in shot.items() if k not in ["input", "actuator", "output"]}

        # Replace windowed data
        shot.update({
            "input": [ np.expand_dims(np.moveaxis(data["values"], -1, 0), axis=0) for var, data in shot["input"].items() ],
            "actuator": [ np.expand_dims(np.moveaxis(data["values"], -1, 0), axis=0) for var, data in shot["actuator"].items() ],
            "output": [ np.expand_dims(np.moveaxis(data["values"], -1, 0), axis=0) for var, data in shot["output"].items() ],
        })
        
        return {
            'x': shot["input"] + shot["actuator"],
            'y':  shot["output"] 
            }
    
    # ------------------------------------------------------------------------------------------------------------------
    def _get_timestamp(self, data: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Helper to slice time series data within a given window."""
        new_data = {}

        for key, d_var in data.items():
            # print(key)

            # dt = self.dict_metadata[key]['dt'] 
            # timestamp_index = int( dt_sec / dt )
            shape_values = self.dict_metadata[key]['values_shape']

            if d_var["values"].size == 0:
                # Handle when var not present or not enough datapoint
                # times = np.concatenate([np.full(1, np.nan), d_var["time"]], axis=-1)
                # values = np.concatenate([np.full(shape_values + (1,), np.nan, dtype=float)], axis=-1)
                new_data[key] = {
                    "time": np.full(1, np.nan), 
                    "values": np.full(shape_values + (1,), np.nan, dtype=float)
                }
            else:
                # values = np.concatenate([np.full(shape_values + (1,), np.nan, dtype=float), d_var["values"]], axis=-1)

                if mode == "first":

                    new_data[key] = {
                        "time": d_var["time"][:1],
                        "values": d_var["values"][..., :1],
                    }
                    # print('first')
                    # print(key, mode, values[..., :1].shape)
                
                elif mode == "last":

                    new_data[key] = {
                        "time": d_var["time"][-1:],
                        "values": d_var["values"][..., -1:],
                    }
                    # print('last')
                    # print(key, mode, values[..., -1:].shape)

            # elif mode == "future":

            #     # Handle when var not present or not enough datapoint
            #     times = np.concatenate([d_var["time"], np.full(1, np.nan)], axis=-1)
            #     if d_var["values"].size == 0:
            #         values = np.concatenate([np.full(shape_values + (1,), np.nan, dtype=float)], axis=-1)
            #     else:
            #         values = np.concatenate([d_var["values"], np.full(shape_values + (1,), np.nan, dtype=float)], axis=-1)

            #     new_data[key] = {
            #         "time": times[timestamp_index],
            #         "values": values[..., timestamp_index],
            #     }
            #     # print('future')
            #     # print(key, mode, values[..., timestamp_index].shape)
                else:
                    raise ValueError(f"Invalid mode '{mode}' — expected 'past' or 'future'.")

        return new_data


