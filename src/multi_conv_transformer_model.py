
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchinfo import summary
# import torch.nn.functional as F
# import numpy as np

from src.model_transform import _make_dummy_outputs  # _resample
from src.conv_encoders_decoders import Conv1DEncoder, Conv2DEncoder, Conv3DEncoder, Conv1DDecoder, Conv2DDecoder, Conv3DDecoder
from src.module import ARPredictor
from tokamark.tools.utils import get_device

# ----------------------------------------------------------------------------------------------------------------------
# Set device
device = get_device()
# print(f"Using device: {device}\n")

padding = 1
kernel_size = 3
stride = 3
layers_encoder = 3
layers_decoder = 3
bb_factor = 2

# Transformer / world-model backbone defaults (overridable via the YAML config).
# wm.* knobs
D = 16
embed_dim = None        # None -> D * bb_factor
embed_dim_act = None    # None -> D * bb_factor
mem_window = None        # None -> attend over the full available history (no sliding limit)
# predictor.* knobs
depth = 4
heads = 4
dim_head = 32
mlp_dim = None           # None -> 4 * embed_dim
dropout = 0.1
emb_dropout = 0.0


# ----------------------------------------------------------------------------------------------------------------------
def _make_encoder(var_shape, D):
    """Build a per-variable CNN encoder matching the variable rank."""
    if len(var_shape) == 5:    # (2, T, 15, 17) images evolving in time
        return Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
    elif len(var_shape) == 4:  # (1, T, 15) profiles evolving in time
        return Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
    elif len(var_shape) == 3:  # (7, T,) time series evolving in time
        return Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
    raise ValueError(f"Unsupported input shape: {var_shape[1:]}")


# ----------------------------------------------------------------------------------------------------------------------
def _make_decoder(var_shape, D):
    """Build a per-variable CNN decoder matching the variable rank."""
    if len(var_shape) == 5:
        return Conv3DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
    elif len(var_shape) == 4:
        return Conv2DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
    elif len(var_shape) == 3:
        return Conv1DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
    raise ValueError(f"Unsupported input shape: {var_shape[1:]}")


# ----------------------------------------------------------------------------------------------------------------------
def create_transformer_architecture(
    dataloader_, dict_metadata, wm_config=None, predictor_config=None, verbose=True
):

    # Hyper-parameters are read from the YAML config (`wm` / `predictor`
    # sections); any missing key falls back to the module-level defaults.
    wm_config = wm_config or {}
    predictor_config = predictor_config or {}

    if verbose:
        print("\n\n----------TRANSFORMER WORLD-MODEL INITIALIZATION----------\n")

    input_shapes = []
    exogenous_shapes = []
    output_shapes = []

    # ------------------------------------------------------------
    # 1. Extract one valid sample for shape inference
    # ------------------------------------------------------------
    for l, first_window in enumerate(dataloader_.dataset):
        try:
            input_shapes = [arr.shape for arr in first_window["input"]]
            exogenous_shapes = [arr.shape for arr in first_window["exogenous"]]
            output_shapes = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Actuator future shapes are: {exogenous_shapes}")
                print(f"Output shapes are: {output_shapes}")

            break

        except Exception as e:
            print(f"Skipping sample {l} because not trainable: {e}")
            continue

    # ------------------------------------------------------------
    # 2. Create model
    # ------------------------------------------------------------
    model = MultiConv_Transformer(
        input_shapes=input_shapes,
        exogenous_shapes=exogenous_shapes,
        output_shapes=output_shapes,
        dict_metadata=dict_metadata,
        D=wm_config.get("D", D),
        embed_dim=wm_config.get("embed_dim", embed_dim),
        embed_dim_act=wm_config.get("embed_dim_act", embed_dim_act),
        mem_window=wm_config.get("mem_window", mem_window),
        depth=predictor_config.get("depth", depth),
        heads=predictor_config.get("heads", heads),
        dim_head=predictor_config.get("dim_head", dim_head),
        mlp_dim=predictor_config.get("mlp_dim", mlp_dim),
        dropout=predictor_config.get("dropout", dropout),
        emb_dropout=predictor_config.get("emb_dropout", emb_dropout),
    ).to(device)

    # ------------------------------------------------------------
    # 3. Build dummy input sizes for torchinfo (raw tensors)
    # ------------------------------------------------------------
    input_sizes = []
    for shape in (input_shapes + exogenous_shapes):
        input_sizes.append((2,) + shape)

    # ------------------------------------------------------------
    # 4. Model summary
    # ------------------------------------------------------------
    if verbose:
        summary(model, input_size=input_sizes)

    return model


# ======================================================================================================================
class MultiConv_Transformer(nn.Module):
    """CNN-token world model with a conditional-injection Transformer predictor.

    Pipeline
    --------
    * Each *diagnostic* variable (``dict_metadata["input"]``) is compressed by
      its own CNN encoder into a fixed-length vector. Per window step these are
      concatenated into a single *input* vector and projected by one linear
      layer to ``embed_dim`` -> state tokens ``(B, W_in, embed_dim)``.
    * Each *actuator* variable (``dict_metadata["actuator"]``) is compressed by
      its own CNN encoder (past + future), concatenated per step into a single
      *action* vector and projected by one linear layer to ``embed_dim_act``
      -> action tokens ``(B, W_in + W_fut, embed_dim_act)``.
    * An autoregressive Transformer (``ARPredictor``) rolls the state tokens
      forward: the latent token at step ``W_{t+1}`` is predicted from the
      tokens in the memory window ``W_{t-n} .. W_t`` (causal attention),
      with the action vector of each step injected as an AdaLN-zero condition.
    * The predicted future latent tokens are projected back to the decoder
      width and decoded per output variable by CNN decoders.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(  # NOSONAR - Ignore cognitive complexity
        self,
        input_shapes,
        exogenous_shapes,
        output_shapes,
        dict_metadata,
        D=D,
        embed_dim=embed_dim,
        embed_dim_act=embed_dim_act,
        mem_window=mem_window,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ):

        super().__init__()

        self.D = D
        self.W_in = input_shapes[0][0]

        # per-variable CNN encoder output width
        self.d_var = D * bb_factor
        # decoder input width (kept identical to the LSTM/MLP baselines)
        self.dec_dim = D * bb_factor

        # latent token widths
        self.embed_dim = embed_dim if embed_dim is not None else D * bb_factor
        self.embed_dim_act = embed_dim_act if embed_dim_act is not None else D * bb_factor

        # --------------------------------------------------------------------------------------------------------------
        # Split the flattened branch list into diagnostics vs. actuators.
        #   item["input"]     = diagnostics (n_diag) + actuator_past (n_act)
        #   item["exogenous"] = actuator_future (n_act)
        # --------------------------------------------------------------------------------------------------------------
        self.n_diag = len(dict_metadata["input"])
        self.n_act = len(dict_metadata["actuator"])

        diag_shapes = input_shapes[: self.n_diag]
        act_past_shapes = input_shapes[self.n_diag: self.n_diag + self.n_act]
        act_fut_shapes = exogenous_shapes  # actuator_future (n_act)

        # --------------------------------------------------------------------------------------------------------------
        # Output (decoder) latent shapes
        # --------------------------------------------------------------------------------------------------------------
        self.output_shapes = output_shapes
        y = _make_dummy_outputs(output_shapes, dict_metadata)
        output_latent_shapes = [arr.shape for arr in y]
        self.W_out = output_latent_shapes[0][0]

        # --------------------------------------------------------------------------------------------------------------
        # 1. Per-variable CNN encoders
        # --------------------------------------------------------------------------------------------------------------
        # diagnostics -> state tokens
        self.diag_branches = nn.ModuleList(
            [_make_encoder(s, D) for s in diag_shapes]
        )
        # actuators -> action tokens (separate encoders for past / future as
        # the resampled window length can differ between them)
        self.act_past_branches = nn.ModuleList(
            [_make_encoder(s, D) for s in act_past_shapes]
        )
        self.act_fut_branches = nn.ModuleList(
            [_make_encoder(s, D) for s in act_fut_shapes]
        )

        # --------------------------------------------------------------------------------------------------------------
        # 2. Linear alignment layers (concat-of-variables -> single token)
        # --------------------------------------------------------------------------------------------------------------
        self.state_proj = nn.Linear(self.n_diag * self.d_var, self.embed_dim)
        self.action_proj = nn.Linear(self.n_act * self.d_var, self.embed_dim_act)

        # --------------------------------------------------------------------------------------------------------------
        # 3. Conditional-injection Transformer predictor (rollout backbone)
        # --------------------------------------------------------------------------------------------------------------
        self.mem_window = mem_window
        self.predictor = ARPredictor(
            num_frames=self.W_in + self.W_out,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim if mlp_dim is not None else 4 * self.embed_dim,
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            cond_dim=self.embed_dim_act,
            output_dim=self.embed_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        # project predicted latent tokens back to the CNN-decoder width
        self.dec_proj = (
            nn.Linear(self.embed_dim, self.dec_dim)
            if self.embed_dim != self.dec_dim
            else nn.Identity()
        )

        # --------------------------------------------------------------------------------------------------------------
        # 4. Per-variable CNN decoders
        # --------------------------------------------------------------------------------------------------------------
        self.output_branches = nn.ModuleList(
            [_make_decoder(s, D) for s in output_latent_shapes]
        )

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _run_cnn_encoder(branch, x):
        B, W = x.shape[:2]  # batch, num_windows
        x = x.view(B * W, *x.shape[2:])
        out = branch(x)
        out = out.view(B, W, -1)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _run_cnn_decoder(branch, goal_shape, x):
        B, W = x.shape[:2]
        x = x.reshape(B * W, *x.shape[2:])
        out = branch(x)
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])
        target_len = goal_shape[0]
        out = out[:, -target_len:]
        return out

    # ------------------------------------------------------------------------------------------------------------------
    def _encode_group(self, branches, tensors):
        """Encode each variable with its own CNN, then concat per step.

        Returns (B, W, n_var * d_var).
        """
        outs = []
        for branch, x in zip(branches, tensors):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            outs.append(out)
        return torch.cat(outs, dim=2)

    # ------------------------------------------------------------------------------------------------------------------
    def _rollout(self, state_tokens, action_tokens):
        """Autoregressive rollout with a sliding memory window.

        state_tokens : (B, W_in, embed_dim)         observed history
        action_tokens: (B, >=W_in+W_out, e_act)     per-step conditioning

        Returns the predicted future latents (B, W_out, embed_dim).
        """
        seq = state_tokens
        preds = []

        for _ in range(self.W_out):
            cur_len = seq.size(1)  # global index of the step being predicted

            # restrict the context to the last `mem_window` tokens
            if self.mem_window is not None:
                ctx = seq[:, -self.mem_window:]
            else:
                ctx = seq
            ctx_len = ctx.size(1)

            # actions aligned with the context positions (global indices
            # [cur_len - ctx_len, cur_len - 1]): action a_t conditions the
            # prediction of state s_{t+1}.
            ctx_act = action_tokens[:, cur_len - ctx_len: cur_len]

            out = self.predictor(ctx, ctx_act)  # (B, ctx_len, embed_dim)
            next_token = out[:, -1:]             # (B, 1, embed_dim)

            preds.append(next_token)
            seq = torch.cat([seq, next_token], dim=1)

        return torch.cat(preds, dim=1)  # (B, W_out, embed_dim)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *args):

        # -----------------------------
        # Split incoming tensors
        # -----------------------------
        diag = args[: self.n_diag]
        act_past = args[self.n_diag: self.n_diag + self.n_act]
        act_fut = args[self.n_diag + self.n_act: self.n_diag + 2 * self.n_act]

        # -----------------------------
        # CNN ENCODERS -> tokens
        # -----------------------------
        # diagnostics -> state tokens (B, W_in, embed_dim)
        state = self._encode_group(self.diag_branches, diag)
        state_tokens = self.state_proj(state)

        # actuators -> action tokens over the full horizon (past + future)
        act_past_emb = self._encode_group(self.act_past_branches, act_past)
        act_fut_emb = self._encode_group(self.act_fut_branches, act_fut)
        action = torch.cat([act_past_emb, act_fut_emb], dim=1)  # (B, W_in+W_fut, n_act*d_var)
        action_tokens = self.action_proj(action)                # (B, W_in+W_fut, embed_dim_act)

        # -----------------------------
        # TRANSFORMER ROLLOUT (predictor)
        # -----------------------------
        pred_latent = self._rollout(state_tokens, action_tokens)  # (B, W_out, embed_dim)

        # -----------------------------
        # DECODE predicted latents
        # -----------------------------
        dec_out = self.dec_proj(pred_latent)  # (B, W_out, dec_dim)

        outputs = []
        for branch, goal_shape in zip(self.output_branches, self.output_shapes):
            out = checkpoint(self._run_cnn_decoder, branch, goal_shape, dec_out, use_reentrant=False)
            outputs.append(out)

        return outputs
