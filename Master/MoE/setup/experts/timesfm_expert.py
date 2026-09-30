import torch
import torch.nn as nn
import timesfm


class TimesFMExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._cache = {}

    def _get_model(self, prediction_length: int):
        key = ("timesfm", prediction_length)
        if key not in self._cache:
            print(f"[cache] carregando TimesFM (horizon={prediction_length})", flush=True)
            self._cache[key] = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if self.device.lower() == "cuda" else "cpu",
                    per_core_batch_size=32,
                    horizon_len=prediction_length,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
                ),
            )
        return self._cache[key]

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        model = self._get_model(prediction_length)
        input_np = input_tensor.clone().detach().cpu().numpy()
        out, _ = model.forecast(input_np)
        return torch.from_numpy(out).float()