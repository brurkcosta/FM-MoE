import torch
import torch.nn as nn
from chronos import BaseChronosPipeline


class ChronosExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._cache = {}

    def _get_pipeline(self):
        if "pipe" not in self._cache:
            print("[cache] carregando chronos-bolt-base", flush=True)
            self._cache["pipe"] = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        return self._cache["pipe"]

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        pipe = self._get_pipeline()
        _, out_mean = pipe.predict_quantiles(
            context=input_tensor,
            prediction_length=prediction_length,
        )
        return out_mean