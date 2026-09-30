import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TimeMoEExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._model = None          # carregado na primeira chamada

    def _get_model(self):
        if self._model is None:
            print("[cache] carregando TimeMoE-200M (uma única vez)", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Maple728/TimeMoE-200M",
                trust_remote_code=True,
                device_map=self.device,
            )
            model.to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._model = model
        return self._model

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        model = self._get_model()
        input_tensor = input_tensor.to(self.device)

        out = model.generate(
            input_tensor,
            max_new_tokens=prediction_length
        )

        return out[:, -prediction_length:]