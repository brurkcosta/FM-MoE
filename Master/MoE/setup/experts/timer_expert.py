import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TimerExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._cache = {}

    def _get_model(self):
        if "model" not in self._cache:
            print("[cache] carregando sundial-base-128m", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                "thuml/sundial-base-128m",
                trust_remote_code=True,
                device_map=self.device,
            )
            model.to(self.device)
            for p in model.parameters():
                p.requires_grad = False
            self._cache["model"] = model
        model = self._cache["model"]
        model.eval()
        return model

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        model = self._get_model()
        input_tensor = input_tensor.to(self.device)

        outputs = []
        for i in range(input_tensor.size(0)):
            past_target = input_tensor[i].unsqueeze(0)
            forecast = model.generate(
                past_target,
                max_new_tokens=prediction_length,
                num_samples=20,
            )
            out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1).to(self.device)
            outputs.append(out_row)

        return torch.cat(outputs, dim=0)