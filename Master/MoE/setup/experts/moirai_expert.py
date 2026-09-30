import torch
import torch.nn as nn
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class MoiraiExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self._cache = {}

    def _get_module(self):
        if "module" not in self._cache:
            print("[cache] carregando moirai-1.1-R-large", flush=True)
            module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large")
            module.to(self.device)
            for p in module.parameters():
                p.requires_grad = False
            self._cache["module"] = module
        module = self._cache["module"]
        module.eval()
        return module

    def _get_forecast(self, context_length: int, prediction_length: int):
        key = (context_length, prediction_length)
        if key not in self._cache:
            model = MoiraiForecast(
                module=self._get_module(),
                prediction_length=prediction_length,
                context_length=context_length,
                patch_size=16,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            model.to(self.device)
            model.eval()
            self._cache[key] = model
        return self._cache[key]

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        model = self._get_forecast(input_tensor.shape[1], prediction_length)

        outputs = []
        for i in range(input_tensor.size(0)):
            past_target = input_tensor[i].unsqueeze(0).unsqueeze(-1)
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

            forecast = model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )
            out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1).to(self.device)
            outputs.append(out_row)

        return torch.cat(outputs, dim=0)