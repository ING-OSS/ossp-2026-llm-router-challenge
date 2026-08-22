"""The one student architecture and objective retained by this package."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel


def _masked_mean(hidden: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    count = weights.sum(dim=1)
    pooled = (hidden * weights).sum(dim=1) / count.clamp_min(1.0)
    return torch.where((count > 0).expand_as(pooled), pooled, fallback)


class TinyResponseStudent(nn.Module):
    """Jointly predict q(light), q(ax31), and q(think)."""

    def __init__(
        self,
        model_dir: Path,
        *,
        hidden_size: int,
        fusion_size: int,
        trainable_layers: int,
        dropout: float,
        initial_quality: Sequence[float],
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
        layers = self.encoder.encoder.layer
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        selected_layers = len(layers) if trainable_layers == 0 else trainable_layers
        if not 1 <= selected_layers <= len(layers):
            raise ValueError("invalid trainable layer count")
        for layer in layers[-selected_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.trainable_layers = selected_layers
        self.dropout = nn.Dropout(dropout)

        encoder_size = int(self.encoder.config.hidden_size)
        self.semantic_projection = nn.Linear(encoder_size * 4, hidden_size, bias=False)
        self.semantic_norm = nn.LayerNorm(hidden_size)
        combined_size = hidden_size
        self.fusion_in = nn.Linear(combined_size, fusion_size, bias=False)
        self.fusion_out = nn.Linear(fusion_size, hidden_size, bias=False)
        self.fusion_skip = nn.Linear(combined_size, hidden_size, bias=False)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.quality_head = nn.Linear(hidden_size, 3)

        nn.init.zeros_(self.quality_head.weight)
        initial = torch.as_tensor(initial_quality, dtype=torch.float32).clamp(
            1e-4, 1 - 1e-4
        )
        with torch.no_grad():
            self.quality_head.bias.copy_(torch.logit(initial))

    def set_training_mode(self) -> None:
        self.train()
        self.encoder.embeddings.eval()
        for layer in self.encoder.encoder.layer[: -self.trainable_layers]:
            layer.eval()

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state
        global_mean = _masked_mean(hidden, attention_mask.bool(), hidden[:, 0])
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        midpoint = lengths // 2
        content_mask = (
            attention_mask.bool()
            & (positions > 0)
            & (positions < lengths - 1)
        )
        first_half_mean = _masked_mean(
            hidden, content_mask & (positions < midpoint), global_mean
        )
        second_half_mean = _masked_mean(
            hidden, content_mask & (positions >= midpoint), global_mean
        )
        semantic = self.semantic_norm(
            self.dropout(
                F.gelu(
                    self.semantic_projection(
                        torch.cat(
                            (
                                hidden[:, 0],
                                global_mean,
                                first_half_mean,
                                second_half_mean,
                            ),
                            dim=1,
                        )
                    )
                )
            )
        )
        combined = semantic
        return self.fusion_norm(
            F.gelu(
                self.fusion_skip(combined)
                + self.fusion_out(self.dropout(F.gelu(self.fusion_in(combined))))
            )
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        representation = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.quality_head(self.dropout(representation))
        return logits, F.normalize(representation, p=2, dim=1)


class OnnxStudent(nn.Module):
    """Deployment wrapper: emit probabilities rather than training logits."""

    def __init__(self, student: TinyResponseStudent, temperature: float = 1.0) -> None:
        super().__init__()
        self.student = student
        self.temperature = float(temperature)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        logits, _representation = self.student(
            input_ids, attention_mask, token_type_ids
        )
        return torch.sigmoid(logits / self.temperature)


def distillation_loss(
    quality_logits: Tensor,
    representation: Tensor,
    observed_quality: Tensor,
    trials: Tensor,
    teacher_quality: Tensor,
    teacher_variance: Tensor,
    teacher_representation: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Observed response plus the gain, rank, and relation signals that route."""

    observed_rows = F.binary_cross_entropy_with_logits(
        quality_logits, observed_quality, reduction="none"
    ).mean(dim=1)
    supervised = (observed_rows * trials).sum() / trials.sum().clamp_min(1e-8)
    predicted_quality = quality_logits.sigmoid()
    confidence = 1.0 / (1.0 + 100.0 * teacher_variance.mean(dim=1))
    predicted_gain = predicted_quality[:, 1:] - predicted_quality[:, :-1]
    teacher_gain = teacher_quality[:, 1:] - teacher_quality[:, :-1]
    gain_rows = F.smooth_l1_loss(
        predicted_gain, teacher_gain, beta=0.025, reduction="none"
    ).mean(dim=1)
    gain = (gain_rows * confidence).sum() / confidence.sum().clamp_min(1e-8)

    teacher_delta = teacher_gain - torch.roll(teacher_gain, 1, dims=0)
    predicted_delta = predicted_gain - torch.roll(predicted_gain, 1, dims=0)
    rank_target = torch.sigmoid(teacher_delta / 0.025).detach()
    rank_rows = F.binary_cross_entropy_with_logits(
        predicted_delta / 0.025, rank_target, reduction="none"
    ).mean(dim=1)
    rank_confidence = teacher_delta.abs().mean(dim=1).clamp(max=0.25) / 0.25
    rank_weights = confidence * rank_confidence
    rank = (rank_rows * rank_weights).sum() / rank_weights.sum().clamp_min(1e-8)

    student_similarity = representation @ representation.T
    normalized_teacher = F.normalize(teacher_representation, p=2, dim=1)
    teacher_similarity = normalized_teacher @ normalized_teacher.T
    diagonal = torch.eye(
        len(representation), dtype=torch.bool, device=representation.device
    )
    relation = F.smooth_l1_loss(
        student_similarity[~diagonal],
        teacher_similarity[~diagonal].detach(),
        beta=0.05,
    )
    total = supervised + gain + 0.1 * rank + 0.1 * relation
    return total, {
        "supervised": supervised.detach(),
        "gain": gain.detach(),
        "rank": rank.detach(),
        "relation": relation.detach(),
    }


def optimizer_groups(
    model: TinyResponseStudent,
) -> tuple[list[Tensor], list[Tensor]]:
    """Use Muon for hidden matrices and AdamW for embeddings/head/biases."""

    muon: list[Tensor] = []
    adam: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_muon = (
            parameter.ndim == 2
            and ".embeddings." not in name
            and not name.startswith("quality_head")
        )
        (muon if use_muon else adam).append(parameter)
    if not muon or not adam:
        raise RuntimeError("Muon and AdamW parameter groups must both be non-empty")
    return muon, adam
