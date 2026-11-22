"""Inference adapter for the fine-tuned CREMA-D wav2vec 2.0 checkpoint."""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .baseline import LABELS, SAMPLE_RATE


MODEL_ID = "facebook/wav2vec2-base"
MODEL_REVISION = "bfb1337a27fc4a418743753d6c965436844ab0a2"


class Wav2Vec2Predictor:
    """Load a trained checkpoint once and return relative class scores."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except ImportError as exc:
            raise RuntimeError(
                "Wav2Vec 2.0 dependencies are missing. Install PyTorch and "
                "the packages in ml/requirements-wav2vec.txt."
            ) from exc

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Wav2Vec 2.0 checkpoint not found: {checkpoint_path}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("SER_WAV2VEC_DEVICE must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "SER_WAV2VEC_DEVICE is set to cuda, but CUDA is unavailable. "
                "Use cpu or install a matching CUDA-enabled PyTorch build."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_id") != MODEL_ID:
            raise ValueError("Checkpoint model ID does not match the expected wav2vec model.")
        if checkpoint.get("model_revision") != MODEL_REVISION:
            raise ValueError("Checkpoint model revision does not match the expected revision.")
        if checkpoint.get("labels") != list(LABELS):
            raise ValueError("Checkpoint labels do not match the API emotion order.")
        if checkpoint.get("sample_rate_hz") != SAMPLE_RATE:
            raise ValueError("Checkpoint sample rate does not match the API sample rate.")

        # Load only the small pinned architecture config from Hugging Face.
        # The learned weights come entirely from the local fine-tuned checkpoint.
        config = Wav2Vec2Config.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

        class Wav2VecEmotionClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = Wav2Vec2Model(config)
                self.dropout = torch.nn.Dropout(0.1)
                self.classifier = torch.nn.Linear(int(config.hidden_size), len(LABELS))
                self.encoder.feature_extractor._freeze_parameters()

            def forward(self, input_values: Any, input_lengths: Any) -> Any:
                hidden = self.encoder(input_values=input_values).last_hidden_state
                output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
                output_lengths = output_lengths.clamp(min=1, max=hidden.shape[1])
                steps = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                mask = steps < output_lengths[:, None]
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / output_lengths.unsqueeze(-1).to(hidden.dtype)
                return self.classifier(self.dropout(pooled))

        model = Wav2VecEmotionClassifier()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLE_RATE,
            padding_value=0.0,
            padding_side="right",
            do_normalize=True,
            return_attention_mask=False,
        )
        self._device_name = (
            torch.cuda.get_device_name(self._device) if device == "cuda" else "CPU"
        )

    @property
    def device_name(self) -> str:
        return self._device_name

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        if sample_rate != SAMPLE_RATE:
            divisor = gcd(int(sample_rate), SAMPLE_RATE)
            audio = resample_poly(
                audio,
                up=SAMPLE_RATE // divisor,
                down=int(sample_rate) // divisor,
            ).astype(np.float32, copy=False)
        audio = np.asarray(audio, dtype=np.float32)
        features = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=False,
        ).input_values[0]
        inputs = self._torch.as_tensor(features, dtype=self._torch.float32).unsqueeze(0)
        inputs = inputs.to(self._device)
        lengths = self._torch.tensor([inputs.shape[1]], dtype=self._torch.long, device=self._device)

        with self._torch.inference_mode():
            logits = self._model(inputs, lengths)
            scores = self._torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        result = {label: float(score) for label, score in zip(LABELS, scores)}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("The wav2vec model returned non-finite scores.")
        return result
