"""Orin-local policy runner: the shared one, without the CUDA-only placement.

Organization mirrors ``lead.evaluation.inference.policy_runner.PolicyRunner``
on purpose -- same checkpoint lookup, same ``build_policy`` call, same weight
load -- so the Orin node keeps the exact structure ``src/lead`` uses.  The only
difference is the final device placement: the shared runner finishes with
``self.policy.cuda(device=...)``, which raises on this Orin because its py3.10
torch build has no CUDA.  Inference here runs in the py3.8 TensorRT engine, so
the policy only needs to exist for the feature pipeline; ``.to(device)`` is
enough.

This lives under ``sil/orin/**`` because ``src/lead/**`` is shared code that
the sync script overwrites.
"""

from __future__ import annotations

import logging
import os
import typing

import torch

from lead.api.abstract_policy import AbstractPolicy, build_policy
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


class OrinPolicyRunner:
    """Load a policy and place it on ``device`` (no CUDA-only move)."""

    def __init__(
        self,
        lead_config: LeadConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ) -> None:
        """Load the model checkpoint found in ``model_path``.

        Args:
            lead_config: The config tree the model was trained with.
            model_path: Directory holding the trained model weights.
            device: Device to place the policy on.
            prefix: Prefix of the model weights file to load.

        Raises:
            ValueError: If ``model_path`` does not hold exactly one weights file.
        """
        self.lead_config = lead_config
        self.device = device

        weight_files = sorted(
            file
            for file in os.listdir(model_path)
            if file.startswith(prefix) and file.endswith(".pth")
        )
        if len(weight_files) != 1:
            raise ValueError(
                f"Expected exactly one '{prefix}*.pth' weight file in {model_path}, "
                f"found {len(weight_files)}: {weight_files}",
            )
        weight_path = os.path.join(model_path, weight_files[0])
        LOG.info("Loading model weight from %s", weight_path)

        self.policy: AbstractPolicy = build_policy(lead_config).to(self.device)
        if self.lead_config.training.optimization.sync_batchnorm:
            # convert_sync_batchnorm's stub widens the return type to nn.Module;
            # it always preserves the input module's actual class at runtime.
            self.policy = typing.cast(
                AbstractPolicy,
                torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.policy),
            )
        self.policy.load_state_dict(
            torch.load(weight_path, map_location=self.device, weights_only=True),
            strict=lead_config.evaluation.inference.strict_weight_load,
        )
        # The shared runner calls ``.cuda()`` here; this host is CUDA-less.
        self.policy.to(self.device).eval()

    @torch.inference_mode()
    def forward(self, data: dict[str, typing.Any]) -> typing.Any:
        """Run the policy on one batch of model inputs.

        Args:
            data: The batched model inputs, as built by the policy's
                ``features_to_batch``.

        Returns:
            The raw prediction of the policy.
        """
        # No autocast here: the quantized policy's forward runs the TensorRT
        # engine, which ignores it, and the unquantized fallback should stay in
        # fp32 on this CUDA-less host.
        return self.policy(data)
