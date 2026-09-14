# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""NVIDIA NIM — a first-class declared provider (G8 remake).

One transport face (openai_chat on ``integrate.api.nvidia.com``) and
the hardest per-model capability surface in the tree: NVIDIA hosts
many families on shared infrastructure, and every family spells its
thinking controls DIFFERENTLY — top-level ``reasoning_effort`` on
some, ``chat_template_kwargs`` booleans on others, nothing at all on
the plain ones. The capability matrix below is grounded in the hosted
API docs and live probes (deliberate invalid values read back the
real vocabularies from the 400s; the heavy reasoners that hang on
probes are doc-backed and marked).

Three tiers, all declared — no provider code:

- Top-level-effort families declare ``effort_accept`` and the ladder
  folds everything else (with a visible note per fold).
- Chat-template families declare a ``toggle_field``; the emission
  writes the boolean into ``chat_template_kwargs`` and the top-level
  word never rides.
- Plain families strip the effort word entirely.

Unknown models (the catalog moves — 81 ids today, retirements weekly)
honestly fall to the protocol base vocabulary with visible notes
rather than fabricated mappings: we map what the docs prove, per the
"only some" ruling. The legacy hand-coded handler — five family
branches in ``extra_body`` shapes that the native path never ran — is
gone; every row below replaces it on the live path.
"""

from __future__ import annotations

from .provider_interface import ProviderInterface


class NvidiaProvider(ProviderInterface):
    """NVIDIA's hosted NIM API over the chat-completions face."""

    # Registry registers this module as `nvidia_nim`; JSON config sections
    # address the same identity (one key everywhere).
    config_key_alias = "nvidia_nim"

    # -- transport (the envelope) ---------------------------------------
    speaks = ("openai_chat",)
    native_streaming_supported = True
    default_api_base = "https://integrate.api.nvidia.com/v1"

    # NOTE(for-removal with the cost phase): NIM pricing is not wired.
    skip_cost_calculation = True

    # -- payload shaping: the capability matrix -------------------------
    # Cascade order matters: general rows first, narrower rows later —
    # a later matching row overrides conflicting keys and inherits the
    # rest (CSS cascade). off_word is the family's own OFF spelling on
    # the wire (nvidia families use "none", not "off").
    model_rules = (
        # Kimi K2.x (LIVE catalog: kimi-k2.6): thinking boolean inside
        # chat_template_kwargs; no effort word on the wire at all.
        {
            "match": "*kimi-k2*",
            "effort_accept": ["off", "low", "high"],
            "toggle_field": "chat_template_kwargs.thinking",
            "toggle_on": True,
            "toggle_off": False,
        },
        # Kimi K3: top-level effort, ALWAYS-ON thinking (no off value —
        # an OFF request drops the control with a note rather than
        # fabricating "none", which the model rejects).
        {
            "match": "*kimi-k3*",
            "effort_accept": ["low", "high", "max"],
        },
        # DeepSeek V4 (LIVE: v4-flash-0731, v4-pro): top-level effort
        # none|high|max; the service translates it server-side into the
        # model's chat_template_kwargs.
        {
            "match": "*deepseek-v4*",
            "effort_accept": ["off", "high", "max"],
            "off_word": "none",
        },
        # GLM 5.x: top-level effort, always-on thinking. 5.2 accepts a
        # narrower vocabulary than 5.3 — the later row tightens it.
        {
            "match": "*glm-5.3*",
            "effort_accept": ["low", "high", "max"],
        },
        {
            "match": "*glm-5.2*",
            "effort_accept": ["high", "max"],
        },
        # GPT-OSS (LIVE: gpt-oss-20b): live-confirmed low|medium|high —
        # no off value documented; OFF drops with a note.
        {
            "match": "*gpt-oss*",
            "effort_accept": ["low", "medium", "high"],
        },
        # Muse Glimmer: LIVE-probed full seven-rung vocabulary (the 400
        # on an invalid value listed every accepted word).
        {
            "match": "*muse-glimmer*",
            "effort_accept": [
                "off",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ],
            "off_word": "none",
        },
        # Nemotron 3 Ultra: none|medium|high plus a reasoning budget
        # (-1 disables; clamped to the documented range).
        {
            "match": "*nemotron-3-ultra*",
            "effort_accept": ["off", "medium", "high"],
            "off_word": "none",
            "clamp": {"reasoning_budget": [-1, 32768]},
        },
        # Mistral medium (docs; not in the live catalog today): the
        # none|high subset — the ladder folds everything else up.
        {
            "match": "*mistral-medium*",
            "effort_accept": ["off", "high"],
            "off_word": "none",
        },
        # DiffusionGemma / Qwen 3.5: enable_thinking boolean inside the
        # chat template — toggle-only families.
        {
            "match": "*diffusiongemma*",
            "toggle_field": "chat_template_kwargs.enable_thinking",
        },
        {
            "match": "*qwen3.5*",
            "toggle_field": "chat_template_kwargs.enable_thinking",
        },
        # Plain families (llama, step, the legacy smalls): no thinking
        # controls exist — the effort word strips instead of riding to
        # a vLLM 400.
        {
            "match": "*llama*",
            "strip": ["reasoning_effort"],
        },
        {
            "match": "*step-*",
            "strip": ["reasoning_effort"],
        },
    )

    # -- discovery -----------------------------------------------------------
    # Model listing is the shared, protocol-aware interface implementation
    # (the LIVE catalog — 81 ids at research time; listed ≠ deployed, so
    # an unentitled model 404s at request time and rotates honestly).
