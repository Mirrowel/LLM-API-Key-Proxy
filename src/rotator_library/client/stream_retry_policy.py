# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Backward-compatible import path for stream retry policy."""

from ..streaming.policy import can_retry_stream_after_error

__all__ = ["can_retry_stream_after_error"]
