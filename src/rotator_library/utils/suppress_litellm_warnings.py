# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Utility to suppress harmless Pydantic serialization warnings from LiteLLM.

These warnings occur due to a known litellm issue where streaming response
types (Message, StreamingChoices) have mismatched field counts during
internal serialization. The warnings don't affect functionality.

See: https://github.com/BerriAI/litellm/issues/11759

TODO: Remove this workaround once litellm patches the issue above.
      Check the GitHub issue for resolution status before removing.
"""

import os
import warnings


def suppress_litellm_serialization_warnings():
    """
    Suppress litellm's internal Pydantic serialization warnings.

    Scoped to only silence:
    - UserWarning category
    - From pydantic.main module
    - Matching "Pydantic serializer warnings: PydanticSerializationUnexpectedValue"

    Can be disabled by setting SUPPRESS_LITELLM_SERIALIZATION_WARNINGS=0
    """
    if os.getenv("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", "1") == "1":
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pydantic\.main",
            message=r"Pydantic serializer warnings:\s+PydanticSerializationUnexpectedValue",
        )
