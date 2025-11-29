# src/rotator_library/providers/antigravity_utils/request_helpers.py
"""
Request helper functions for Antigravity API.

Provides utility functions for generating request identifiers, session IDs,
and project IDs used in Antigravity API requests.
"""

import random
import uuid


def generate_request_id() -> str:
    """
    Generate Antigravity request ID in the format: agent-{uuid}.
    
    Returns:
        Request ID string (e.g., "agent-a1b2c3d4...")
    """
    return f"agent-{uuid.uuid4()}"


def generate_session_id() -> str:
    """
    Generate Antigravity session ID in the format: -{random_number}.
    
    Returns:
        Session ID string with 19-digit random number
    """
    n = random.randint(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)
    return f"-{n}"


def generate_project_id() -> str:
    """
    Generate fake project ID in the format: {adj}-{noun}-{random}.
    
    Returns:
        Project ID string (e.g., "useful-fuze-a1b2c")
    """
    adjectives = ["useful", "bright", "swift", "calm", "bold"]
    nouns = ["fuze", "wave", "spark", "flow", "core"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:5]}"