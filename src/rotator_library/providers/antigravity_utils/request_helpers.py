# src/rotator_library/providers/antigravity_utils/request_helpers.py
"""
Request helper functions for Antigravity API.

Provides utility functions for generating request identifiers, session IDs,
and project IDs used in Antigravity API requests.
"""

import secrets
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
    n = secrets.randbelow(9_000_000_000_000_000_000) + 1_000_000_000_000_000_000
    return f"-{n}"


def generate_project_id() -> str:
    """
    Generate a project ID in the format: {adjective}-{noun}-{hex_suffix}.
    
    This ID is used for display/logging purposes in Antigravity API requests.
    It is NOT guaranteed to be globally unique or suitable for security-sensitive
    contexts. For globally unique identifiers, use generate_request_id() instead.
    
    The adjective/noun combination provides human-readable identification,
    while the 12-character hex suffix reduces collision probability.
    
    Returns:
        Project ID string (e.g., "bright-spark-a1b2c3d4e5f6")
    """
    adjectives = [
        "able", "aged", "airy", "alert", "alive", "ample", "apt", "avid", "aware", "azure",
        "basic", "bold", "brave", "brief", "bright", "brisk", "broad", "calm", "chief", "civic",
        "clean", "clear", "close", "cool", "coral", "crisp", "cubic", "daily", "dear", "deep",
        "dense", "eager", "early", "easy", "elite", "equal", "exact", "extra", "faint", "fair",
        "fast", "firm", "first", "fit", "fixed", "flat", "fleet", "focal", "fond", "frank",
        "fresh", "full", "game", "glad", "gold", "good", "grand", "great", "green", "grown",
        "handy", "happy", "hardy", "hasty", "heavy", "high", "holy", "hot", "huge", "humble",
        "ideal", "inner", "ionic", "joint", "jolly", "keen", "key", "kind", "known", "large",
        "last", "late", "lean", "legal", "level", "light", "live", "local", "long", "loud",
        "lovely", "loyal", "lucid", "lucky", "lunar", "lusty", "major", "merry", "mild", "mint",
        "mixed", "model", "moist", "moral", "moved", "naive", "naval", "near", "neat", "new",
        "next", "nice", "ninth", "noble", "novel", "olive", "open", "opted", "oral", "outer",
        "overt", "owing", "paced", "paid", "pale", "peak", "perky", "petty", "pink", "plain",
        "plush", "polar", "posed", "prime", "proud", "pure", "quick", "quiet", "rabid", "rapid",
        "rare", "raw", "ready", "real", "regal", "rich", "right", "rigid", "ripe", "rising",
        "rival", "rosy", "rough", "round", "royal", "rusty", "safe", "same", "sandy", "savvy",
    ]
    nouns = [
        "apex", "arch", "area", "atom", "axis", "band", "bank", "barn", "base", "beam",
        "bell", "belt", "bird", "blade", "block", "bloom", "board", "bolt", "bond", "booth",
        "bound", "bowl", "brand", "brass", "brick", "bridge", "brook", "brush", "build", "bulb",
        "cable", "cache", "cairn", "canal", "cape", "cargo", "cedar", "cell", "chain", "chair",
        "chart", "chasm", "chord", "chunk", "claim", "clasp", "claw", "clay", "cliff", "cloud",
        "coast", "coil", "coral", "core", "craft", "crane", "crate", "creek", "crest", "crypt",
        "cube", "curve", "delta", "depot", "desk", "dome", "door", "draft", "drain", "drift",
        "drive", "drum", "dune", "earth", "edge", "elm", "ember", "facet", "fault", "fern",
        "fiber", "field", "flame", "flare", "flask", "fleet", "float", "flood", "floor", "flow",
        "flux", "foam", "focus", "forge", "fork", "form", "forum", "frame", "frost", "fruit",
        "fuze", "gale", "gate", "gauge", "gear", "gem", "glade", "glass", "gleam", "globe",
        "glow", "gorge", "grain", "graph", "grasp", "grass", "gravel", "grid", "grove", "gulf",
        "hall", "haven", "heart", "heath", "helix", "helm", "hinge", "hive", "hold", "hollow",
        "horizon", "horn", "hub", "hull", "inlet", "iron", "isle", "jade", "jet", "joint",
        "kayak", "keel", "kernel", "key", "knoll", "knot", "lake", "lamp", "lance", "lane",
        "latch", "lawn", "layer", "leaf", "ledge", "lens", "lever", "light", "limb", "link",
    ]
    adj = secrets.choice(adjectives)
    noun = secrets.choice(nouns)
    suffix = secrets.token_hex(6)  # 12 hex characters
    return f"{adj}-{noun}-{suffix}"