"""Pre-defined test scenarios for simulation."""

SCENARIOS = {
    "normal": {
        "name": "Normal Operation",
        "description": "No fire or smoke. All systems normal.",
        "duration_seconds": 30,
        "expected_status": "NORMAL",
    },
    "fire": {
        "name": "Fire Detection",
        "description": "Sustained fire detection in zone.",
        "duration_seconds": 30,
        "expected_status": "ALARM",
    },
    "smoke": {
        "name": "Smoke Detection",
        "description": "Sustained smoke detection.",
        "duration_seconds": 30,
        "expected_status": "ALARM",
    },
    "fire_and_smoke": {
        "name": "Fire and Smoke",
        "description": "Both fire and smoke detected.",
        "duration_seconds": 30,
        "expected_status": "ALARM",
    },
    "intermittent_fire": {
        "name": "Intermittent Fire",
        "description": "Fire appears intermittently. Tests temporal filter.",
        "duration_seconds": 30,
        "expected_status": "MONITORING",
    },
    "false_positive": {
        "name": "False Positive Test",
        "description": "Low confidence detections. Should NOT trigger alarm.",
        "duration_seconds": 30,
        "expected_status": "NORMAL",
    },
    "fault": {
        "name": "System Fault",
        "description": "Simulated camera/system fault.",
        "duration_seconds": 10,
        "expected_status": "FAULT",
    },
}


def get_scenario(name: str) -> dict:
    """Get scenario configuration by name."""
    return SCENARIOS.get(name, SCENARIOS["normal"])


def list_scenarios() -> list:
    """List all available scenarios."""
    return [
        {"id": k, **v}
        for k, v in SCENARIOS.items()
    ]
