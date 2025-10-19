import numpy as np


def _sanitize_report_dict(report: dict) -> dict:
    """Recursively cast numpy numbers in a dictionary to standard Python types."""
    sanitized = {}
    for key, value in report.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_report_dict(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        else:
            sanitized[key] = value
    return sanitized
