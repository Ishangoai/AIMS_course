"""Structure validation tool for checking report format."""

import re


def validate_structure(text: str) -> dict:
    """
    Validate that the report has proper structure.

    Args:
        text: The report text to validate

    Returns:
        Dictionary with validation results
    """
    issues = []

    # Check for introduction section
    intro_patterns = [
        r"#\s*Introduction",
        r"##\s*Introduction",
        r"\*\*Introduction\*\*",
    ]
    has_intro = any(re.search(pattern, text, re.IGNORECASE) for pattern in intro_patterns)
    if not has_intro:
        issues.append("Missing clear Introduction section")

    # Check for conclusion section
    conclusion_patterns = [
        r"#\s*Conclusion",
        r"##\s*Conclusion",
        r"\*\*Conclusion\*\*",
    ]
    has_conclusion = any(re.search(pattern, text, re.IGNORECASE) for pattern in conclusion_patterns)
    if not has_conclusion:
        issues.append("Missing clear Conclusion section")

    # Check for section headers (at least 3 including intro and conclusion)
    header_patterns = [
        r"^#\s+.+$",
        r"^##\s+.+$",
        r"\*\*.+\*\*",
    ]
    headers = []
    for pattern in header_patterns:
        headers.extend(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))

    if len(headers) < 3:
        issues.append(f"Only {len(headers)} section headers found, expected at least 3")

    # Check minimum content length per section
    sections = re.split(r"\n(?=#|##|\*\*)", text)
    if len(sections) < 3:
        issues.append("Report should have at least 3 distinct sections")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "has_introduction": has_intro,
        "has_conclusion": has_conclusion,
        "header_count": len(headers),
        "section_count": len(sections),
        "issues": issues,
        "message": "Structure is valid" if is_valid else f"Structure issues: {'; '.join(issues)}",
    }
