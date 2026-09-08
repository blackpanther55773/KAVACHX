def normalize_domain(domain: str) -> str:
    valid_domains = {
        "Women Safety",
        "Cyber Fraud",
        "Medical Emergency",
    }

    if domain in valid_domains:
        return domain

    return "Women Safety"


def normalize_urgency(urgency: str) -> str:
    valid_levels = {
        "Critical",
        "High",
        "Moderate",
        "Low",
    }

    if urgency in valid_levels:
        return urgency

    return "Low"


def clean_output(text: str) -> str:
    if not text:
        return "I could not generate safe guidance."

    return text.strip()