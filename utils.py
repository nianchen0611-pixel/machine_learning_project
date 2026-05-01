import re


def shorten_text(text, max_length=350):
    if not isinstance(text, str):
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."

# Normalize movie titles for flexible searching.
def normalize_title(title):
    """
    Example:
    "Spider-Man" -> "spiderman"
    "spider man" -> "spiderman"
    "SPIDER MAN" -> "spiderman"

    """
    if not isinstance(title, str):
        return ""

    title = title.lower()
    # Remove common title words that users may skip or include differently
    stop_words = {"the", "a", "an", "and", "of"}
    # Keep only words and numbers
    words = re.findall(r"[a-z0-9]+", title)
    # Remove stop words
    words = [word for word in words if word not in stop_words]

    return "".join(words)