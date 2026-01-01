def preprocess(text):
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove line breaks & tabs
    text = text.replace("\n", " ").replace("\r", " ")

    # normalize whitespace
    text = " ".join(text.split())

    return text
