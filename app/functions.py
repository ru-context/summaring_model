def calculate_summary_length(text: str, min_length: int = 30, max_length: int = 130) -> int:
    text_length = len(text.split())
    summary_length = int(text_length * 0.3)

    if summary_length < min_length:
        return min_length
    elif summary_length > max_length:
        return max_length
    else:
        return summary_length
