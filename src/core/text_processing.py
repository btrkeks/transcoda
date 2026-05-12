def token_ids_to_string(
    token_ids: list[int],
    i2w: dict[int, str],
    pad_token_id: int,
    add_header: bool = True,
) -> str:
    """
    Converts a sequence of token IDs to a human-readable string, cleaning special tokens.

    Args:
        token_ids: List of token IDs to convert.
        i2w: Index-to-word vocabulary mapping.
        pad_token_id: The padding token ID.
        add_header: Whether to add the Humdrum **kern header (default: True).

    Returns:
        Formatted kern string with optional Humdrum header.
    """
    tokens = []
    for token_id in token_ids:
        if token_id == pad_token_id:
            break
        token = i2w.get(token_id, "")
        if token in ("<bos>", "<eos>"):
            continue
        tokens.append(token)

    # Reconstruct kern format from tokens
    text = "".join(tokens)

    # Add Humdrum kern format header if requested
    if add_header:
        text = "**kern\t**kern\n" + text

    return text
