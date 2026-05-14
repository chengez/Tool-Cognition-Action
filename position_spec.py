"""
Position specification module for flexible token position specification.

Supports both integer-based positions (e.g., -1 for last token) and
pattern-based positions (e.g., "<tool_call>" to find a specific token pattern).
"""

from dataclasses import dataclass
from typing import Union, List
import re
import torch


def sanitize_for_filename(pattern: str) -> str:
    """
    Sanitize a pattern string for use in filenames.

    Rules:
    1. Replace non-alphanumeric characters (except underscore/hyphen) with underscores
    2. Collapse multiple underscores
    3. Strip leading/trailing underscores
    4. Limit length to 50 characters
    """
    result = re.sub(r'[^a-zA-Z0-9_-]', '_', pattern)
    result = re.sub(r'_+', '_', result)
    result = result.strip('_')
    if len(result) > 50:
        result = result[:50]
    return result if result else "pattern"


@dataclass
class PositionSpec:
    """
    Represents a token position specification.

    Two modes:
    1. Integer mode: position is an int (positive or negative indexing)
    2. Pattern mode: position is a string pattern with token_offset and occurrence
    """
    position: Union[int, str]
    token_offset: int = 0
    occurrence: int = 0

    @property
    def is_pattern(self) -> bool:
        """Returns True if this is a pattern-based position specification."""
        return isinstance(self.position, str)

    @property
    def is_integer(self) -> bool:
        """Returns True if this is an integer-based position specification."""
        return isinstance(self.position, int)

    def to_filename_safe(self) -> str:
        """
        Convert position spec to a filesystem-safe string for filenames.

        Examples:
            - Integer -1 -> "K-1"
            - Pattern "<tool_call>" with offset=-1, occurrence=0 -> "P_tool_call_O0_T-1"
        """
        if self.is_integer:
            return f"K{self.position}"
        else:
            safe_pattern = sanitize_for_filename(self.position)
            return f"P_{safe_pattern}_O{self.occurrence}_T{self.token_offset}"

    @classmethod
    def from_cli_args(
        cls, position_str: str, token_offset: int = 0, occurrence: int = 0
    ) -> "PositionSpec":
        """
        Create PositionSpec from CLI arguments.

        Args:
            position_str: Either an integer string (e.g., "-1") or a pattern string
            token_offset: Offset within the tokenized pattern (default 0)
            occurrence: Which occurrence of the pattern to use (default 0)
        """
        try:
            pos_int = int(position_str)
            return cls(position=pos_int)
        except ValueError:
            return cls(
                position=position_str,
                token_offset=token_offset,
                occurrence=occurrence
            )


def find_pattern_token_indices(
    query_token_ids: List[int],
    pattern_token_ids: List[int],
) -> List[int]:
    """
    Find the starting indices of all occurrences of a pattern in tokenized query.

    Args:
        query_token_ids: Tokenized query as list of ints
        pattern_token_ids: Tokenized pattern as list of ints

    Returns:
        List of starting indices where the pattern occurs
    """
    pattern_len = len(pattern_token_ids)
    occurrences = []

    for i in range(len(query_token_ids) - pattern_len + 1):
        if query_token_ids[i:i + pattern_len] == pattern_token_ids:
            occurrences.append(i)

    return occurrences


def resolve_position_spec(
    position_spec: PositionSpec,
    query_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    query_idx: int = 0
) -> int:
    """
    Resolve a PositionSpec to an actual token index for a specific query.

    Args:
        position_spec: The position specification to resolve
        query_token_ids: Tokenized query as 1D tensor
        attention_mask: Attention mask for the query
        tokenizer: The tokenizer (needed to tokenize pattern strings)
        query_idx: Index of query in batch (for error messages)

    Returns:
        The resolved token index

    Raises:
        ValueError: If position is out of bounds or pattern not found
    """
    seq_length = int(attention_mask.sum().item())

    if position_spec.is_integer:
        position = position_spec.position
        if position >= 0:
            token_idx = position
            if token_idx >= seq_length:
                raise ValueError(
                    f"Position {position} out of bounds for sequence length "
                    f"{seq_length} (query {query_idx})"
                )
        else:
            token_idx = seq_length + position
            if token_idx < 0:
                raise ValueError(
                    f"Position {position} out of bounds for sequence length "
                    f"{seq_length} (query {query_idx})"
                )
        return token_idx

    # Pattern-based position
    pattern = position_spec.position

    pattern_encoding = tokenizer(
        pattern,
        add_special_tokens=False,
        return_tensors="pt"
    )
    pattern_token_ids = pattern_encoding["input_ids"][0].tolist()

    if len(pattern_token_ids) == 0:
        raise ValueError(f"Pattern '{pattern}' tokenizes to empty sequence")

    query_ids = query_token_ids.tolist()
    occurrences = find_pattern_token_indices(query_ids, pattern_token_ids)

    if len(occurrences) == 0:
        raise ValueError(
            f"Pattern '{pattern}' not found in query {query_idx}. "
            f"Pattern tokens: {pattern_token_ids}"
        )

    occurrence = position_spec.occurrence
    if occurrence >= 0:
        if occurrence >= len(occurrences):
            raise ValueError(
                f"Occurrence {occurrence} requested but only "
                f"{len(occurrences)} occurrence(s) of pattern '{pattern}' found "
                f"in query {query_idx}"
            )
        pattern_start_idx = occurrences[occurrence]
    else:
        if abs(occurrence) > len(occurrences):
            raise ValueError(
                f"Occurrence {occurrence} out of range. "
                f"Only {len(occurrences)} occurrence(s) found in query {query_idx}"
            )
        pattern_start_idx = occurrences[occurrence]

    pattern_len = len(pattern_token_ids)
    token_offset = position_spec.token_offset

    if token_offset >= 0:
        if token_offset >= pattern_len:
            raise ValueError(
                f"Token offset {token_offset} out of bounds for pattern "
                f"of length {pattern_len}"
            )
        token_idx = pattern_start_idx + token_offset
    else:
        if abs(token_offset) > pattern_len:
            raise ValueError(
                f"Token offset {token_offset} out of bounds for pattern "
                f"of length {pattern_len}"
            )
        token_idx = pattern_start_idx + pattern_len + token_offset

    if token_idx < 0 or token_idx >= seq_length:
        raise ValueError(
            f"Resolved token index {token_idx} out of bounds for "
            f"sequence length {seq_length} (query {query_idx})"
        )

    return token_idx
