"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI gpt-5.x and Responses-API relays onto the chat wire) emit one
``reasoning_content`` delta per *completed* summary part, each opening with a bold heading.
The chat wire lacks the Responses API's ``summary_index`` delimiter (verified live on Nous
Portal ``openai/gpt-5.6-sol``), so plain concatenation glues ``**One****Two**`` into one
half-bold paragraph. The boundary is re-derived from a delta opening a bold heading, matching
the blank-line join Hermes' own Responses adapter does.
"""

from __future__ import annotations

from typing import Any

from agent.message_content import flatten_message_text

__all__ = ["separate_glued_reasoning_blocks", "ReasoningDeltaAccumulator"]


def separate_glued_reasoning_blocks(previous: str, delta: Any) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    A break is inserted when *delta* opens a *closed* bold heading and *previous* is mid-line
    (heading butting heading, or prose butting heading). Token-streamed reasoning is left
    alone: its deltas carry their own whitespace, and a fragment that merely opens emphasis
    (``**`` alone) is not a part boundary — summary parts carry the whole heading in one delta.
    """
    # Relays also emit content-part lists/dicts; fragments carry their own whitespace.
    delta = flatten_message_text(delta, sep="")
    glued = previous and delta and not previous[-1].isspace() and delta.startswith("**") and "**" in delta[2:]
    return f"\n\n{delta}" if glued else delta


# Carry #59009 (5b474ad235): shorter overlaps may be real token repetition.
_MIN_REASONING_OVERLAP = 24


def normalize_reasoning_delta(accumulated: str, delta: str) -> str:
    """Trim full snapshots/echoes and long suffix-prefix reconnect redeliveries.

    Never discard arbitrary contained text: ordinary reasoning repeats words.
    Even whole-buffer matches need the overlap gate, or early repeated tokens
    ("the", "the") disappear. This is a provider-redelivery heuristic, not a
    general-purpose prose deduplicator.
    """
    if not delta or not accumulated:
        return delta
    if len(accumulated) >= _MIN_REASONING_OVERLAP and delta.startswith(accumulated):
        return delta[len(accumulated):]
    for overlap in range(min(len(accumulated), len(delta)), _MIN_REASONING_OVERLAP - 1, -1):
        if accumulated.endswith(delta[:overlap]):
            return delta[overlap:]
    return delta


class ReasoningDeltaAccumulator:
    """Request-local reasoning for either consumer or Relay (never shared).

    Match echoes against RAW text: adding the display's paragraph breaks first
    changes the prefix and defeats dedup for bold-heading cumulative snapshots.
    Only normalized, separated parts reach storage and the reasoning callback.
    """

    def __init__(self) -> None:
        self._raw_parts: list[str] = []
        self.parts: list[str] = []

    def feed(self, delta: Any) -> str:
        delta = flatten_message_text(delta, sep="")
        # Short tokens cannot meet the overlap gate. Avoid copying the whole
        # response per token; only summary-sized chunks need the raw history.
        if len(delta) >= _MIN_REASONING_OVERLAP:
            delta = normalize_reasoning_delta("".join(self._raw_parts), delta)
        if not delta:
            return ""
        self._raw_parts.append(delta)
        text = separate_glued_reasoning_blocks(self.parts[-1] if self.parts else "", delta)
        self.parts.append(text)
        return text
