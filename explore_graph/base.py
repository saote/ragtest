from typing import Any
import pandas as pd
from graphrag.model import Entity
from dataclasses import dataclass,field

@dataclass
class QueryResult:
    """A Structured Search Result."""
    find_answer: bool
    response: str | dict[str, Any] | list[dict[str, Any]]
    explanation: str
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    # actual text strings that are in the context window, built from context_data
    context_text: str | list[str] | dict[str, str]
    num_iter: int = -1
    completion_time: float = -1
    prompt_tokens: int = -1

@dataclass
class ExploreResult:
    summary: str = ''
    selected_entities: list[Entity] = field(default_factory=list)
