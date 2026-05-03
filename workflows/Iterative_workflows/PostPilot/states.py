from typing import TypedDict, Literal, Optional, Any
from pydantic import BaseModel

# State
class PGState(BaseModel):
    topic: str
    file: Optional[Any] = None
    post: Optional[str] = None
    context: Optional[str] = None
    evaluation: Optional[Literal["Approved", "Needs Improvement"]] = None
    feedback: Optional[str] = None
    iteration: int = 0
    max_iteration: int = 5

