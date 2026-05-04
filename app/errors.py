"""Custom exceptions used by routes.py.

Route handlers raise these as HTTPException subclasses so FastAPI emits the
right status code without each handler hand-rolling its own. Kept small —
only the cases the API genuinely distinguishes from generic 4xx.
"""

from fastapi import HTTPException, status


class ResearchNotFound(HTTPException):
    def __init__(self, detail: str = "Session not found") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class PlanNotReady(HTTPException):
    def __init__(self, detail: str = "Plan not yet generated") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class Unauthorized(HTTPException):
    def __init__(self, detail: str = "Invalid or missing X-API-Key header.") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class Forbidden(HTTPException):
    def __init__(self, detail: str = "This session belongs to another API key.") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class MalformedThreadId(HTTPException):
    def __init__(self, detail: str = "Malformed thread_id") -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


__all__ = [
    "ResearchNotFound",
    "PlanNotReady",
    "Unauthorized",
    "Forbidden",
    "MalformedThreadId",
    "GraphTimeout",
]


class GraphTimeout(HTTPException):
    def __init__(self, detail: str = "Graph execution timed out.") -> None:
        super().__init__(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)
