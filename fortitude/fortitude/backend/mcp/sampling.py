from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message in the sampling conversation"""
    role: Literal["user", "assistant"]
    content: Dict[str, Any]


class ModelPreferences(BaseModel):
    """Preferences for model selection"""
    hints: Optional[List[Dict[str, Optional[str]]]] = None
    costPriority: Optional[float] = None  # 0-1
    speedPriority: Optional[float] = None  # 0-1
    intelligencePriority: Optional[float] = None  # 0-1


class SamplingRequest(BaseModel):
    """Request parameters for sampling/createMessage"""
    messages: List[Message]
    modelPreferences: Optional[ModelPreferences] = None
    systemPrompt: Optional[str] = None
    includeContext: Optional[Literal["none", "thisServer", "allServers"]] = None
    temperature: Optional[float] = None
    maxTokens: int
    stopSequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SamplingResponse(BaseModel):
    """Response from sampling/createMessage"""
    model: str
    stopReason: Optional[str] = None
    role: Literal["user", "assistant"]
    content: Dict[str, Any]
