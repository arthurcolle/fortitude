import json
import aiohttp
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from .sampling import SamplingRequest, SamplingResponse, Message, ModelPreferences


class MCPClient:
    """Client for interacting with MCP-enabled clients"""
    
    def __init__(self, endpoint_url: str = "http://localhost:8888/mcp"):
        """Initialize MCP client
        
        Args:
            endpoint_url: URL of the MCP endpoint
        """
        self.endpoint_url = endpoint_url
    
    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        """Send a sampling/createMessage request to an MCP client
        
        Args:
            request: The sampling request parameters
            
        Returns:
            The sampling response from the client
        """
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "method": "sampling/createMessage",
                    "params": request.model_dump(exclude_none=True)
                }
                
                async with session.post(self.endpoint_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data:
                            return SamplingResponse.model_validate(data["result"])
                        else:
                            raise ValueError(f"Invalid response: {data}")
                    else:
                        error_text = await response.text()
                        raise ValueError(f"Request failed with status {response.status}: {error_text}")
            except Exception as e:
                raise ValueError(f"Error during MCP sampling: {str(e)}")
    
    async def sample_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        include_context: str = "thisServer",
        max_tokens: int = 1000,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Simplified method to sample text from an LLM
        
        Args:
            prompt: The text prompt
            system_prompt: Optional system prompt
            include_context: Context inclusion level
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text from the LLM
        """
        messages = [
            Message(
                role="user",
                content={"type": "text", "text": prompt}
            )
        ]
        
        request = SamplingRequest(
            messages=messages,
            systemPrompt=system_prompt,
            includeContext=include_context,
            maxTokens=max_tokens,
            temperature=temperature,
            stopSequences=stop_sequences
        )
        
        response = await self.create_message(request)
        
        if response.content.get("type") == "text":
            return response.content.get("text", "")
        else:
            raise ValueError(f"Expected text response, got {response.content.get('type')}")
