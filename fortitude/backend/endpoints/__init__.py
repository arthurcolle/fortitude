from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel
from ..models import FortitudeBaseModel
from ..mcp.client import MCPClient

class Endpoint:
    """Base class for Fortitude endpoints that can be registered as tools"""
    def __init__(self, model: Type[FortitudeBaseModel], name: str = None):
        self.model = model
        self.name = name or model.__name__
        self.mcp_client = MCPClient()
        
    def register(self):
        """Register this endpoint as a tool in the registry"""
        # Logic to register with https://arthurcolle--registry.modal.run
        pass
        
    async def sample_llm(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000) -> str:
        """Sample from LLM using MCP"""
        try:
            return await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
        except Exception as e:
            # Fallback to a simpler implementation if MCP is not available
            return f"Error sampling from LLM: {str(e)}. MCP support may not be available in this client."

class CRUDEndpoint(Endpoint):
    """Endpoint with CRUD operations for a model"""
    
    async def create(self, data: Dict[str, Any]) -> FortitudeBaseModel:
        """Create a new instance of the model"""
        instance = self.model(**data)
        # Logic to save to database
        return instance
    
    async def read(self, id: str) -> FortitudeBaseModel:
        """Read an existing instance of the model"""
        # Logic to retrieve from database
        pass
    
    async def update(self, id: str, data: Dict[str, Any]) -> FortitudeBaseModel:
        """Update an existing instance of the model"""
        # Logic to update in database
        pass
    
    async def delete(self, id: str) -> bool:
        """Delete an existing instance of the model"""
        # Logic to delete from database
        return True
    
    async def list(self) -> List[FortitudeBaseModel]:
        """List all instances of the model"""
        # Logic to list from database
        return []
    
    async def analyze_with_llm(self, id: str, question: str) -> str:
        """Analyze model data with LLM using MCP"""
        try:
            # Get the model instance
            instance = await self.read(id)
            if not instance:
                return f"Could not find {self.model.__name__} with id {id}"
                
            # Format data for the LLM
            data_str = instance.model_dump_json(indent=2)
            
            # Create prompt
            prompt = f"Based on this {self.model.__name__} data:\n{data_str}\n\nQuestion: {question}"
            system_prompt = f"You are an expert at analyzing {self.model.__name__} data. Provide concise, accurate insights."
            
            # Sample from LLM
            return await self.sample_llm(prompt, system_prompt=system_prompt)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
