import aiohttp
import json
from typing import Dict, Any, Type
from .models import FortitudeBaseModel
from .endpoints import Endpoint

class ToolRegistry:
    """Client for interacting with the external tool registry"""
    
    def __init__(self, registry_url: str = "https://arthurcolle--registry.modal.run"):
        self.registry_url = registry_url
    
    async def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], output_schema: Dict[str, Any]):
        """Register a tool with the registry"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema
            }
            
            try:
                async with session.post(f"{self.registry_url}/register", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Failed to register tool: {error_text}")
                        return None
            except Exception as e:
                print(f"Error registering tool: {str(e)}")
                return None
    
    async def get_tool(self, name: str):
        """Get a tool from the registry"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.registry_url}/tools/{name}") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
            except Exception as e:
                print(f"Error getting tool: {str(e)}")
                return None
    
    @staticmethod
    def model_to_schema(model: Type[FortitudeBaseModel]) -> Dict[str, Any]:
        """Convert a Pydantic model to a JSON schema"""
        return model.schema()
    
    @staticmethod
    def endpoint_to_tool(endpoint: Endpoint):
        """Convert an endpoint to a tool definition"""
        model = endpoint.model
        model_name = model.__name__
        
        input_schema = {
            "type": "object",
            "properties": {}
        }
        
        # Use model schema as output schema
        output_schema = ToolRegistry.model_to_schema(model)
        
        return {
            "name": endpoint.name,
            "description": f"Interact with {model_name} data",
            "input_schema": input_schema,
            "output_schema": output_schema
        }
