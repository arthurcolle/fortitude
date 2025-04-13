import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel

from .models import FortitudeBaseModel
from .endpoints import Endpoint, CRUDEndpoint
from .mcp.client import MCPClient

class SamplingRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    include_context: Optional[str] = "thisServer"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = None

class FortitudeServer:
    """FastAPI server for Fortitude"""
    
    def __init__(self, app_name: str):
        self.app = FastAPI(title=f"{app_name} API")
        self.setup_middleware()
        self.models = {}
        self.endpoints = {}
        self.mcp_client = MCPClient()
        self.setup_default_routes()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this to UI server
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_default_routes(self):
        """Setup default routes for the API"""
        # Models metadata route
        @self.app.get("/api/models")
        async def get_models():
            return {
                "models": [
                    {
                        "name": model_name,
                        "fields": [
                            {
                                "name": field_name,
                                "type": str(field_info.annotation),
                                "required": field_info.default is ...,
                                "default": None if field_info.default is ... else field_info.default
                            } 
                            for field_name, field_info in model.model_fields.items()
                        ]
                    }
                    for model_name, model in self.models.items()
                ]
            }
        
        # MCP sampling endpoint
        @self.app.post("/api/sample")
        async def sample_llm(request: SamplingRequest):
            try:
                # Use the MCP client to sample from LLM
                result = await self.mcp_client.sample_text(
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    include_context=request.include_context,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                return {"result": result}
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error during MCP sampling: {str(e)}"},
                )
                
    def register_model(self, model: Type[FortitudeBaseModel]):
        """Register a model and create CRUD endpoints"""
        self.models[model.__name__] = model
        endpoint = CRUDEndpoint(model)
        self.register_endpoint(endpoint)
        return model
    
    def register_endpoint(self, endpoint: Endpoint):
        """Register an endpoint with FastAPI routes"""
        self.endpoints[endpoint.name] = endpoint
        
        # Register routes if it's a CRUD endpoint
        if isinstance(endpoint, CRUDEndpoint):
            model_name = endpoint.model.__name__.lower()
            
            # Create route
            @self.app.post(f"/api/{model_name}", response_model=endpoint.model)
            async def create(data: Dict[str, Any]):
                return await endpoint.create(data)
            
            # Read route
            @self.app.get(f"/api/{model_name}/{{id}}", response_model=endpoint.model)
            async def read(id: str):
                item = await endpoint.read(id)
                if not item:
                    raise HTTPException(status_code=404, detail="Item not found")
                return item
            
            # Update route
            @self.app.put(f"/api/{model_name}/{{id}}", response_model=endpoint.model)
            async def update(id: str, data: Dict[str, Any]):
                item = await endpoint.update(id, data)
                if not item:
                    raise HTTPException(status_code=404, detail="Item not found")
                return item
            
            # Delete route
            @self.app.delete(f"/api/{model_name}/{{id}}", response_model=bool)
            async def delete(id: str):
                result = await endpoint.delete(id)
                if not result:
                    raise HTTPException(status_code=404, detail="Item not found")
                return result
            
            # List route
            @self.app.get(f"/api/{model_name}", response_model=List[endpoint.model])
            async def list():
                return await endpoint.list()
                
            # LLM analysis route
            @self.app.post(f"/api/{model_name}/{{id}}/analyze")
            async def analyze(id: str, request: Dict[str, str]):
                if "question" not in request:
                    raise HTTPException(status_code=400, detail="'question' field is required")
                    
                result = await endpoint.analyze_with_llm(id, request["question"])
                return {"result": result}
    
    def start(self, port: int = 9997):
        """Start the FastAPI server"""
        uvicorn.run(self.app, host="0.0.0.0", port=port)
