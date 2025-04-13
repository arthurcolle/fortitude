import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .sampling import SamplingRequest, SamplingResponse, Message

logger = logging.getLogger(__name__)

class MCPServer:
    """A server that implements the Model Context Protocol
    
    This server can be used to handle MCP requests from clients and process them
    using custom handlers. It provides a standardized interface for MCP operations.
    """
    
    def __init__(self, name: str = "Fortitude MCP Server", port: int = 8888):
        """Initialize the MCP server
        
        Args:
            name: Name of the MCP server
            port: Port to run the server on
        """
        self.name = name
        self.port = port
        self.app = FastAPI(title=name)
        self.methods: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        
        # Set up CORS and routes
        self._setup_middleware()
        self._setup_routes()
        
        # Register default methods
        self.register_method("sampling/createMessage", self._default_sampling_handler)
        
    def _setup_middleware(self):
        """Set up middleware for the FastAPI app"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Set up routes for the FastAPI app"""
        
        @self.app.post("/mcp")
        async def handle_mcp_request(request: Request):
            """Handle MCP requests"""
            try:
                # Parse the request body
                body = await request.json()
                
                # Validate the required fields
                if "method" not in body:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Missing 'method' field"}
                    )
                
                # Get the method handler
                method = body.get("method")
                handler = self.methods.get(method)
                
                if not handler:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Method '{method}' not found"}
                    )
                
                # Call the handler with the params
                params = body.get("params", {})
                result = await handler(params)
                
                # Return the result
                return JSONResponse(content={"result": result})
                
            except Exception as e:
                logger.exception("Error handling MCP request")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error processing request: {str(e)}"}
                )
                
        @self.app.get("/mcp/info")
        async def get_mcp_info():
            """Get information about the MCP server"""
            return {
                "name": self.name,
                "methods": list(self.methods.keys()),
                "resources": list(self.resources.keys()),
                "tools": list(self.tools.keys())
            }
        
    async def _default_sampling_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler for sampling/createMessage
        
        This is a placeholder that should be overridden with a proper implementation.
        
        Args:
            params: Parameters for sampling
            
        Returns:
            The sampling response
        """
        # This should be overridden with a proper implementation
        try:
            # Parse the request as a SamplingRequest
            request = SamplingRequest.model_validate(params)
            
            # In a real implementation, this would call an LLM
            # Here we just return a placeholder message
            return {
                "model": "placeholder-model",
                "stopReason": "endTurn",
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "This is a placeholder response from the MCP server. Please override the sampling handler with a proper implementation."
                }
            }
        except Exception as e:
            logger.exception("Error in default sampling handler")
            raise ValueError(f"Error processing sampling request: {str(e)}")
    
    def register_method(self, method_name: str, handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """Register a method handler
        
        Args:
            method_name: Name of the MCP method
            handler: Async function that handles the method calls
        """
        self.methods[method_name] = handler
        logger.info(f"Registered MCP method: {method_name}")
        
    def register_resource(self, resource_id: str, resource: Any):
        """Register a resource
        
        Args:
            resource_id: ID of the resource
            resource: The resource object
        """
        self.resources[resource_id] = resource
        logger.info(f"Registered resource: {resource_id}")
        
    def register_tool(self, tool_id: str, tool_config: Dict[str, Any]):
        """Register a tool
        
        Args:
            tool_id: ID of the tool
            tool_config: Configuration for the tool
        """
        self.tools[tool_id] = tool_config
        logger.info(f"Registered tool: {tool_id}")
        
    def register_sampling_handler(self, handler: Callable[[SamplingRequest], Awaitable[SamplingResponse]]):
        """Register a custom sampling handler
        
        Args:
            handler: Function that handles sampling requests
        """
        async def sampling_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            request = SamplingRequest.model_validate(params)
            response = await handler(request)
            return response.model_dump(exclude_none=True)
            
        self.register_method("sampling/createMessage", sampling_wrapper)
        logger.info("Registered custom sampling handler")
        
    async def start(self):
        """Start the MCP server"""
        import uvicorn
        logger.info(f"Starting MCP server on port {self.port}")
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
        
    def run(self):
        """Run the MCP server (blocking)"""
        import uvicorn
        logger.info(f"Running MCP server on port {self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)
