"""FortitudeMCP - Simplified MCP server implementation for Fortitude

This module provides a simplified, Fortitude-specific implementation of the Model Context Protocol
server that makes it easy to create resources and tools for LLM interaction.
"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable, get_type_hints

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

class FortitudeMCP:
    """Fortitude implementation of an MCP server
    
    Provides a simplified interface for creating MCP servers that expose resources and tools
    to LLM clients. Compatible with the Model Context Protocol specification.
    """
    
    def __init__(self, name: str):
        """Initialize the FortitudeMCP server
        
        Args:
            name: Name of the MCP server
        """
        self.name = name
        self.app = FastAPI(title=name)
        self.resources: Dict[str, Callable] = {}
        self.tools: Dict[str, Callable] = {}
        self.prompts: Dict[str, Callable] = {}
        
        # Set up CORS and routes
        self._setup_middleware()
        self._setup_routes()
    
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
        """Set up MCP protocol routes"""
        @self.app.post("/mcp")
        async def handle_mcp_request(request: Request):
            """Main MCP protocol endpoint that handles all MCP requests"""
            try:
                # Parse the request body
                body = await request.json()
                
                # Validate the required fields
                if "method" not in body:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Missing 'method' field"}
                    )
                
                method = body.get("method")
                params = body.get("params", {})
                
                # Handle different MCP method types
                if method == "resources/list":
                    return JSONResponse(content={"result": await self._handle_list_resources()})
                elif method == "resources/read":
                    return JSONResponse(content={"result": await self._handle_read_resource(params)})
                elif method == "tools/list":
                    return JSONResponse(content={"result": await self._handle_list_tools()})
                elif method == "tools/call":
                    return JSONResponse(content={"result": await self._handle_call_tool(params)})
                elif method == "prompts/list":
                    return JSONResponse(content={"result": await self._handle_list_prompts()})
                elif method == "prompts/get":
                    return JSONResponse(content={"result": await self._handle_get_prompt(params)})
                elif method == "sampling/createMessage":
                    return JSONResponse(content={"result": await self._handle_sampling(params)})
                else:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Method '{method}' not supported"}
                    )
                
            except Exception as e:
                logger.exception("Error handling MCP request")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error processing request: {str(e)}"}
                )
        
        @self.app.get("/mcp/info")
        async def get_mcp_info():
            """Information about the MCP server capabilities"""
            return {
                "name": self.name,
                "capabilities": {
                    "resources": {
                        "list": True,
                        "read": True,
                        "subscribe": False,
                        "listChanged": False,
                    },
                    "tools": {
                        "list": True,
                        "call": True,
                        "listChanged": False,
                    },
                    "prompts": {
                        "list": len(self.prompts) > 0,
                        "get": len(self.prompts) > 0,
                        "listChanged": False,
                    }
                },
                "resources": list(self.resources.keys()),
                "tools": list(self.tools.keys()),
                "prompts": list(self.prompts.keys()) if self.prompts else []
            }
    
    async def _handle_list_resources(self) -> List[Dict[str, Any]]:
        """Handle resources/list method"""
        resources = []
        for name, handler in self.resources.items():
            # Get docstring and param info
            docstring = handler.__doc__ or f"Resource: {name}"
            params = []
            
            # Extract URI parameters from resource name (e.g. {id} in "user://{id}")
            import re
            uri_params = re.findall(r'{([^}]+)}', name)
            
            for param in uri_params:
                params.append({
                    "name": param,
                    "type": "string",
                    "description": f"Parameter: {param}",
                    "required": True
                })
            
            resources.append({
                "name": name,
                "description": docstring.strip(),
                "parameters": params
            })
        
        return resources
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read method"""
        name = params.get("name")
        if not name:
            raise ValueError("Missing resource name")
        
        # Find the right handler by matching the resource URI pattern
        handler = None
        handler_args = {}
        
        for pattern, resource_handler in self.resources.items():
            # Convert the pattern to a regex
            import re
            regex_pattern = pattern.replace("{", "(?P<").replace("}", ">.*?)")
            
            match = re.fullmatch(regex_pattern, name)
            if match:
                handler = resource_handler
                handler_args = match.groupdict()
                break
        
        if not handler:
            raise ValueError(f"Resource '{name}' not found")
        
        # Call the handler with the extracted arguments
        try:
            result = await handler(**handler_args) if inspect.iscoroutinefunction(handler) else handler(**handler_args)
            
            # Determine content type (simple heuristic)
            content_type = "text/plain"
            if isinstance(result, dict) or result.strip().startswith(("{", "[")):
                try:
                    # Try to parse as JSON to validate
                    if isinstance(result, dict):
                        result = json.dumps(result)
                    else:
                        json.loads(result)
                    content_type = "application/json"
                except:
                    pass
            
            return {
                "content": result,
                "mime_type": content_type
            }
        except Exception as e:
            logger.exception(f"Error reading resource '{name}'")
            raise ValueError(f"Error reading resource: {str(e)}")
    
    async def _handle_list_tools(self) -> List[Dict[str, Any]]:
        """Handle tools/list method"""
        tools = []
        for name, handler in self.tools.items():
            # Get function signature info
            sig = inspect.signature(handler)
            docstring = handler.__doc__ or f"Tool: {name}"
            
            # Get type hints
            type_hints = get_type_hints(handler)
            
            # Build parameters list
            parameters = []
            for param_name, param in sig.parameters.items():
                # Skip 'self' and internal params
                if param_name == 'self' or param_name.startswith('_'):
                    continue
                
                param_type = "string"
                try:
                    if param_name in type_hints:
                        hint = type_hints[param_name]
                        # Map Python types to JSON Schema types
                        if hint == int:
                            param_type = "integer"
                        elif hint == float:
                            param_type = "number"
                        elif hint == bool:
                            param_type = "boolean"
                        elif hint == list or str(hint).startswith("typing.List"):
                            param_type = "array"
                        elif hint == dict or str(hint).startswith("typing.Dict"):
                            param_type = "object"
                except Exception:
                    # If type detection fails, default to string
                    pass
                
                parameters.append({
                    "name": param_name,
                    "type": param_type,
                    "description": f"Parameter: {param_name}",
                    "required": param.default == inspect.Parameter.empty
                })
            
            tools.append({
                "name": name,
                "description": docstring.strip(),
                "parameters": parameters
            })
        
        return tools
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call method"""
        name = params.get("name")
        if not name:
            raise ValueError("Missing tool name")
        
        arguments = params.get("arguments", {})
        
        handler = self.tools.get(name)
        if not handler:
            raise ValueError(f"Tool '{name}' not found")
        
        try:
            result = await handler(**arguments) if inspect.iscoroutinefunction(handler) else handler(**arguments)
            
            # For JSON compatibility
            if isinstance(result, (dict, list)):
                return result
            else:
                return {"result": str(result)}
        except Exception as e:
            logger.exception(f"Error calling tool '{name}'")
            raise ValueError(f"Error calling tool: {str(e)}")
    
    async def _handle_list_prompts(self) -> List[Dict[str, Any]]:
        """Handle prompts/list method"""
        prompts = []
        for name, handler in self.prompts.items():
            # Get function signature info
            sig = inspect.signature(handler)
            docstring = handler.__doc__ or f"Prompt: {name}"
            
            # Build parameters list
            parameters = []
            for param_name, param in sig.parameters.items():
                # Skip 'self' and internal params
                if param_name == 'self' or param_name.startswith('_'):
                    continue
                
                parameters.append({
                    "name": param_name,
                    "description": f"Parameter: {param_name}",
                    "required": param.default == inspect.Parameter.empty
                })
            
            prompts.append({
                "name": name,
                "description": docstring.strip(),
                "parameters": parameters
            })
        
        return prompts
    
    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get method"""
        name = params.get("name")
        if not name:
            raise ValueError("Missing prompt name")
        
        arguments = params.get("arguments", {})
        
        handler = self.prompts.get(name)
        if not handler:
            raise ValueError(f"Prompt '{name}' not found")
        
        try:
            result = await handler(**arguments) if inspect.iscoroutinefunction(handler) else handler(**arguments)
            
            # Check if the result is a list of messages or a single string
            if isinstance(result, str):
                # Convert to a single user message
                messages = [{
                    "role": "user", 
                    "content": {
                        "type": "text",
                        "text": result
                    }
                }]
            elif isinstance(result, list):
                # Assume it's a list of Message objects with role and content
                messages = []
                for msg in result:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        messages.append({
                            "role": msg.role,
                            "content": {
                                "type": "text",
                                "text": msg.content
                            }
                        })
            else:
                raise ValueError("Prompt handler must return a string or list of Message objects")
            
            return {
                "description": handler.__doc__ or f"Prompt: {name}",
                "messages": messages
            }
        except Exception as e:
            logger.exception(f"Error getting prompt '{name}'")
            raise ValueError(f"Error getting prompt: {str(e)}")
    
    async def _handle_sampling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sampling/createMessage method"""
        # This is a placeholder - in a real app, this would call an LLM
        return {
            "model": "placeholder-model",
            "stopReason": "endTurn",
            "role": "assistant",
            "content": {
                "type": "text",
                "text": f"This is a placeholder response from the {self.name} MCP server. To implement LLM sampling, use a custom sampling handler."
            }
        }
    
    def resource(self, uri_pattern: str):
        """Decorator to register a resource handler
        
        Args:
            uri_pattern: The URI pattern for the resource, like "user://{id}"
        """
        def decorator(func):
            self.resources[uri_pattern] = func
            return func
        return decorator
    
    def tool(self, name: Optional[str] = None):
        """Decorator to register a tool handler
        
        Args:
            name: Optional name for the tool. If not provided, uses the function name.
        """
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = func
            return func
        return decorator
    
    def prompt(self, name: Optional[str] = None):
        """Decorator to register a prompt handler
        
        Args:
            name: Optional name for the prompt. If not provided, uses the function name.
        """
        def decorator(func):
            prompt_name = name or func.__name__
            self.prompts[prompt_name] = func
            return func
        return decorator
    
    def run(self, host: str = "0.0.0.0", port: int = 8888):
        """Run the MCP server
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        logger.info(f"Starting {self.name} MCP server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Message class for prompt templates
class Message:
    """Message class for prompt templates"""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class UserMessage(Message):
    """User message for prompt templates"""
    
    def __init__(self, content: str):
        super().__init__("user", content)

class AssistantMessage(Message):
    """Assistant message for prompt templates"""
    
    def __init__(self, content: str):
        super().__init__("assistant", content)

class SystemMessage(Message):
    """System message for prompt templates"""
    
    def __init__(self, content: str):
        super().__init__("system", content)