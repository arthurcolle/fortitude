import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .sampling import SamplingRequest, SamplingResponse, Message

try:
    from ...core.visualization import AgentInteraction, AgentAction, AgentTrainingManager
except ImportError:
    # When running from a local directory, we need to handle imports differently
    core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'core')
    if core_path not in sys.path:
        import sys
        sys.path.append(core_path)
    from visualization import AgentInteraction, AgentAction, AgentTrainingManager

logger = logging.getLogger(__name__)

class MCPServer:
    """A server that implements the Model Context Protocol
    
    This server can be used to handle MCP requests from clients and process them
    using custom handlers. It provides a standardized interface for MCP operations.
    """
    
    def __init__(self, name: str = "Fortitude MCP Server", port: int = 8888, collect_training_data: bool = False):
        """Initialize the MCP server
        
        Args:
            name: Name of the MCP server
            port: Port to run the server on
            collect_training_data: Whether to collect agent training data
        """
        self.name = name
        self.port = port
        self.app = FastAPI(title=name)
        self.methods: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.collect_training_data = collect_training_data
        
        # Initialize training data collection if enabled
        if self.collect_training_data:
            self.training_manager = AgentTrainingManager()
            self.current_interactions: Dict[str, AgentInteraction] = {}
        
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
            
            # Record the interaction for training data if enabled
            if self.collect_training_data:
                request_id = str(uuid.uuid4())
                # Extract prompt from messages
                human_input = ""
                for msg in request.messages:
                    if msg.role == "user" and msg.content.type == "text":
                        human_input = msg.content.text
                        break
                
                # Create and store an interaction
                interaction = await self.training_manager.record_interaction(
                    human_input=human_input,
                    context_info={
                        "request_id": request_id,
                        "system_prompt": request.system_prompt,
                        "model": request.model,
                        "parameters": {
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature
                        }
                    }
                )
                self.current_interactions[request_id] = interaction
            
            # In a real implementation, this would call an LLM
            # Here we just return a placeholder message
            response_text = "This is a placeholder response from the MCP server. Please override the sampling handler with a proper implementation."
            
            # Complete the interaction for training data if enabled
            if self.collect_training_data and request_id in self.current_interactions:
                interaction = self.current_interactions[request_id]
                await self.training_manager.complete_interaction(
                    interaction=interaction,
                    final_response=response_text,
                    status="completed"
                )
                
                # Create a corpus if it doesn't exist
                corpus_id = "default"
                if corpus_id not in self.training_manager.corpora:
                    await self.training_manager.create_corpus(
                        name="Default Corpus",
                        description="Default corpus for agent training data",
                        tags=["default"]
                    )
                    corpus_id = list(self.training_manager.corpora.keys())[0]
                
                # Add interaction to corpus
                await self.training_manager.add_to_corpus(corpus_id, interaction)
                
                # Clean up
                del self.current_interactions[request_id]
            
            return {
                "model": "placeholder-model",
                "stopReason": "endTurn",
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": response_text
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
            
            # Record interaction for training data if enabled
            request_id = None
            if self.collect_training_data:
                request_id = str(uuid.uuid4())
                # Extract prompt from messages
                human_input = ""
                for msg in request.messages:
                    if msg.role == "user" and msg.content.type == "text":
                        human_input = msg.content.text
                        break
                
                # Create and store an interaction
                interaction = await self.training_manager.record_interaction(
                    human_input=human_input,
                    context_info={
                        "request_id": request_id,
                        "system_prompt": request.system_prompt,
                        "model": request.model,
                        "parameters": {
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature
                        }
                    }
                )
                self.current_interactions[request_id] = interaction
            
            # Call the handler to get the response
            response = await handler(request)
            
            # Complete the interaction for training data if enabled
            if self.collect_training_data and request_id and request_id in self.current_interactions:
                interaction = self.current_interactions[request_id]
                response_text = ""
                if response.content and response.content.type == "text":
                    response_text = response.content.text
                
                # Record the response
                await self.training_manager.complete_interaction(
                    interaction=interaction,
                    final_response=response_text,
                    status="completed"
                )
                
                # Create a corpus if it doesn't exist
                corpus_id = "default"
                if corpus_id not in self.training_manager.corpora:
                    await self.training_manager.create_corpus(
                        name="Default Corpus",
                        description="Default corpus for agent training data",
                        tags=["default"]
                    )
                    corpus_id = list(self.training_manager.corpora.keys())[0]
                
                # Add interaction to corpus
                await self.training_manager.add_to_corpus(corpus_id, interaction)
                
                # Clean up
                del self.current_interactions[request_id]
            
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
        
    async def export_training_data(self, corpus_ids: List[str] = None, output_path: str = None, format: str = "jsonl") -> str:
        """Export collected agent training data
        
        Args:
            corpus_ids: List of corpus IDs to include (defaults to all)
            output_path: Path to save the data (if None, returns as string)
            format: Export format ("jsonl" or "json")
            
        Returns:
            Path to the exported file or the data as a string
        """
        if not self.collect_training_data:
            raise ValueError("Training data collection is not enabled")
            
        # Use all corpora if none specified
        if corpus_ids is None:
            corpus_ids = list(self.training_manager.corpora.keys())
            
        # Create dataset name based on corpora
        corpus_names = [self.training_manager.corpora[cid].name for cid in corpus_ids if cid in self.training_manager.corpora]
        dataset_name = f"Agent Dataset - {', '.join(corpus_names)}"
        
        # Generate dataset
        dataset = await self.training_manager.create_training_dataset(
            corpus_ids=corpus_ids,
            name=dataset_name,
            description=f"Training dataset generated from {len(corpus_ids)} corpora"
        )
        
        # Export data
        data = await self.training_manager.export_dataset(dataset.dataset_id, format=format)
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(data)
            return output_path
        
        return data
        
    def register_tool_handler(self, tool_id: str, handler: Callable[[Dict[str, Any]], Awaitable[Any]]):
        """Register a tool handler that will capture tool usage for training
        
        Args:
            tool_id: ID of the tool
            handler: Function that handles tool calls
        """
        # Get tool config
        tool_config = self.tools.get(tool_id, {})
        if not tool_config:
            tool_config = {
                "name": tool_id,
                "description": f"Tool: {tool_id}",
                "parameters": {}
            }
            self.tools[tool_id] = tool_config
            
        async def tool_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            # Try to find an active interaction
            request_id = params.get("request_id", "")
            interaction = None
            if self.collect_training_data and request_id in self.current_interactions:
                interaction = self.current_interactions[request_id]
                
                # Record tool action
                action = await self.training_manager.record_action(
                    interaction=interaction,
                    action_type="tool_call",
                    description=tool_id,
                    parameters=params
                )
            
            # Call the actual handler
            try:
                result = await handler(params)
                
                # Record result if tracking
                if self.collect_training_data and interaction and action:
                    await self.training_manager.complete_action(
                        action=action,
                        result=result
                    )
                
                return {"result": result}
            except Exception as e:
                # Record error if tracking
                if self.collect_training_data and interaction and action:
                    await self.training_manager.complete_action(
                        action=action,
                        error=str(e)
                    )
                raise
        
        # Register the tool method
        self.register_method(f"tools/call/{tool_id}", tool_wrapper)
        logger.info(f"Registered tool handler: {tool_id}")
