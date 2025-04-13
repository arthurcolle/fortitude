#!/usr/bin/env python3

import os
import importlib.util
import sys
from typing import Dict, Any, Optional, Type, List, Union
from pydantic import BaseModel, create_model, Field

from .sampling import SamplingRequest, SamplingResponse

def generate_mcp_model_client(model_class: Type[BaseModel], output_path: str, endpoint_url: str = "http://localhost:8888/mcp"):
    """Generate an MCP client for interacting with a model
    
    This creates a simple client that can perform CRUD operations on a model
    using an MCP server. The client uses the model's schema to validate data.
    
    Args:
        model_class: The Pydantic model class
        output_path: Path to write the client file
        endpoint_url: URL of the MCP endpoint
    """
    model_name = model_class.__name__
    schema = model_class.model_json_schema()
    
    # Generate client code
    code = f"""#!/usr/bin/env python3

import asyncio
import sys
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fortitude.backend.mcp import MCPClient

# Generated model class based on {model_name}
class {model_name}(BaseModel):
    # Schema from original model
{_format_schema_as_model(schema)}

class {model_name}Client:
    """Client for interacting with {model_name} through MCP"""
    
    def __init__(self, endpoint_url: str = "{endpoint_url}"):
        """Initialize the client
        
        Args:
            endpoint_url: URL of the MCP endpoint
        """
        self.mcp_client = MCPClient(endpoint_url=endpoint_url)
        self.model_name = "{model_name}"
    
    async def create(self, data: Dict[str, Any]) -> {model_name}:
        """Create a new {model_name}
        
        Args:
            data: The data for the new instance
            
        Returns:
            The created instance
        """
        # Validate the data using the model
        instance = {model_name}(**data)
        
        # Sample from LLM to process the creation
        prompt = f"Create a new {model_name} with the following data:\n{{json.dumps(data, indent=2)}}"
        system_prompt = f"You are a data processor for {model_name} instances. Return only valid JSON."
        
        try:
            response = await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                return {model_name}(**result)
            except json.JSONDecodeError:
                # If the response is not valid JSON, just return the original instance
                return instance
        except Exception as e:
            print(f"Error during MCP sampling: {{str(e)}}")
            return instance
    
    async def get(self, id: str) -> Optional[{model_name}]:
        """Get a {model_name} by ID
        
        Args:
            id: The ID of the instance
            
        Returns:
            The instance, or None if not found
        """
        # In a real implementation, this would fetch from a database
        # Here we just simulate it with MCP
        prompt = f"Get {model_name} with ID {{id}}"
        system_prompt = f"You are a data retrieval system for {model_name} instances. Return only valid JSON."
        
        try:
            response = await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                return {model_name}(**result)
            except json.JSONDecodeError:
                return None
        except Exception as e:
            print(f"Error during MCP sampling: {{str(e)}}")
            return None
    
    async def list(self) -> List[{model_name}]:
        """List all {model_name} instances
        
        Returns:
            A list of instances
        """
        # In a real implementation, this would fetch from a database
        # Here we just simulate it with MCP
        prompt = f"List all {model_name} instances"
        system_prompt = f"You are a data retrieval system for {model_name} instances. Return only valid JSON array."
        
        try:
            response = await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                return [{model_name}(**item) for item in result]
            except json.JSONDecodeError:
                return []
        except Exception as e:
            print(f"Error during MCP sampling: {{str(e)}}")
            return []

async def main():
    """Example usage"""
    client = {model_name}Client()
    
    # Example: Create a new instance
    new_instance = await client.create({{
        # Add example data here
{_format_example_data(schema)}
    }})
    
    print(f"Created instance: {{new_instance.model_dump_json(indent=2)}}")
    
    # Example: List instances
    instances = await client.list()
    print(f"Found {{len(instances)}} instances")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    # Write the client code to the output file
    with open(output_path, "w") as f:
        f.write(code)
    
    # Make the file executable
    os.chmod(output_path, 0o755)
    
    return output_path

def generate_mcp_model_server(model_class: Type[BaseModel], output_path: str, port: int = 8888):
    """Generate an MCP server for a model
    
    This creates a simple MCP server that can handle CRUD operations on a model.
    The server uses the model's schema to validate data.
    
    Args:
        model_class: The Pydantic model class
        output_path: Path to write the server file
        port: Port to run the server on
    """
    model_name = model_class.__name__
    schema = model_class.model_json_schema()
    
    # Generate server code
    code = f"""#!/usr/bin/env python3

import asyncio
import sys
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fortitude.backend.mcp import MCPServer, SamplingRequest, SamplingResponse

# Generated model class based on {model_name}
class {model_name}(BaseModel):
    # Schema from original model
{_format_schema_as_model(schema)}

# In-memory storage for instances
instances: Dict[str, {model_name}] = {{}}

async def sampling_handler(request: SamplingRequest) -> SamplingResponse:
    """Custom sampling handler for {model_name}"""
    # Get the messages from the request
    messages = request.messages
    
    if not messages or len(messages) == 0:
        return SamplingResponse(
            model="{model_name}-server",
            stopReason="endTurn",
            role="assistant",
            content={{
                "type": "text",
                "text": "No messages provided."
            }}
        )
    
    # Get the last message
    last_message = messages[-1]
    if last_message.content.get("type") != "text":
        return SamplingResponse(
            model="{model_name}-server",
            stopReason="endTurn",
            role="assistant",
            content={{
                "type": "text",
                "text": "Only text messages are supported."
            }}
        )
    
    # Get the prompt text
    prompt_text = last_message.content.get("text", "")
    
    # Process the prompt
    if "create" in prompt_text.lower() and "{{" in prompt_text:
        # Extract JSON data from the prompt
        try:
            # Find JSON in the prompt
            json_start = prompt_text.find("{{")
            json_end = prompt_text.rfind("}}") + 1
            json_str = prompt_text[json_start:json_end]
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Create a new instance
            instance = {model_name}(**data)
            
            # Store the instance
            instances[instance.id] = instance
            
            return SamplingResponse(
                model="{model_name}-server",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": json.dumps(instance.model_dump(), indent=2)
                }}
            )
        except Exception as e:
            return SamplingResponse(
                model="{model_name}-server",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": f"Error creating instance: {{str(e)}}"
                }}
            )
    elif "get" in prompt_text.lower() and "id" in prompt_text.lower():
        # Extract ID from the prompt
        try:
            # Find ID in the prompt
            id_start = prompt_text.find("id") + 2
            id_end = prompt_text.find("\n", id_start) if "\n" in prompt_text[id_start:] else len(prompt_text)
            id_str = prompt_text[id_start:id_end].strip()
            
            # Clean up the ID
            id_str = id_str.strip("\"': ")
            
            # Get the instance
            instance = instances.get(id_str)
            
            if instance:
                return SamplingResponse(
                    model="{model_name}-server",
                    stopReason="endTurn",
                    role="assistant",
                    content={{
                        "type": "text",
                        "text": json.dumps(instance.model_dump(), indent=2)
                    }}
                )
            else:
                return SamplingResponse(
                    model="{model_name}-server",
                    stopReason="endTurn",
                    role="assistant",
                    content={{
                        "type": "text",
                        "text": f"Instance with ID {{id_str}} not found."
                    }}
                )
        except Exception as e:
            return SamplingResponse(
                model="{model_name}-server",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": f"Error getting instance: {{str(e)}}"
                }}
            )
    elif "list" in prompt_text.lower():
        # List all instances
        try:
            instance_list = [instance.model_dump() for instance in instances.values()]
            
            return SamplingResponse(
                model="{model_name}-server",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": json.dumps(instance_list, indent=2)
                }}
            )
        except Exception as e:
            return SamplingResponse(
                model="{model_name}-server",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": f"Error listing instances: {{str(e)}}"
                }}
            )
    else:
        return SamplingResponse(
            model="{model_name}-server",
            stopReason="endTurn",
            role="assistant",
            content={{
                "type": "text",
                "text": f"I'm a {model_name} server. You can use me to create, get, or list {model_name} instances."
            }}
        )

def main():
    """Run the MCP server"""
    # Create the MCP server
    server = MCPServer(name="{model_name} MCP Server", port={port})
    
    # Register the sampling handler
    server.register_sampling_handler(sampling_handler)
    
    # Register the model as a resource
    server.register_resource("{model_name}", {{
        "schema": {json.dumps(schema)}
    }})
    
    print(f"Starting {model_name} MCP Server on port {port}...")
    print(f"Use this server with the generated client or with an MCP-compatible client.")
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
"""
    
    # Write the server code to the output file
    with open(output_path, "w") as f:
        f.write(code)
    
    # Make the file executable
    os.chmod(output_path, 0o755)
    
    return output_path

def _format_schema_as_model(schema: Dict[str, Any]) -> str:
    """Format a JSON schema as a Pydantic model
    
    Args:
        schema: The JSON schema
        
    Returns:
        Formatted model fields
    """
    if "properties" not in schema:
        return "    pass"
    
    lines = []
    required = schema.get("required", [])
    
    for prop_name, prop_def in schema.get("properties", {}).items():
        prop_type = _get_python_type(prop_def)
        is_required = prop_name in required
        default = None
        
        if "default" in prop_def:
            default = prop_def["default"]
        
        if is_required and default is None:
            lines.append(f"    {prop_name}: {prop_type}")
        else:
            if isinstance(default, str):
                default = f'"{default}"'
            elif default is None:
                default = "None"
            lines.append(f"    {prop_name}: {prop_type} = {default}")
    
    return "\n".join(lines)

def _get_python_type(prop_def: Dict[str, Any]) -> str:
    """Convert JSON schema type to Python type
    
    Args:
        prop_def: Property definition from JSON schema
        
    Returns:
        Python type as string
    """
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "List",
        "object": "Dict[str, Any]"
    }
    
    if "type" not in prop_def:
        return "Any"
    
    prop_type = prop_def["type"]
    
    if prop_type == "array" and "items" in prop_def:
        items_type = _get_python_type(prop_def["items"])
        return f"List[{items_type}]"
    elif prop_type in type_map:
        return type_map[prop_type]
    else:
        return "Any"

def _format_example_data(schema: Dict[str, Any]) -> str:
    """Format example data for a schema
    
    Args:
        schema: The JSON schema
        
    Returns:
        Formatted example data
    """
    if "properties" not in schema:
        return "        # No properties defined"
    
    lines = []
    
    for prop_name, prop_def in schema.get("properties", {}).items():
        # Skip id field
        if prop_name == "id":
            continue
        
        example = _get_example_value(prop_def)
        lines.append(f"        \"{prop_name}\": {example},")
    
    return "\n".join(lines)

def _get_example_value(prop_def: Dict[str, Any]) -> str:
    """Get an example value for a property
    
    Args:
        prop_def: Property definition from JSON schema
        
    Returns:
        Example value as string
    """
    if "example" in prop_def:
        example = prop_def["example"]
        if isinstance(example, str):
            return f'"{example}"'
        return str(example)
    
    if "default" in prop_def:
        default = prop_def["default"]
        if isinstance(default, str):
            return f'"{default}"'
        return str(default)
    
    if "type" not in prop_def:
        return "None"
    
    prop_type = prop_def["type"]
    
    if prop_type == "string":
        return '"example"'
    elif prop_type == "integer":
        return "42"
    elif prop_type == "number":
        return "3.14"
    elif prop_type == "boolean":
        return "True"
    elif prop_type == "array":
        if "items" in prop_def:
            item_example = _get_example_value(prop_def["items"])
            return f"[{item_example}]"
        return "[]"
    elif prop_type == "object":
        return "{}"
    else:
        return "None"
