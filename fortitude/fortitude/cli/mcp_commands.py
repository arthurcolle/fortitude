#!/usr/bin/env python3

import os
import sys
import importlib.util
import asyncio
from typing import Optional

def start_mcp_server(name: str, port: int):
    """Start an MCP server
    
    Args:
        name: Name of the MCP server
        port: Port to run the MCP server on
    """
    try:
        # First try to import from the local project
        sys.path.insert(0, os.getcwd())
        from fortitude.backend.mcp import MCPServer
    except ImportError:
        try:
            # Then try to import from the installed package
            from fortitude.backend.mcp import MCPServer
        except ImportError:
            print("Error: Could not import MCPServer. Make sure fortitude is installed.")
            sys.exit(1)
            
    print(f"Starting MCP server '{name}' on port {port}...")
    
    # Check if there's a config file for the MCP server
    config_path = os.path.join(os.getcwd(), "mcp_config.py")
    custom_handler = None
    
    if os.path.exists(config_path):
        print(f"Found MCP config at {config_path}")
        spec = importlib.util.spec_from_file_location("mcp_config", config_path)
        if spec and spec.loader:
            mcp_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mcp_config)
            
            # Check if there's a custom sampling handler
            if hasattr(mcp_config, "sampling_handler"):
                custom_handler = mcp_config.sampling_handler
                print("Using custom sampling handler from mcp_config.py")
            
    # Create and run the MCP server
    server = MCPServer(name=name, port=port)
    
    if custom_handler:
        server.register_sampling_handler(custom_handler)
    
    # Check for custom tools in the config
    if os.path.exists(config_path) and hasattr(mcp_config, "tools"):
        for tool_id, tool_config in mcp_config.tools.items():
            server.register_tool(tool_id, tool_config)
            print(f"Registered tool: {tool_id}")
    
    # Check for custom resources in the config
    if os.path.exists(config_path) and hasattr(mcp_config, "resources"):
        for resource_id, resource in mcp_config.resources.items():
            server.register_resource(resource_id, resource)
            print(f"Registered resource: {resource_id}")
    
    # Run the server
    server.run()

def scaffold_mcp_client(name: str, endpoint: str):
    """Scaffold an MCP client
    
    Args:
        name: Name of the MCP client
        endpoint: MCP endpoint URL
    """
    output_path = os.path.join(os.getcwd(), f"{name.lower()}_client.py")
    
    if os.path.exists(output_path):
        print(f"Error: File {output_path} already exists")
        sys.exit(1)
        
    with open(output_path, "w") as f:
        f.write(f"""#!/usr/bin/env python3

import asyncio
import sys
from fortitude.backend.mcp import MCPClient, SamplingRequest, Message

async def main():
    # Create an MCP client
    client = MCPClient(endpoint_url="{endpoint}")
    
    # Example: Sample from LLM
    prompt = "Hello, I am {name} client. What can you tell me about Fortitude?"
    
    print(f"Sending prompt to MCP endpoint: {{prompt}}")
    
    try:
        response = await client.sample_text(
            prompt=prompt,
            system_prompt="You are a helpful assistant that explains Fortitude clearly and concisely.",
            max_tokens=1000
        )
        
        print("\\nResponse from LLM:")
        print(response)
    except Exception as e:
        print(f"Error sampling from LLM: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
""")
    
    # Make the file executable
    os.chmod(output_path, 0o755)
    
    print(f"Created MCP client: {output_path}")
    print(f"Run it with: python {output_path}")

def scaffold_mcp_handler(name: str, handler_type: str):
    """Scaffold an MCP handler
    
    Args:
        name: Name of the MCP handler
        handler_type: Type of handler to scaffold (sampling, resource, tool)
    """
    config_path = os.path.join(os.getcwd(), "mcp_config.py")
    
    # Create the config file if it doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""#!/usr/bin/env python3

\"\"\"MCP configuration for Fortitude

This file contains configuration for the MCP server, including custom handlers,
tools, and resources.
\"\"\"

from fortitude.backend.mcp import SamplingRequest, SamplingResponse

# Dictionary of tools to register with the MCP server
tools = {}

# Dictionary of resources to register with the MCP server
resources = {}

# Custom sampling handler (optional)
# async def sampling_handler(request: SamplingRequest) -> SamplingResponse:
#     # Custom logic for handling sampling requests
#     pass
""")
    
    # Add the handler to the config file
    with open(config_path, "r") as f:
        content = f.read()
    
    if handler_type == "sampling":
        # Add a sampling handler
        if "async def sampling_handler" in content and not content.startswith("#"):
            print(f"Warning: A sampling handler already exists in {config_path}")
            print("Uncomment and modify the existing handler instead.")
            sys.exit(1)
            
        with open(config_path, "a") as f:
            f.write(f"""

async def sampling_handler(request: SamplingRequest) -> SamplingResponse:
    \"\"\"Custom sampling handler for {name}\"\"\"
    # Get the messages from the request
    messages = request.messages
    
    # For demonstration, we'll just echo back the last message
    if messages and len(messages) > 0:
        last_message = messages[-1]
        if last_message.content.get("type") == "text":
            prompt_text = last_message.content.get("text", "")
            
            # Process the prompt (in a real implementation, this would call an LLM)
            response_text = f"[{name}] You said: {{prompt_text}}"
            
            # Return a sampling response
            return SamplingResponse(
                model="{name}-model",
                stopReason="endTurn",
                role="assistant",
                content={{
                    "type": "text",
                    "text": response_text
                }}
            )
    
    # Fallback response
    return SamplingResponse(
        model="{name}-model",
        stopReason="endTurn",
        role="assistant",
        content={{
            "type": "text",
            "text": "I couldn't process that request."
        }}
    )
""")
            
    elif handler_type == "tool":
        # Add a tool definition
        with open(config_path, "a") as f:
            f.write(f"""

# Tool definition for {name}
tools["{name}"] = {{
    "name": "{name}",
    "description": "A tool for {name}",
    "parameters": {{
        "type": "object",
        "properties": {{
            "input": {{
                "type": "string",
                "description": "The input to process"
            }}
        }},
        "required": ["input"]
    }},
    "handler": lambda params: {{
        "result": f"Processed input: {{params.get('input', '')}}"
    }}
}}
""")
            
    elif handler_type == "resource":
        # Add a resource definition
        with open(config_path, "a") as f:
            f.write(f"""

# Resource definition for {name}
class {name.capitalize()}Resource:
    \"\"\"A resource for {name}\"\"\"
    
    def __init__(self):
        self.data = {{}}
    
    async def get(self, id: str):
        \"\"\"Get a resource by ID\"\"\"
        return self.data.get(id)
    
    async def set(self, id: str, value):
        \"\"\"Set a resource value\"\"\"
        self.data[id] = value
        return {{
            "success": True,
            "id": id
        }}

resources["{name}"] = {name.capitalize()}Resource()
""")
    
    print(f"Added {handler_type} handler '{name}' to {config_path}")
    print(f"Start the MCP server with: fort mcp-server")