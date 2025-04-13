#!/usr/bin/env python3

import os
import sys
import importlib.util
from typing import Dict, Any, Optional, Type

def _find_model_class(model_name: str):
    """Find a model class by name
    
    Args:
        model_name: Name of the model class
        
    Returns:
        The model class
    """
    # First check in the local project
    try:
        # Look in backend/models directory
        model_file = os.path.join(os.getcwd(), 'backend', 'models', f"{model_name.lower()}.py")
        model_module = f"backend.models.{model_name.lower()}"
        model_attr = model_name.capitalize()
        
        if os.path.exists(model_file):
            # Import the module
            spec = importlib.util.spec_from_file_location(model_module, model_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the model class
                if hasattr(module, model_attr):
                    return getattr(module, model_attr)
                else:
                    # Try other capitalizations
                    candidates = [attr for attr in dir(module) if attr.lower() == model_name.lower()]
                    if candidates:
                        return getattr(module, candidates[0])
    except Exception as e:
        print(f"Warning: Failed to import model from local project: {str(e)}")
    
    # Then check in the Fortitude example models
    try:
        from fortitude.backend.models.example import User, Product, Order
        
        if model_name.lower() == 'user':
            return User
        elif model_name.lower() == 'product':
            return Product
        elif model_name.lower() == 'order':
            return Order
    except ImportError:
        print("Warning: Failed to import example models from Fortitude")
    
    # Finally, give up
    print(f"Error: Could not find model class '{model_name}'")
    print("Make sure the model is defined in backend/models/{model_name.lower()}.py")
    sys.exit(1)

def generate_mcp_model_client(model_name: str, endpoint_url: str):
    """Generate an MCP client for a model
    
    Args:
        model_name: Name of the model
        endpoint_url: MCP endpoint URL
    """
    try:
        # Import the scaffolds module
        from fortitude.backend.mcp.scaffolds import generate_mcp_model_client as scaffold_client
    except ImportError:
        try:
            sys.path.insert(0, os.getcwd())
            from fortitude.backend.mcp.scaffolds import generate_mcp_model_client as scaffold_client
        except ImportError:
            print("Error: Could not import MCP scaffolds. Make sure fortitude is installed.")
            sys.exit(1)
    
    # Find the model class
    model_class = _find_model_class(model_name)
    
    # Generate the client
    output_path = os.path.join(os.getcwd(), f"{model_name.lower()}_mcp_client.py")
    
    if os.path.exists(output_path):
        print(f"Error: File {output_path} already exists")
        sys.exit(1)
    
    # Generate the client
    try:
        client_path = scaffold_client(model_class, output_path, endpoint_url)
        print(f"Generated MCP client for {model_name}: {client_path}")
        print(f"Run it with: python {client_path}")
    except Exception as e:
        print(f"Error generating MCP client: {str(e)}")
        sys.exit(1)

def generate_mcp_model_server(model_name: str, port: int):
    """Generate an MCP server for a model
    
    Args:
        model_name: Name of the model
        port: Port to run the MCP server on
    """
    try:
        # Import the scaffolds module
        from fortitude.backend.mcp.scaffolds import generate_mcp_model_server as scaffold_server
    except ImportError:
        try:
            sys.path.insert(0, os.getcwd())
            from fortitude.backend.mcp.scaffolds import generate_mcp_model_server as scaffold_server
        except ImportError:
            print("Error: Could not import MCP scaffolds. Make sure fortitude is installed.")
            sys.exit(1)
    
    # Find the model class
    model_class = _find_model_class(model_name)
    
    # Generate the server
    output_path = os.path.join(os.getcwd(), f"{model_name.lower()}_mcp_server.py")
    
    if os.path.exists(output_path):
        print(f"Error: File {output_path} already exists")
        sys.exit(1)
    
    # Generate the server
    try:
        server_path = scaffold_server(model_class, output_path, port)
        print(f"Generated MCP server for {model_name}: {server_path}")
        print(f"Run it with: python {server_path}")
    except Exception as e:
        print(f"Error generating MCP server: {str(e)}")
        sys.exit(1)