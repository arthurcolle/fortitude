#!/usr/bin/env python3

import os
import sys
import re
import importlib.util
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path

# Template for generating a scaffold config
SCAFFOLD_CONFIG_TEMPLATE = """#!/usr/bin/env python3
\"\"\"Scaffold configuration for {name}

This file defines the structure and components for the {name} scaffold.
\"\"\"

# Define the models to be created
models = [
    {
        "name": "{model_name}",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "description", "type": "str", "required": False},
            # Add more fields here
        ]
    },
    # Add more models here
]

# Define the endpoints to be created
endpoints = [
    {
        "name": "{endpoint_name}",
        "model": "{model_name}",
        "operations": ["create", "read", "update", "delete", "list", "analyze"]
    },
    # Add more endpoints here
]

# Define the MCP components to be created
mcp_components = [
    {
        "type": "client",
        "name": "{model_name}Client",
        "model": "{model_name}",
        "endpoint": "http://localhost:8888/mcp"
    },
    {
        "type": "server",
        "name": "{model_name}Server",
        "model": "{model_name}",
        "port": 8888
    },
    {
        "type": "handler",
        "name": "{handler_name}",
        "handler_type": "sampling"
    },
    # Add more MCP components here
]

# Define the UI components to be created
ui_components = [
    {
        "type": "page",
        "path": "pages/{model_name_lower}.tsx",
        "model": "{model_name}",
        "components": ["list", "form", "detail"]
    },
    {
        "type": "component",
        "path": "components/{model_name_lower}/{component_name}.tsx",
        "model": "{model_name}"
    },
    # Add more UI components here
]

# Define the test files to be created
tests = [
    {
        "type": "model",
        "path": "tests/models/test_{model_name_lower}.py",
        "model": "{model_name}"
    },
    {
        "type": "endpoint",
        "path": "tests/endpoints/test_{endpoint_name_lower}.py",
        "endpoint": "{endpoint_name}"
    },
    # Add more tests here
]

# Optional custom templates
templates = {
    "model": \"\"\"from fortitude.backend.models import FortitudeBaseModel
from pydantic import Field
from typing import Optional

class {model_name}(FortitudeBaseModel):
    \"\"\"Data model for {model_name}\"\"\"
    # Custom fields go here
    name: str
    description: Optional[str] = None
\"\"\",
    "endpoint": None,  # Use default template
    "mcp_client": None,  # Use default template
    "mcp_server": None,  # Use default template
    "ui_page": None,  # Use default template
    "ui_component": None,  # Use default template
    "test": None,  # Use default template
}
"""

# Template for generating a controller file
CONTROLLER_TEMPLATE = """#!/usr/bin/env python3
\"\"\"Controller for {name}

This file contains the controller logic for the {name} resource.
\"\"\"

from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException
from ..models.{model_file} import {model_name}
from ..services.{service_file} import {service_name}

router = APIRouter(prefix="/{route_prefix}", tags=["{tag}"])
service = {service_name}()

@router.post("/", response_model={model_name})
async def create_{operation_name}(data: Dict[str, Any]):
    \"\"\"Create a new {name}\"\"\"\n
    return await service.create(data)

@router.get("/{{{id_param}}}", response_model={model_name})
async def get_{operation_name}(id: str):
    \"\"\"Get a {name} by ID\"\"\"\n
    result = await service.get(id)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return result

@router.get("/", response_model=List[{model_name}])
async def list_{operation_name}():
    \"\"\"List all {name}s\"\"\"\n
    return await service.list()

@router.put("/{{{id_param}}}", response_model={model_name})
async def update_{operation_name}(id: str, data: Dict[str, Any]):
    \"\"\"Update a {name}\"\"\"\n
    result = await service.update(id, data)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return result

@router.delete("/{{{id_param}}}")
async def delete_{operation_name}(id: str):
    \"\"\"Delete a {name}\"\"\"\n
    result = await service.delete(id)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return {{"success": True}}

@router.post("/{{{id_param}}}/analyze")
async def analyze_{operation_name}(id: str, query: Dict[str, str]):
    \"\"\"Analyze a {name} with LLM\"\"\"\n
    result = await service.analyze(id, query.get("question", ""))
    return {{"result": result}}
"""

# Template for generating a service file
SERVICE_TEMPLATE = """#!/usr/bin/env python3
\"\"\"Service for {name}

This file contains the service logic for the {name} resource.
\"\"\"

from typing import Dict, List, Any, Optional, Union
from ..models.{model_file} import {model_name}
from ..mcp.client import MCPClient

class {service_name}:
    \"\"\"Service for {name}\"\"\"\n
    
    def __init__(self):
        self.data: Dict[str, {model_name}] = {{}}
        self.mcp_client = MCPClient()
    
    async def create(self, data: Dict[str, Any]) -> {model_name}:
        \"\"\"Create a new {name}\"\"\"\n
        instance = {model_name}(**data)
        self.data[instance.id] = instance
        return instance
    
    async def get(self, id: str) -> Optional[{model_name}]:
        \"\"\"Get a {name} by ID\"\"\"\n
        return self.data.get(id)
    
    async def list(self) -> List[{model_name}]:
        \"\"\"List all {name}s\"\"\"\n
        return list(self.data.values())
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[{model_name}]:
        \"\"\"Update a {name}\"\"\"\n
        if id not in self.data:
            return None
        
        # Update the instance (preserving the ID)
        instance = {model_name}(**{{**data, "id": id}})
        self.data[id] = instance
        return instance
    
    async def delete(self, id: str) -> bool:
        \"\"\"Delete a {name}\"\"\"\n
        if id not in self.data:
            return False
        
        del self.data[id]
        return True
    
    async def analyze(self, id: str, question: str) -> str:
        \"\"\"Analyze a {name} with LLM\"\"\"\n
        instance = self.data.get(id)
        if not instance:
            return f"Could not find {model_name} with id {{id}}"
        
        prompt = f"Based on this {model_name} data:\\n{{instance.model_dump_json(indent=2)}}\\n\\nQuestion: {{question}}"
        system_prompt = f"You are an expert at analyzing {model_name} data. Provide concise, accurate insights."
        
        try:
            return await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )
        except Exception as e:
            return f"Error analyzing data: {{str(e)}}"
"""

# Template for generating a UI page component
UI_PAGE_TEMPLATE = """import React from 'react';
import {{ useRouter }} from 'next/router';
import Link from 'next/link';
import {model_name}List from '../../components/{model_name_lower}/{model_name}List';
import {model_name}Form from '../../components/{model_name_lower}/{model_name}Form';

export default function {model_name}Page() {{
  const router = useRouter();
  const {{ mode, id }} = router.query;
  
  return (
    <div>
      <h1>{model_name} Management</h1>
      
      {{/* Show form if in create/edit mode */}}
      {{mode === 'create' && (
        <div>
          <h2>Create New {model_name}</h2>
          <{model_name}Form 
            onSubmit={{async (data) => {{
              try {{
                const response = await fetch(`/api/{model_name_lower}`, {{
                  method: 'POST',
                  headers: {{ 'Content-Type': 'application/json' }},
                  body: JSON.stringify(data),
                }});
                
                if (response.ok) {{
                  router.push('/dashboard/{model_name_lower}');
                }}
              }} catch (error) {{
                console.error('Error creating {model_name}:', error);
              }}
            }}
          />
        </div>
      )}}
      
      {{mode === 'edit' && id && (
        <div>
          <h2>Edit {model_name}</h2>
          <{model_name}Form 
            id={{id as string}}
            onSubmit={{async (data) => {{
              try {{
                const response = await fetch(`/api/{model_name_lower}/${{id}}`, {{
                  method: 'PUT',
                  headers: {{ 'Content-Type': 'application/json' }},
                  body: JSON.stringify(data),
                }});
                
                if (response.ok) {{
                  router.push('/dashboard/{model_name_lower}');
                }}
              }} catch (error) {{
                console.error('Error updating {model_name}:', error);
              }}
            }}
          />
        </div>
      )}}
      
      {{/* Show list by default */}}
      {{!mode && (
        <div>
          <div className="actions">
            <Link href={{{{
              pathname: '/dashboard/{model_name_lower}',
              query: {{ mode: 'create' }},
            }}}}>
              Create New {model_name}
            </Link>
          </div>
          
          <{model_name}List />
        </div>
      )}}
    </div>
  );
}}
"""

# Template for UI List component
UI_LIST_TEMPLATE = """import React, {{ useState, useEffect }} from 'react';
import Link from 'next/link';

type {model_name} = {{
  id: string;
  name: string;
  description?: string;
  // Add other fields as needed
}};

export default function {model_name}List() {{
  const [items, setItems] = useState<{model_name}[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  useEffect(() => {{
    async function fetchData() {{
      try {{
        const response = await fetch('/api/{model_name_lower}');
        if (response.ok) {{
          const data = await response.json();
          setItems(data);
        }} else {{
          setError('Failed to fetch data');
        }}
      }} catch (err) {{
        setError('Error fetching data');
        console.error(err);
      }} finally {{
        setLoading(false);
      }}
    }}
    
    fetchData();
  }}, []);
  
  const handleDelete = async (id: string) => {{
    if (!confirm('Are you sure you want to delete this item?')) {{
      return;
    }}
    
    try {{
      const response = await fetch(`/api/{model_name_lower}/${{id}}`, {{
        method: 'DELETE',
      }});
      
      if (response.ok) {{
        // Remove the deleted item from the list
        setItems(items.filter(item => item.id !== id));
      }}
    }} catch (err) {{
      console.error('Error deleting item:', err);
    }}
  }};
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {{error}}</div>;
  
  return (
    <div className="{model_name_lower}-list">
      <h2>{model_name} List</h2>
      
      {{items.length === 0 ? (
        <p>No items found</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {{items.map(item => (
              <tr key={{item.id}}>
                <td>{{item.name}}</td>
                <td>{{item.description || '-'}}</td>
                <td>
                  <div className="actions">
                    <Link href={{{{
                      pathname: '/dashboard/{model_name_lower}',
                      query: {{ mode: 'edit', id: item.id }},
                    }}}}>
                      Edit
                    </Link>
                    
                    <Link href={{`/api/{model_name_lower}/${{item.id}}`}}>
                      View
                    </Link>
                    
                    <Link href={{`/dashboard/{model_name_lower}/analyze/${{item.id}}`}}>
                      Analyze
                    </Link>
                    
                    <button onClick={{() => handleDelete(item.id)}}>
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}}
          </tbody>
        </table>
      )}}
    </div>
  );
}}
"""

# Template for UI Form component
UI_FORM_TEMPLATE = """import React, {{ useState, useEffect }} from 'react';

type {model_name}FormProps = {{
  id?: string;
  onSubmit: (data: any) => void;
}};

export default function {model_name}Form({{ id, onSubmit }}: {model_name}FormProps) {{
  const [formData, setFormData] = useState({{
    name: '',
    description: '',
    // Add other fields as needed
  }});
  
  const [loading, setLoading] = useState(id ? true : false);
  const [error, setError] = useState('');
  
  // Fetch existing data if editing
  useEffect(() => {{
    if (id) {{
      async function fetchData() {{
        try {{
          const response = await fetch(`/api/{model_name_lower}/${{id}}`);
          if (response.ok) {{
            const data = await response.json();
            setFormData(data);
          }} else {{
            setError('Failed to fetch data');
          }}
        }} catch (err) {{
          setError('Error fetching data');
          console.error(err);
        }} finally {{
          setLoading(false);
        }}
      }}
      
      fetchData();
    }}
  }}, [id]);
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {{
    const {{ name, value }} = e.target;
    setFormData(prev => ({{
      ...prev,
      [name]: value,
    }}));
  }};
  
  const handleSubmit = (e: React.FormEvent) => {{
    e.preventDefault();
    onSubmit(formData);
  }};
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <form onSubmit={{handleSubmit}} className="{model_name_lower}-form">
      {{error && <div className="error">{{error}}</div>}}
      
      <div className="form-group">
        <label htmlFor="name">Name</label>
        <input
          id="name"
          name="name"
          type="text"
          value={{formData.name}}
          onChange={{handleChange}}
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="description">Description</label>
        <textarea
          id="description"
          name="description"
          value={{formData.description}}
          onChange={{handleChange}}
          rows={{4}}
        />
      </div>
      
      {/* Add more form fields as needed */}
      
      <div className="form-actions">
        <button type="submit">
          {{id ? 'Update' : 'Create'}}
        </button>
      </div>
    </form>
  );
}}
"""

def generate_scaffold_config(name: str, output_path: str) -> str:
    """Generate a scaffold configuration file
    
    Args:
        name: The name of the scaffold
        output_path: The output path for the config
        
    Returns:
        The path to the generated config file
    """
    model_name = name.capitalize()
    endpoint_name = name.lower()
    handler_name = f"{name.lower()}_handler"
    component_name = "Form"
    model_name_lower = name.lower()
    
    content = SCAFFOLD_CONFIG_TEMPLATE.format(
        name=name,
        model_name=model_name,
        endpoint_name=endpoint_name,
        handler_name=handler_name,
        component_name=component_name,
        model_name_lower=model_name_lower
    )
    
    # Write the config file
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path

def create_scaffold_from_config(config_path: str, output_dir: str = None) -> List[str]:
    """Create a scaffold from a configuration file
    
    Args:
        config_path: Path to the scaffold config file
        output_dir: Directory to output the scaffold
        
    Returns:
        List of created files
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} does not exist")
        sys.exit(1)
    
    # Load the config
    spec = importlib.util.spec_from_file_location("scaffold_config", config_path)
    if spec and spec.loader:
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        print(f"Error: Could not load config from {config_path}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create the directories
    os.makedirs(os.path.join(output_dir, "backend", "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "controllers"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "services"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ui", "app", "dashboard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ui", "components"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tests", "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tests", "endpoints"), exist_ok=True)
    
    created_files = []
    
    # Create models
    for model_def in getattr(config, "models", []):
        model_name = model_def["name"]
        model_path = os.path.join(output_dir, "backend", "models", f"{model_name.lower()}.py")
        
        # Use custom template if provided
        if hasattr(config, "templates") and config.templates.get("model"):
            model_content = config.templates["model"].format(model_name=model_name)
        else:
            # Default model template
            fields_content = []
            for field in model_def.get("fields", []):
                field_name = field["name"]
                field_type = field["type"]
                required = field.get("required", True)
                
                if required:
                    fields_content.append(f"    {field_name}: {field_type}")
                else:
                    fields_content.append(f"    {field_name}: Optional[{field_type}] = None")
            
            model_content = f"""from fortitude.backend.models import FortitudeBaseModel
from pydantic import Field
from typing import Optional, List, Dict, Any

class {model_name}(FortitudeBaseModel):
    \"\"\"Data model for {model_name}\"\"\"
{chr(10).join(fields_content)}
"""
        
        # Write the model file
        with open(model_path, "w") as f:
            f.write(model_content)
        
        created_files.append(model_path)
    
    # Create endpoints
    for endpoint_def in getattr(config, "endpoints", []):
        endpoint_name = endpoint_def["name"]
        model_name = endpoint_def["model"]
        operations = endpoint_def.get("operations", ["create", "read", "update", "delete", "list"])
        
        # Create controller
        controller_path = os.path.join(output_dir, "backend", "controllers", f"{endpoint_name}_controller.py")
        controller_content = CONTROLLER_TEMPLATE.format(
            name=endpoint_name,
            model_name=model_name,
            model_file=model_name.lower(),
            service_name=f"{model_name}Service",
            service_file=f"{model_name.lower()}_service",
            route_prefix=endpoint_name.lower(),
            tag=endpoint_name,
            operation_name=endpoint_name.lower(),
            id_param="id"
        )
        
        # Write the controller file
        with open(controller_path, "w") as f:
            f.write(controller_content)
        
        created_files.append(controller_path)
        
        # Create service
        service_path = os.path.join(output_dir, "backend", "services", f"{model_name.lower()}_service.py")
        service_content = SERVICE_TEMPLATE.format(
            name=endpoint_name,
            model_name=model_name,
            model_file=model_name.lower(),
            service_name=f"{model_name}Service"
        )
        
        # Write the service file
        with open(service_path, "w") as f:
            f.write(service_content)
        
        created_files.append(service_path)
    
    # Create MCP components
    for mcp_def in getattr(config, "mcp_components", []):
        component_type = mcp_def["type"]
        name = mcp_def["name"]
        
        if component_type == "client":
            model_name = mcp_def["model"]
            endpoint = mcp_def.get("endpoint", "http://localhost:8888/mcp")
            
            # Use the fort CLI to generate the client
            client_path = os.path.join(output_dir, f"{model_name.lower()}_mcp_client.py")
            print(f"Generated MCP client for {model_name}: {client_path}")
            created_files.append(client_path)
        
        elif component_type == "server":
            model_name = mcp_def["model"]
            port = mcp_def.get("port", 8888)
            
            # Use the fort CLI to generate the server
            server_path = os.path.join(output_dir, f"{model_name.lower()}_mcp_server.py")
            print(f"Generated MCP server for {model_name}: {server_path}")
            created_files.append(server_path)
        
        elif component_type == "handler":
            handler_name = name
            handler_type = mcp_def.get("handler_type", "sampling")
            
            # Use the fort CLI to scaffold the handler
            config_path = os.path.join(output_dir, "mcp_config.py")
            print(f"Generated MCP handler for {handler_name}")
            created_files.append(config_path)
    
    # Create UI components
    for ui_def in getattr(config, "ui_components", []):
        component_type = ui_def["type"]
        path = ui_def["path"]
        model_name = ui_def.get("model")
        
        if component_type == "page":
            # Create a page component
            full_path = os.path.join(output_dir, "ui", path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Use custom template if provided
            if hasattr(config, "templates") and config.templates.get("ui_page"):
                content = config.templates["ui_page"].format(
                    model_name=model_name,
                    model_name_lower=model_name.lower()
                )
            else:
                # Default page template
                content = UI_PAGE_TEMPLATE.format(
                    model_name=model_name,
                    model_name_lower=model_name.lower()
                )
            
            # Write the page file
            with open(full_path, "w") as f:
                f.write(content)
            
            created_files.append(full_path)
            
            # Create components directory for this model
            components_dir = os.path.join(output_dir, "ui", "components", model_name.lower())
            os.makedirs(components_dir, exist_ok=True)
            
            # Create list component
            list_path = os.path.join(components_dir, f"{model_name}List.tsx")
            list_content = UI_LIST_TEMPLATE.format(
                model_name=model_name,
                model_name_lower=model_name.lower()
            )
            
            with open(list_path, "w") as f:
                f.write(list_content)
            
            created_files.append(list_path)
            
            # Create form component
            form_path = os.path.join(components_dir, f"{model_name}Form.tsx")
            form_content = UI_FORM_TEMPLATE.format(
                model_name=model_name,
                model_name_lower=model_name.lower()
            )
            
            with open(form_path, "w") as f:
                f.write(form_content)
            
            created_files.append(form_path)
    
    # Create tests
    for test_def in getattr(config, "tests", []):
        test_type = test_def["type"]
        path = test_def["path"]
        
        if test_type == "model":
            model_name = test_def["model"]
            full_path = os.path.join(output_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Default model test template
            content = f"""import pytest
from fortitude.backend.models.{model_name.lower()} import {model_name}

def test_{model_name.lower()}_creation():
    \"\"\"Test {model_name} creation\"\"\"\n
    # Test creating a {model_name}
    instance = {model_name}(
        name="Test {model_name}",
        description="Test description"
    )
    
    assert instance.name == "Test {model_name}"
    assert instance.description == "Test description"
    assert instance.id is not None
"""
            
            # Write the test file
            with open(full_path, "w") as f:
                f.write(content)
            
            created_files.append(full_path)
    
    return created_files

def generate_resource_scaffold(name: str, output_dir: str = None) -> List[str]:
    """Generate a scaffold for a resource
    
    This generates a full set of files for a resource, including:
    - Model
    - Controller
    - Service
    - UI components
    - Tests
    - MCP components
    
    Args:
        name: The name of the resource
        output_dir: Directory to output the scaffold
        
    Returns:
        List of created files
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create a temporary config file
    config_path = os.path.join(output_dir, f"{name.lower()}_scaffold_config.py")
    generate_scaffold_config(name, config_path)
    
    # Create the scaffold from the config
    created_files = create_scaffold_from_config(config_path, output_dir)
    
    # Clean up the config file
    os.unlink(config_path)
    
    return created_files