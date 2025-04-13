
from typing import List, Optional, Any, Dict, Union, Tuple, Set, Callable
from pydantic import BaseModel, Field
from IPython.display import display, HTML
import uuid
import html
import json
import os
import re
from pathlib import Path
import asyncio
import datetime

_rendered_css = False  # global state

class PrettyRenderableModel(BaseModel):
    def _repr_html_(self):
        global _rendered_css
        if not _rendered_css:
            display(HTML("""
            <style>
                /* Modern Card Design */
                .pretty-card {
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.98);
                    padding: 1.5em;
                    margin: 1.2em 0;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro', Roboto, sans-serif;
                    border: 1px solid rgba(150,150,150,0.15);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                
                .pretty-card:hover {
                    box-shadow: 0 12px 28px rgba(0,0,0,0.15);
                    transform: translateY(-2px);
                }
                
                /* Card Header */
                .pretty-header {
                    font-size: 1.4em;
                    font-weight: 600;
                    color: #1a3c6e;
                    margin-bottom: 0.8em;
                    padding-bottom: 0.6em;
                    border-bottom: 2px solid rgba(0,51,102,0.1);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                
                .pretty-header-badge {
                    font-size: 0.65em;
                    background: #e8f0fe;
                    color: #1967d2;
                    padding: 0.3em 0.6em;
                    border-radius: 20px;
                    font-weight: 500;
                }
                
                /* Key-Value Pairs */
                .pretty-kv {
                    margin: 0.5em 0;
                    line-height: 1.5;
                    display: flex;
                    flex-wrap: wrap;
                }
                
                .pretty-key {
                    font-weight: 500;
                    color: #333;
                    margin-right: 0.5em;
                    min-width: 120px;
                }
                
                .pretty-value {
                    flex: 1;
                    color: #444;
                }
                
                /* Different Value Types */
                .pretty-string {
                    color: #0971A6;
                }
                
                .pretty-number {
                    color: #1967d2;
                    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
                }
                
                .pretty-boolean-true {
                    color: #188038;
                    font-weight: 500;
                }
                
                .pretty-boolean-false {
                    color: #d93025;
                    font-weight: 500;
                }
                
                .pretty-null {
                    color: #888;
                    font-style: italic;
                }
                
                /* Nested Elements */
                .pretty-nested {
                    margin-left: 1.5em;
                    border-left: 3px solid #e8eaed;
                    padding-left: 1em;
                    margin-top: 0.5em;
                }
                
                /* Lists */
                ul.pretty-list {
                    padding-left: 1.2em;
                    margin: 0.3em 0;
                    list-style-type: none;
                }
                
                ul.pretty-list li {
                    position: relative;
                    margin: 0.3em 0;
                }
                
                ul.pretty-list li:before {
                    content: "•";
                    color: #5f6368;
                    position: absolute;
                    left: -1em;
                    top: 0.1em;
                }
                
                /* Collapsible sections */
                .pretty-collapsible {
                    cursor: pointer;
                }
                
                .pretty-collapsible:after {
                    content: "▼";
                    font-size: 0.8em;
                    margin-left: 0.5em;
                    color: #5f6368;
                }
                
                .pretty-collapsed:after {
                    content: "►";
                }
                
                .pretty-collapsed + .pretty-content {
                    display: none;
                }
                
                /* Copy button */
                .pretty-copy-btn {
                    position: absolute;
                    top: 1em;
                    right: 1em;
                    padding: 0.3em 0.6em;
                    background: rgba(0,0,0,0.05);
                    border: none;
                    border-radius: 4px;
                    font-size: 0.8em;
                    color: #5f6368;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .pretty-copy-btn:hover {
                    background: rgba(0,0,0,0.1);
                }
                
                /* Type indicators */
                .pretty-type-indicator {
                    font-size: 0.7em;
                    color: #777;
                    margin-left: 0.5em;
                    vertical-align: super;
                }
                
                /* Special field styling */
                .pretty-field-important {
                    background: #fef7e0;
                    border-radius: 4px;
                    padding: 0.2em 0.5em;
                }
                
                /* Confidence meter */
                .confidence-meter {
                    height: 6px;
                    width: 100px;
                    background: #f1f3f4;
                    border-radius: 3px;
                    display: inline-block;
                    margin-left: 8px;
                    vertical-align: middle;
                    overflow: hidden;
                }
                
                .confidence-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #34a853 0%, #4285f4 100%);
                }
            </style>
            <script>
                // Add collapsible functionality
                function toggleCollapse(id) {
                    const header = document.getElementById(id);
                    header.classList.toggle('pretty-collapsed');
                }
                
                // Add copy functionality
                function copyToClipboard(id) {
                    const card = document.getElementById(id);
                    const textToCopy = JSON.stringify(JSON.parse(card.getAttribute('data-json')), null, 2);
                    
                    navigator.clipboard.writeText(textToCopy).then(() => {
                        const btn = card.querySelector('.pretty-copy-btn');
                        const originalText = btn.textContent;
                        btn.textContent = 'Copied!';
                        setTimeout(() => {
                            btn.textContent = originalText;
                        }, 2000);
                    });
                }
            </script>
            """))
            _rendered_css = True

        title = self.__class__.__name__
        html_id = f"pretty_{uuid.uuid4().hex}"
        json_data = json.dumps(self.model_dump())
        
        return f"""
        <div id='{html_id}' class='pretty-card' data-json='{html.escape(json_data)}'>
            <button class='pretty-copy-btn' onclick="copyToClipboard('{html_id}')">Copy JSON</button>
            {self._render_fields(title, html_id)}
        </div>
        """

    def _render_fields(self, title=None, parent_id=None):
        data = self.model_dump()
        html_parts = []
        
        if title:
            type_count = len(data.keys())
            html_parts.append(f"""
            <div class='pretty-header'>
                <span>🔹 {html.escape(title)}</span>
                <span class='pretty-header-badge'>{type_count} fields</span>
            </div>
            """)

        for k, v in data.items():
            # Highlight special fields
            key_class = ""
            if k in ["request_id", "provenance_id", "goal", "priority"]:
                key_class = "pretty-field-important"
                
            field_id = f"{parent_id}_{k}" if parent_id else f"field_{uuid.uuid4().hex}"
            
            html_parts.append(f"""
            <div class='pretty-kv'>
                <span class='pretty-key {key_class}'>{html.escape(str(k))}</span>
                <span class='pretty-value'>{self._render_value(v, field_id)}</span>
            </div>
            """)

        return "\n".join(html_parts)

    def _render_value(self, val, field_id=None):
        if val is None:
            return "<span class='pretty-null'>null</span>"
            
        if isinstance(val, bool):
            if val:
                return "<span class='pretty-boolean-true'>true</span>"
            else:
                return "<span class='pretty-boolean-false'>false</span>"
                
        if isinstance(val, (int, float)):
            # Special handling for confidence values
            if field_id and "confidence" in field_id and 0 <= val <= 1:
                percentage = int(val * 100)
                return f"""
                <span class='pretty-number'>{val}</span>
                <span class='confidence-meter'>
                    <span class='confidence-fill' style='width: {percentage}%'></span>
                </span>
                <span>({percentage}%)</span>
                """
            return f"<span class='pretty-number'>{val}</span>"
            
        if isinstance(val, str):
            if len(val) > 100:  # Collapsible for long strings
                short_val = html.escape(val[:100]) + "..."
                full_val = html.escape(val)
                collapse_id = f"collapse_{uuid.uuid4().hex}"
                return f"""
                <span class='pretty-collapsible pretty-collapsed pretty-string' id='{collapse_id}' 
                      onclick="toggleCollapse('{collapse_id}')">{short_val}</span>
                <div class='pretty-content pretty-string'>{full_val}</div>
                """
            return f"<span class='pretty-string'>{html.escape(val)}</span>"
            
        if isinstance(val, list):
            if not val:
                return "<i class='pretty-null'>[]</i>"
                
            collapse_id = f"collapse_{uuid.uuid4().hex}"
            item_count = len(val)
            
            # Make lists collapsible if they have many items
            if item_count > 3:
                return f"""
                <div>
                    <span class='pretty-collapsible' id='{collapse_id}' 
                          onclick="toggleCollapse('{collapse_id}')">
                        List ({item_count} items)
                    </span>
                    <div class='pretty-content'>
                        <ul class='pretty-list'>
                            {"".join(f"<li>{self._render_value(v)}</li>" for v in val)}
                        </ul>
                    </div>
                </div>
                """
            
            return "<ul class='pretty-list'>" + "".join(f"<li>{self._render_value(v)}</li>" for v in val) + "</ul>"
            
        elif isinstance(val, dict):
            collapse_id = f"collapse_{uuid.uuid4().hex}"
            key_count = len(val.keys())
            
            if key_count > 5:  # Collapsible for large dicts
                return f"""
                <div>
                    <span class='pretty-collapsible' id='{collapse_id}' 
                          onclick="toggleCollapse('{collapse_id}')">
                        Object ({key_count} properties)
                    </span>
                    <div class='pretty-content pretty-nested'>
                        {"".join(
                            f"<div><span class='pretty-key'>{html.escape(str(k))}</span>: {self._render_value(v)}</div>"
                            for k, v in val.items()
                        )}
                    </div>
                </div>
                """
            
            return "<div class='pretty-nested'>" + "".join(
                f"<div><span class='pretty-key'>{html.escape(str(k))}</span>: {self._render_value(v)}</div>"
                for k, v in val.items()
            ) + "</div>"
            
        elif isinstance(val, PrettyRenderableModel):
            collapse_id = f"collapse_{uuid.uuid4().hex}"
            return f"""
            <div>
                <span class='pretty-collapsible' id='{collapse_id}' 
                      onclick="toggleCollapse('{collapse_id}')">
                    {val.__class__.__name__}
                </span>
                <div class='pretty-content pretty-nested'>
                    {val._render_fields()}
                </div>
            </div>
            """
        else:
            return html.escape(str(val))


class ThoughtSegment(PrettyRenderableModel):
    content: str
    inference_type: str
    confidence: float

class ChainOfThought(PrettyRenderableModel):
    thoughts: List[ThoughtSegment]
    goal: str
    provenance_id: str

class Step(PrettyRenderableModel):
    instructions: str
    dependencies: List[int]
    tool_hint: str
    estimated_cost: float

class Plan(PrettyRenderableModel):
    chain_of_thought: ChainOfThought
    steps: List[Step]
    priority: int
    origin_agent: str

class DecomposedUserRequest(PrettyRenderableModel):
    initial_prompt: str
    plan: Plan
    tags: List[str]
    request_id: str
    

class ScaffoldAnalyzer:
    """Analyzer for scaffolded code"""
    
    def __init__(self, project_root: str):
        """Initialize the analyzer
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.models: Dict[str, Dict[str, Any]] = {}
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.services: Dict[str, Dict[str, Any]] = {}
        self.ui_components: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
    
    def scan_project(self) -> Dict[str, Any]:
        """Scan the project and analyze the components
        
        Returns:
            Dictionary with analysis results
        """
        self._scan_models()
        self._scan_endpoints()
        self._scan_services()
        self._scan_ui()
        self._analyze_relationships()
        
        return {
            "models": self.models,
            "endpoints": self.endpoints,
            "services": self.services,
            "ui_components": self.ui_components,
            "relationships": self.relationships
        }
    
    def _scan_models(self):
        """Scan and analyze model files"""
        models_dir = os.path.join(self.project_root, "backend", "models")
        if not os.path.exists(models_dir):
            return
        
        for file in os.listdir(models_dir):
            if file.endswith(".py") and not file.startswith("__"):
                path = os.path.join(models_dir, file)
                model_name = self._extract_model_name(path)
                if model_name:
                    fields = self._extract_model_fields(path)
                    self.models[model_name] = {
                        "name": model_name,
                        "path": path,
                        "fields": fields
                    }
    
    def _scan_endpoints(self):
        """Scan and analyze endpoint files"""
        controllers_dir = os.path.join(self.project_root, "backend", "controllers")
        if not os.path.exists(controllers_dir):
            return
        
        for file in os.listdir(controllers_dir):
            if file.endswith(".py") and not file.startswith("__"):
                path = os.path.join(controllers_dir, file)
                endpoint_info = self._extract_endpoint_info(path)
                if endpoint_info:
                    name = endpoint_info.get("name", "")
                    self.endpoints[name] = endpoint_info
    
    def _scan_services(self):
        """Scan and analyze service files"""
        services_dir = os.path.join(self.project_root, "backend", "services")
        if not os.path.exists(services_dir):
            return
        
        for file in os.listdir(services_dir):
            if file.endswith(".py") and not file.startswith("__"):
                path = os.path.join(services_dir, file)
                service_info = self._extract_service_info(path)
                if service_info:
                    name = service_info.get("name", "")
                    self.services[name] = service_info
    
    def _scan_ui(self):
        """Scan and analyze UI components"""
        ui_dir = os.path.join(self.project_root, "ui", "components")
        if not os.path.exists(ui_dir):
            return
        
        for root, dirs, files in os.walk(ui_dir):
            for file in files:
                if file.endswith((".tsx", ".jsx")) and not file.startswith("_"):
                    path = os.path.join(root, file)
                    component_info = self._extract_component_info(path)
                    if component_info:
                        name = component_info.get("name", "")
                        self.ui_components[name] = component_info
    
    def _analyze_relationships(self):
        """Analyze relationships between components"""
        # Model to endpoint relationships
        for endpoint_name, endpoint in self.endpoints.items():
            model_name = endpoint.get("model")
            if model_name and model_name in self.models:
                self.relationships.append({
                    "source": f"model:{model_name}",
                    "target": f"endpoint:{endpoint_name}",
                    "type": "uses"
                })
        
        # Endpoint to service relationships
        for endpoint_name, endpoint in self.endpoints.items():
            service_name = endpoint.get("service")
            if service_name and service_name in self.services:
                self.relationships.append({
                    "source": f"endpoint:{endpoint_name}",
                    "target": f"service:{service_name}",
                    "type": "uses"
                })
        
        # Service to model relationships
        for service_name, service in self.services.items():
            model_name = service.get("model")
            if model_name and model_name in self.models:
                self.relationships.append({
                    "source": f"service:{service_name}",
                    "target": f"model:{model_name}",
                    "type": "uses"
                })
        
        # UI components to service (via API) relationships
        for component_name, component in self.ui_components.items():
            api_endpoints = component.get("api_endpoints", [])
            for endpoint in api_endpoints:
                if endpoint in self.endpoints:
                    self.relationships.append({
                        "source": f"ui:{component_name}",
                        "target": f"endpoint:{endpoint}",
                        "type": "calls"
                    })
    
    def _extract_model_name(self, path: str) -> Optional[str]:
        """Extract model name from a file
        
        Args:
            path: Path to the model file
            
        Returns:
            Model name if found, None otherwise
        """
        with open(path, "r") as f:
            content = f.read()
        
        # Look for class definitions that inherit from FortitudeBaseModel
        match = re.search(r"class\s+(\w+)\s*\(.*FortitudeBaseModel", content)
        if match:
            return match.group(1)
        return None
    
    def _extract_model_fields(self, path: str) -> List[Dict[str, Any]]:
        """Extract fields from a model file
        
        Args:
            path: Path to the model file
            
        Returns:
            List of field definitions
        """
        fields = []
        with open(path, "r") as f:
            content = f.read()
        
        # Extract field definitions
        field_pattern = r"^\s+(\w+)\s*:\s*(?:Optional\[)?([^=\s]+)(?:\])?\s*(?:=\s*(.+))?$"
        matches = re.finditer(field_pattern, content, re.MULTILINE)
        
        for match in matches:
            field_name = match.group(1)
            field_type = match.group(2)
            default_value = match.group(3)
            
            fields.append({
                "name": field_name,
                "type": field_type,
                "required": default_value is None
            })
        
        return fields
    
    def _extract_endpoint_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Extract endpoint information from a controller file
        
        Args:
            path: Path to the controller file
            
        Returns:
            Endpoint information if found, None otherwise
        """
        with open(path, "r") as f:
            content = f.read()
        
        # Extract router prefix
        prefix_match = re.search(r"router\s*=\s*APIRouter\s*\(\s*prefix\s*=\s*['\"]([^'\"]+)['\"]", content)
        prefix = prefix_match.group(1) if prefix_match else ""
        
        # Extract the model being used
        model_match = re.search(r"from\s+\.\.models\.(\w+)\s+import\s+(\w+)", content)
        model_file = model_match.group(1) if model_match else ""
        model_name = model_match.group(2) if model_match else ""
        
        # Extract the service being used
        service_match = re.search(r"from\s+\.\.services\.(\w+)\s+import\s+(\w+)", content)
        service_file = service_match.group(1) if service_match else ""
        service_name = service_match.group(2) if service_match else ""
        
        # Extract route operations
        operations = []
        route_patterns = [
            r"@router\.(\w+)\s*\(['\"]([^'\"]*)['\"]",
            r"@router\.(\w+)\s*\(['\"]([^'\"]*)['\"][^)]*response_model\s*=\s*(\w+)"
        ]
        
        for pattern in route_patterns:
            for match in re.finditer(pattern, content):
                method = match.group(1)
                path = match.group(2)
                response_model = match.group(3) if len(match.groups()) > 2 else None
                
                operations.append({
                    "method": method,
                    "path": path,
                    "response_model": response_model
                })
        
        # Get the controller name from the file name
        file_name = os.path.basename(path)
        controller_name = file_name.replace("_controller.py", "")
        
        return {
            "name": controller_name,
            "prefix": prefix,
            "model": model_name,
            "service": service_name,
            "operations": operations,
            "path": path
        }
    
    def _extract_service_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Extract service information from a service file
        
        Args:
            path: Path to the service file
            
        Returns:
            Service information if found, None otherwise
        """
        with open(path, "r") as f:
            content = f.read()
        
        # Extract the model being used
        model_match = re.search(r"from\s+\.\.models\.(\w+)\s+import\s+(\w+)", content)
        model_file = model_match.group(1) if model_match else ""
        model_name = model_match.group(2) if model_match else ""
        
        # Extract service class name
        class_match = re.search(r"class\s+(\w+):", content)
        class_name = class_match.group(1) if class_match else ""
        
        # Extract service methods
        methods = []
        method_pattern = r"async\s+def\s+(\w+)\s*\([^)]*\).*:"
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            methods.append(method_name)
        
        # Get the service name from the file name
        file_name = os.path.basename(path)
        service_name = file_name.replace("_service.py", "")
        
        return {
            "name": class_name,
            "model": model_name,
            "methods": methods,
            "path": path
        }
    
    def _extract_component_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Extract component information from a UI component file
        
        Args:
            path: Path to the component file
            
        Returns:
            Component information if found, None otherwise
        """
        with open(path, "r") as f:
            content = f.read()
        
        # Extract component name
        name_match = re.search(r"function\s+(\w+)", content)
        if not name_match:
            name_match = re.search(r"const\s+(\w+)\s*=", content)
        
        component_name = name_match.group(1) if name_match else ""
        
        # Extract API endpoints used
        api_endpoints = []
        api_pattern = r"fetch\(['\"]([^'\"]*)['\"]"
        for match in re.finditer(api_pattern, content):
            endpoint = match.group(1)
            # Extract the endpoint name from the path
            endpoint_match = re.search(r"/api/(\w+)", endpoint)
            if endpoint_match:
                api_endpoints.append(endpoint_match.group(1))
        
        return {
            "name": component_name,
            "path": path,
            "api_endpoints": list(set(api_endpoints))
        }

def generate_scaffold_graph(analysis_result: Dict[str, Any], output_path: str):
    """Generate a graph visualization of the scaffold
    
    Args:
        analysis_result: The analysis result from ScaffoldAnalyzer
        output_path: Path to save the output file
    """
    nodes = []
    edges = []
    
    # Add model nodes
    for model_name, model in analysis_result["models"].items():
        nodes.append({
            "id": f"model:{model_name}",
            "label": model_name,
            "type": "model",
            "details": model
        })
    
    # Add endpoint nodes
    for endpoint_name, endpoint in analysis_result["endpoints"].items():
        nodes.append({
            "id": f"endpoint:{endpoint_name}",
            "label": endpoint_name,
            "type": "endpoint",
            "details": endpoint
        })
    
    # Add service nodes
    for service_name, service in analysis_result["services"].items():
        nodes.append({
            "id": f"service:{service_name}",
            "label": service_name,
            "type": "service",
            "details": service
        })
    
    # Add UI component nodes
    for component_name, component in analysis_result["ui_components"].items():
        nodes.append({
            "id": f"ui:{component_name}",
            "label": component_name,
            "type": "ui",
            "details": component
        })
    
    # Add edges
    for relationship in analysis_result["relationships"]:
        edges.append({
            "source": relationship["source"],
            "target": relationship["target"],
            "label": relationship["type"]
        })
    
    graph = {
        "nodes": nodes,
        "edges": edges
    }
    
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)

def generate_rails_style_scaffold_preview(name: str, fields: List[Dict[str, Any]], output_dir: str = None) -> str:
    """Generate a Rails-style preview of what will be scaffolded
    
    Args:
        name: Name of the resource
        fields: List of field definitions
        output_dir: Directory to output the preview
        
    Returns:
        Path to the generated preview file
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    output_path = os.path.join(output_dir, f"{name.lower()}_scaffold_preview.md")
    
    markdown = f"# Scaffold Preview for {name}\n\n"
    
    # Model preview
    markdown += "## Model\n\n"
    markdown += "```python\n"
    markdown += f"class {name}(FortitudeBaseModel):\n"
    for field in fields:
        field_name = field["name"]
        field_type = field["type"]
        required = field.get("required", True)
        
        if required:
            markdown += f"    {field_name}: {field_type}\n"
        else:
            markdown += f"    {field_name}: Optional[{field_type}] = None\n"
    markdown += "```\n\n"
    
    # Controller preview
    markdown += "## Controller\n\n"
    markdown += "```python\n"
    markdown += f"router = APIRouter(prefix=\"/{name.lower()}\", tags=[\"{name}\"])\n\n"
    markdown += f"@router.post(\"/\", response_model={name})\n"
    markdown += f"async def create_{name.lower()}(data: Dict[str, Any]):\n"
    markdown += f"    return await service.create(data)\n\n"
    markdown += f"@router.get(\"/{{{name.lower()}_id}}\", response_model={name})\n"
    markdown += f"async def get_{name.lower()}(id: str):\n"
    markdown += f"    return await service.get(id)\n\n"
    markdown += f"@router.get(\"/\", response_model=List[{name}])\n"
    markdown += f"async def list_{name.lower()}():\n"
    markdown += f"    return await service.list()\n\n"
    markdown += f"@router.put(\"/{{{name.lower()}_id}}\", response_model={name})\n"
    markdown += f"async def update_{name.lower()}(id: str, data: Dict[str, Any]):\n"
    markdown += f"    return await service.update(id, data)\n\n"
    markdown += f"@router.delete(\"/{{{name.lower()}_id}}\")\n"
    markdown += f"async def delete_{name.lower()}(id: str):\n"
    markdown += f"    return await service.delete(id)\n"
    markdown += "```\n\n"
    
    # UI components preview
    markdown += "## UI Components\n\n"
    
    # List component
    markdown += "### List Component\n\n"
    markdown += "```tsx\n"
    markdown += f"export default function {name}List() {{\n"
    markdown += "  const [items, setItems] = useState([]);\n\n"
    markdown += "  useEffect(() => {\n"
    markdown += f"    fetch('/api/{name.lower()}')\n"
    markdown += "      .then(res => res.json())\n"
    markdown += "      .then(data => setItems(data));\n"
    markdown += "  }, []);\n\n"
    markdown += "  return (\n"
    markdown += "    <table>\n"
    markdown += "      <thead>\n"
    markdown += "        <tr>\n"
    for field in fields:
        markdown += f"          <th>{field['name']}</th>\n"
    markdown += "          <th>Actions</th>\n"
    markdown += "        </tr>\n"
    markdown += "      </thead>\n"
    markdown += "      <tbody>\n"
    markdown += "        {items.map(item => (\n"
    markdown += "          <tr key={item.id}>\n"
    for field in fields:
        field_name = field["name"]
        markdown += f"            <td>{{item.{field_name}}}</td>\n"
    markdown += "            <td>\n"
    markdown += f"              <Link href={{`/{name.lower()}/edit/${{item.id}}`}}>Edit</Link>\n"
    markdown += f"              <button onClick={{() => handleDelete(item.id)}}>Delete</button>\n"
    markdown += "            </td>\n"
    markdown += "          </tr>\n"
    markdown += "        ))}\n"
    markdown += "      </tbody>\n"
    markdown += "    </table>\n"
    markdown += "  );\n"
    markdown += "}\n"
    markdown += "```\n\n"
    
    # Form component
    markdown += "### Form Component\n\n"
    markdown += "```tsx\n"
    markdown += f"export default function {name}Form({{ id, onSubmit }}) {{\n"
    markdown += "  const [formData, setFormData] = useState({\n"
    for field in fields:
        field_name = field["name"]
        default_value = '""' if field["type"] in ["str", "string"] else "null"
        markdown += f"    {field_name}: {default_value},\n"
    markdown += "  });\n\n"
    markdown += "  // Fetch data if editing\n"
    markdown += "  useEffect(() => {\n"
    markdown += "    if (id) {\n"
    markdown += f"      fetch(`/api/{name.lower()}/${{id}}`)\n"
    markdown += "        .then(res => res.json())\n"
    markdown += "        .then(data => setFormData(data));\n"
    markdown += "    }\n"
    markdown += "  }, [id]);\n\n"
    markdown += "  return (\n"
    markdown += "    <form onSubmit={(e) => {\n"
    markdown += "      e.preventDefault();\n"
    markdown += "      onSubmit(formData);\n"
    markdown += "    }}>\n"
    for field in fields:
        field_name = field["name"]
        required = "required" if field.get("required", True) else ""
        markdown += f"      <div>\n"
        markdown += f"        <label htmlFor=\"{field_name}\">{field_name.capitalize()}</label>\n"
        markdown += f"        <input\n"
        markdown += f"          id=\"{field_name}\"\n"
        markdown += f"          name=\"{field_name}\"\n"
        markdown += f"          value={{formData.{field_name} || ''}}\n"
        markdown += f"          onChange={{e => setFormData({{...formData, {field_name}: e.target.value}})}} {required}\n"
        markdown += f"        />\n"
        markdown += f"      </div>\n"
    markdown += "      <button type=\"submit\">{id ? 'Update' : 'Create'}</button>\n"
    markdown += "    </form>\n"
    markdown += "  );\n"
    markdown += "}\n"
    markdown += "```\n\n"
    
    # Files that would be generated
    markdown += "## Files That Will Be Generated\n\n"
    markdown += "```\n"
    markdown += f"├── backend/\n"
    markdown += f"│   ├── models/\n"
    markdown += f"│   │   └── {name.lower()}.py\n"
    markdown += f"│   ├── controllers/\n"
    markdown += f"│   │   └── {name.lower()}_controller.py\n"
    markdown += f"│   └── services/\n"
    markdown += f"│       └── {name.lower()}_service.py\n"
    markdown += f"├── ui/\n"
    markdown += f"│   ├── pages/\n"
    markdown += f"│   │   └── {name.lower()}.tsx\n"
    markdown += f"│   └── components/\n"
    markdown += f"│       └── {name.lower()}/\n"
    markdown += f"│           ├── {name}List.tsx\n"
    markdown += f"│           └── {name}Form.tsx\n"
    markdown += f"└── tests/\n"
    markdown += f"    ├── models/\n"
    markdown += f"    │   └── test_{name.lower()}.py\n"
    markdown += f"    ├── controllers/\n"
    markdown += f"    │   └── test_{name.lower()}_controller.py\n"
    markdown += f"    └── services/\n"
    markdown += f"        └── test_{name.lower()}_service.py\n"
    markdown += "```\n"
    
    with open(output_path, "w") as f:
        f.write(markdown)
    
    return output_path

# ----------------- NEW CLASSES FOR AGENT TRAINING DATA GENERATION -----------------

class AgentAction(PrettyRenderableModel):
    """Represents a single action taken by an agent"""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str
    description: str
    parameters: Dict[str, Any] = {}
    result: Optional[Any] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

class AgentSession(PrettyRenderableModel):
    """Represents a longer-running session with an agent across multiple interactions"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_metadata: Dict[str, Any] = {}
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    session_goals: List[str] = []
    accumulated_context: Dict[str, Any] = {}
    active: bool = True
    interactions: List[str] = []  # Interaction IDs

class AgentInteraction(PrettyRenderableModel):
    """Represents an interaction between human and agent"""
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    human_input: str
    agent_actions: List[AgentAction] = []
    final_response: Optional[str] = None
    context_info: Dict[str, Any] = {}
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completion_status: str = "pending"  # pending, completed, failed
    session_id: Optional[str] = None  # Reference to parent session if any
    semantic_parser_results: Optional[Dict[str, Any]] = None  # Structured parsing of input
    response_evaluation: Optional[Dict[str, Any]] = None  # Feedback on response quality
    thinking_steps: List[Dict[str, Any]] = []  # Reasoning steps (if available)
    time_to_completion_ms: Optional[float] = None
    resource_usage: Dict[str, Any] = {}  # CPU, memory, etc. used during processing
    human_feedback: Optional[Dict[str, Any]] = None  # User feedback on interaction

class AgentTrainingCorpus(PrettyRenderableModel):
    """Collection of agent interactions for training purposes"""
    corpus_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    interactions: List[AgentInteraction] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    tags: List[str] = []
    version: str = "1.0"

class ActionSequence(PrettyRenderableModel):
    """A normalized sequence of actions, suitable for training"""
    sequence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str
    actions: List[Dict[str, Any]] = []
    expected_output: Optional[str] = None
    difficulty: float = 0.5  # 0.0 to 1.0
    domain: str
    features: List[str] = []
    requires_external_context: bool = False
    contains_counterfactual: bool = False
    semantic_embedding: Optional[List[float]] = None
    challenge_factors: Dict[str, float] = {}  # e.g., {"ambiguity": 0.7, "complexity": 0.8}
    quality_score: Optional[float] = None
    variants: List[str] = []  # IDs of variant sequences (paraphrases, etc.)
    interaction_dependencies: List[str] = []  # IDs of prerequisite interactions
    state_tracking: Dict[str, Any] = {}  # Information about state changes
    annotations: Dict[str, Any] = {}

class TrainingCurriculum(PrettyRenderableModel):
    """A structured curriculum for progressive agent training"""
    curriculum_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    stages: List[Dict[str, Any]] = []  # Progressive difficulty stages
    prerequisites: Dict[str, List[str]] = {}  # Skill dependencies
    mastery_criteria: Dict[str, Dict[str, float]] = {}  # Metrics to achieve for each skill
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    target_capabilities: List[str] = []
    expected_training_steps: Optional[int] = None

class SupervisedFinetuningExample(PrettyRenderableModel):
    """A processed example for supervised finetuning"""
    example_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_prompt: str
    target_completion: str
    domain: str
    difficulty: float = 0.5
    task_type: str  # classification, generation, etc.
    metadata: Dict[str, Any] = {}
    original_sequence_id: Optional[str] = None  # Reference to source ActionSequence

class TrainingDataset(PrettyRenderableModel):
    """A dataset of action sequences for training agents"""
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    sequences: List[ActionSequence] = []
    statistics: Dict[str, Any] = {}
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    version: str = "1.0"
    split: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    curriculum: Optional[TrainingCurriculum] = None
    supervised_examples: List[SupervisedFinetuningExample] = []
    schema_version: str = "2.0"
    validation_metrics: Dict[str, float] = {}
    preprocessing_pipeline: List[Dict[str, Any]] = []
    taxonomies: Dict[str, List[str]] = {}  # Domain-specific categorization schemes
    capability_matrix: Dict[str, Dict[str, float]] = {}  # Capability coverage assessment

class NeuralAdapter:
    """Neural adapter for processing agent training data"""
    
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.embeddings_dimension = 1536  # Default for modern embeddings
        self.batch_size = 32
        self.adapter_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "prompt_template": "{system_prompt}\n\n{instruction}\n\n{input}",
            "max_tokens": 2048
        }
        self.cache = {}
        self.last_used = datetime.datetime.now()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (placeholder implementation)"""
        # In a real implementation, this would call a vector embedding model
        # For demonstration, we'll use a deterministic hash-based approach
        if text in self.cache:
            return self.cache[text]
            
        # Create a simple embedding (would be a real model in production)
        hash_val = sum(ord(c) * (i+1) for i, c in enumerate(text[:100])) % 1000
        result = [((hash_val + i) % 1000) / 1000 for i in range(self.embeddings_dimension)]
        
        # Cache the result
        self.cache[text] = result
        self.last_used = datetime.datetime.now()
        
        return result
    
    async def generate_completion(self, 
                               system_prompt: str,
                               instruction: str,
                               input_text: str) -> str:
        """Generate completion (placeholder implementation)"""
        # In a real implementation, this would call a language model
        # For demonstration, we'll just concatenate the inputs with a templated response
        prompt = self.adapter_config["prompt_template"].format(
            system_prompt=system_prompt,
            instruction=instruction,
            input=input_text
        )
        
        # Simple deterministic response generation
        response = f"Response to: {input_text[:50]}..."
        self.last_used = datetime.datetime.now()
        
        return response
    
    async def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze text for various properties"""
        # Placeholder for text analysis
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        if analysis_type == "intent":
            # Simple intent analysis
            intents = []
            if "?" in text:
                intents.append("question")
            if any(cmd in text.lower() for cmd in ["create", "make", "add"]):
                intents.append("creation")
            if any(cmd in text.lower() for cmd in ["update", "change", "modify"]):
                intents.append("modification")
            if any(cmd in text.lower() for cmd in ["delete", "remove"]):
                intents.append("deletion")
            
            return {
                "intents": intents,
                "confidence": min(1.0, max(0.5, unique_words / max(1, word_count)))
            }
        elif analysis_type == "complexity":
            # Simple complexity analysis
            complexity_score = min(1.0, word_count / 50)
            return {
                "complexity": complexity_score,
                "word_count": word_count,
                "vocabulary_diversity": unique_words / max(1, word_count)
            }
        
        return {"analysis_type": analysis_type, "text_length": len(text)}

class TrainingPipeline:
    """Processing pipeline for training data generation"""
    
    def __init__(self, name: str, steps: List[Dict[str, Any]]):
        self.name = name
        self.steps = steps
        self.metrics: Dict[str, Any] = {}
        self.created_at = datetime.datetime.now()
        self.last_run = None
        
    async def run(self, data: Any, config: Dict[str, Any] = None) -> Any:
        """Run the pipeline on input data"""
        if config is None:
            config = {}
            
        result = data
        self.metrics = {"started_at": datetime.datetime.now()}
        
        try:
            for i, step in enumerate(self.steps):
                step_name = step.get("name", f"step_{i}")
                step_func = step.get("function")
                step_config = {**step.get("config", {}), **config}
                
                if step_func:
                    # Time the step execution
                    start_time = datetime.datetime.now()
                    
                    # Run the step
                    result = await step_func(result, **step_config)
                    
                    # Record metrics
                    end_time = datetime.datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Update step metrics
                    self.metrics[step_name] = {
                        "duration_seconds": duration,
                        "input_size": self._get_size(data),
                        "output_size": self._get_size(result),
                        "finished_at": end_time
                    }
                    
                    # Update step-specific metrics if provided
                    if hasattr(result, "metrics") and isinstance(result.metrics, dict):
                        self.metrics[step_name].update(result.metrics)
            
            self.metrics["total_duration"] = (datetime.datetime.now() - self.metrics["started_at"]).total_seconds()
            self.metrics["status"] = "completed"
            
        except Exception as e:
            self.metrics["error"] = str(e)
            self.metrics["status"] = "failed"
            raise
        
        self.last_run = datetime.datetime.now()
        return result
    
    def _get_size(self, data: Any) -> int:
        """Get approximate size of data"""
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
        if hasattr(data, "__len__"):
            return len(data)
        return 0

class NeuralTrainingEnvironment:
    """Environment for neural model training"""
    
    def __init__(self):
        self.adapters: Dict[str, NeuralAdapter] = {
            "default": NeuralAdapter("default")
        }
        self.active_pipelines: Dict[str, TrainingPipeline] = {}
        self.vector_database = {}  # Simple in-memory vector DB
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.telemetry: Dict[str, Any] = {
            "api_calls": 0,
            "embeddings_generated": 0,
            "training_examples_created": 0,
            "start_time": datetime.datetime.now()
        }
        
    async def get_adapter(self, model_name: str = "default") -> NeuralAdapter:
        """Get or create a neural adapter"""
        if model_name not in self.adapters:
            self.adapters[model_name] = NeuralAdapter(model_name)
        return self.adapters[model_name]
        
    async def create_pipeline(self, name: str, steps: List[Dict[str, Any]]) -> TrainingPipeline:
        """Create a new processing pipeline"""
        pipeline = TrainingPipeline(name, steps)
        self.active_pipelines[name] = pipeline
        return pipeline
        
    async def register_vector(self, key: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Register a vector in the vector database"""
        if metadata is None:
            metadata = {}
        self.vector_database[key] = {"vector": vector, "metadata": metadata}
        self.telemetry["embeddings_generated"] += 1
        
    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vectors by similarity"""
        if not self.vector_database:
            return []
            
        # Simple cosine similarity
        scores = []
        for key, entry in self.vector_database.items():
            vector = entry["vector"]
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(query_vector, vector))
            # Calculate magnitudes
            mag_query = sum(a * a for a in query_vector) ** 0.5
            mag_vector = sum(a * a for a in vector) ** 0.5
            # Calculate similarity
            similarity = dot_product / (mag_query * mag_vector) if mag_query * mag_vector > 0 else 0
            
            scores.append({
                "key": key,
                "similarity": similarity,
                "metadata": entry["metadata"]
            })
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        return scores[:top_k]
        
    async def start_session(self, session_id: str, config: Dict[str, Any] = None) -> str:
        """Start a new training session"""
        if config is None:
            config = {}
            
        self.active_sessions[session_id] = {
            "started_at": datetime.datetime.now(),
            "config": config,
            "status": "active",
            "metrics": {},
            "processed_items": 0
        }
        
        return session_id
        
    async def update_session(self, session_id: str, metrics: Dict[str, Any]):
        """Update session metrics"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["metrics"].update(metrics)
            self.active_sessions[session_id]["updated_at"] = datetime.datetime.now()
            self.active_sessions[session_id]["processed_items"] += metrics.get("processed_items", 0)
        
    async def end_session(self, session_id: str, status: str = "completed"):
        """End a training session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = status
            self.active_sessions[session_id]["ended_at"] = datetime.datetime.now()
            duration = (self.active_sessions[session_id]["ended_at"] - 
                       self.active_sessions[session_id]["started_at"]).total_seconds()
            self.active_sessions[session_id]["duration_seconds"] = duration

class AgentTrainingManager:
    """Manages the generation and processing of agent training data"""
    
    def __init__(self):
        self.corpora: Dict[str, AgentTrainingCorpus] = {}
        self.datasets: Dict[str, TrainingDataset] = {}
        self.action_templates: Dict[str, Dict[str, Any]] = {}
        self.interaction_observers: List[Callable] = []
        self.domain_specific_handlers: Dict[str, Callable] = {}
        self.anomaly_detectors: Dict[str, Callable] = {}
        self.quality_filters: Dict[str, Callable] = {}
        self.counterfactual_generators: Dict[str, Callable] = {}
        self.embedding_cache: Dict[str, Any] = {}
        
        # Advanced components
        self.neural_env = NeuralTrainingEnvironment()
        self.fine_tuning_adapters: Dict[str, Any] = {}
        self.active_sessions: Dict[str, AgentSession] = {}
        self.data_augmentation_strategies: Dict[str, Callable] = {}
        self.feature_extraction_pipelines: Dict[str, TrainingPipeline] = {}
        self.simulation_environments: Dict[str, Any] = {}
        self.evaluation_benchmarks: Dict[str, Dict[str, Any]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
    async def record_interaction(self, human_input: str, context_info: Dict[str, Any] = {}) -> AgentInteraction:
        """Records a new human-agent interaction"""
        interaction = AgentInteraction(
            human_input=human_input,
            context_info=context_info
        )
        
        # Notify observers
        for observer in self.interaction_observers:
            await observer(interaction)
            
        return interaction
    
    async def record_action(self, interaction: AgentInteraction, 
                          action_type: str, description: str,
                          parameters: Dict[str, Any] = {}) -> AgentAction:
        """Records an action taken by the agent"""
        start_time = datetime.datetime.now()
        
        action = AgentAction(
            action_type=action_type,
            description=description,
            parameters=parameters,
            timestamp=start_time
        )
        
        # Add to interaction
        interaction.agent_actions.append(action)
        
        return action
    
    async def complete_action(self, action: AgentAction, result: Any = None, error: Optional[str] = None):
        """Completes an action with its result or error"""
        end_time = datetime.datetime.now()
        execution_time = (end_time - action.timestamp).total_seconds() * 1000
        
        action.result = result
        action.error = error
        action.execution_time_ms = execution_time
        
    async def complete_interaction(self, interaction: AgentInteraction, final_response: str, 
                                 status: str = "completed"):
        """Completes an interaction with the final response"""
        interaction.final_response = final_response
        interaction.completion_status = status
        
    async def create_corpus(self, name: str, description: str, tags: List[str] = []) -> AgentTrainingCorpus:
        """Creates a new training corpus"""
        corpus = AgentTrainingCorpus(
            name=name,
            description=description,
            tags=tags
        )
        
        self.corpora[corpus.corpus_id] = corpus
        return corpus
    
    async def add_to_corpus(self, corpus_id: str, interaction: AgentInteraction):
        """Adds an interaction to a corpus"""
        if corpus_id not in self.corpora:
            raise ValueError(f"Corpus with ID {corpus_id} not found")
        
        corpus = self.corpora[corpus_id]
        corpus.interactions.append(interaction)
        corpus.updated_at = datetime.datetime.now()
        
    async def create_training_dataset(self, corpus_ids: List[str], name: str, 
                                    description: str, 
                                    create_curriculum: bool = False,
                                    generate_supervised_examples: bool = True) -> TrainingDataset:
        """Creates a training dataset from multiple corpora with advanced processing options
        
        Args:
            corpus_ids: List of corpus IDs to include
            name: Name of the dataset
            description: Description of the dataset
            create_curriculum: Whether to create a training curriculum
            generate_supervised_examples: Whether to generate supervised examples
            
        Returns:
            The created dataset
        """
        dataset = TrainingDataset(
            name=name,
            description=description
        )
        
        # Process all interactions in all corpora
        all_interactions = []
        for corpus_id in corpus_ids:
            if corpus_id not in self.corpora:
                raise ValueError(f"Corpus with ID {corpus_id} not found")
            
            corpus = self.corpora[corpus_id]
            all_interactions.extend(corpus.interactions)
        
        # Convert to action sequences
        sequences = await self._convert_to_sequences(all_interactions)
        dataset.sequences = sequences
        
        # Generate embeddings for sequences if not already present
        for sequence in sequences:
            if sequence.semantic_embedding is None:
                try:
                    # Generate a simple embedding from the input text (in a real system, use a proper embedding model)
                    # This is a placeholder for demonstration
                    simple_hash = sum(ord(c) for c in sequence.input) % 100
                    sequence.semantic_embedding = [simple_hash / 100.0] * 10  # 10-dim placeholder embedding
                except Exception as e:
                    print(f"Error generating embedding: {e}")
        
        # Create a training curriculum if requested
        if create_curriculum:
            curriculum = await self._create_curriculum(sequences)
            dataset.curriculum = curriculum
        
        # Generate supervised examples if requested
        if generate_supervised_examples:
            supervised_examples = await self._generate_supervised_examples(sequences)
            dataset.supervised_examples = supervised_examples
        
        # Generate counterfactuals for challenging sequences
        await self._generate_counterfactuals(sequences, dataset)
        
        # Compute statistics
        dataset.statistics = await self._compute_dataset_statistics(sequences)
        
        # Calculate capability coverage
        capability_matrix = await self._compute_capability_coverage(sequences)
        dataset.capability_matrix = capability_matrix
        
        # Generate taxonomies
        taxonomies = self._extract_taxonomies(sequences)
        dataset.taxonomies = taxonomies
        
        # Store preprocessing pipeline configuration for reproducibility
        dataset.preprocessing_pipeline = [
            {"step": "sequence_conversion", "params": {"filter_incomplete": True}},
            {"step": "embedding_generation", "params": {"dimensions": 10, "method": "simple_hash"}},
            {"step": "curriculum_creation", "params": {"enabled": create_curriculum}},
            {"step": "supervised_example_generation", "params": {"enabled": generate_supervised_examples}},
            {"step": "counterfactual_generation", "params": {"threshold": 0.7}},
            {"step": "quality_filtering", "params": {"filters": list(self.quality_filters.keys())}},
        ]
        
        # Validate dataset quality
        validation_metrics = await self._validate_dataset(dataset)
        dataset.validation_metrics = validation_metrics
        
        self.datasets[dataset.dataset_id] = dataset
        return dataset
        
    async def _create_curriculum(self, sequences: List[ActionSequence]) -> TrainingCurriculum:
        """Creates a progressive training curriculum from sequences"""
        # Group sequences by difficulty
        difficulty_groups = {}
        for sequence in sequences:
            # Round difficulty to nearest 0.1 for grouping
            rounded_diff = round(sequence.difficulty * 10) / 10
            if rounded_diff not in difficulty_groups:
                difficulty_groups[rounded_diff] = []
            difficulty_groups[rounded_diff].append(sequence)
        
        # Sort difficulties
        sorted_difficulties = sorted(difficulty_groups.keys())
        
        # Create curriculum stages
        stages = []
        for i, difficulty in enumerate(sorted_difficulties):
            sequences_in_stage = difficulty_groups[difficulty]
            # Extract most common features at this difficulty level
            feature_counts = {}
            for seq in sequences_in_stage:
                for feature in seq.features:
                    if feature not in feature_counts:
                        feature_counts[feature] = 0
                    feature_counts[feature] += 1
            
            # Get top features
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            stage = {
                "level": i + 1,
                "difficulty": difficulty,
                "sequence_count": len(sequences_in_stage),
                "focus_features": [feature for feature, _ in top_features],
                "sequence_ids": [seq.sequence_id for seq in sequences_in_stage]
            }
            stages.append(stage)
        
        # Create feature prerequisites (simple heuristic)
        prerequisites = {}
        base_features = set()
        for i, stage in enumerate(stages):
            if i == 0:
                # First stage has no prerequisites
                base_features.update(stage["focus_features"])
                continue
                
            for feature in stage["focus_features"]:
                if feature not in base_features:
                    # New feature depends on previous stage features
                    prerequisites[feature] = list(base_features)
            
            # Add this stage's features to base features
            base_features.update(stage["focus_features"])
        
        # Create mastery criteria
        mastery_criteria = {}
        for i, stage in enumerate(stages):
            for feature in stage["focus_features"]:
                # Simple mastery criteria
                mastery_criteria[feature] = {
                    "accuracy": 0.8,
                    "completion_rate": 0.9,
                    "average_time_ms": 5000 * (i + 1)  # Higher stages allow more time
                }
        
        # Extract target capabilities
        all_features = set()
        for sequence in sequences:
            all_features.update(sequence.features)
        
        return TrainingCurriculum(
            name="Adaptive Fortitude Agent Curriculum",
            description="Progressive training curriculum for Fortitude agents",
            stages=stages,
            prerequisites=prerequisites,
            mastery_criteria=mastery_criteria,
            target_capabilities=list(all_features),
            expected_training_steps=len(sequences) * 5  # Simple estimation
        )
        
    async def _generate_supervised_examples(self, sequences: List[ActionSequence]) -> List[SupervisedFinetuningExample]:
        """Generates supervised fine-tuning examples from action sequences"""
        examples = []
        
        for sequence in sequences:
            # Skip low quality sequences
            if sequence.quality_score and sequence.quality_score < 0.6:
                continue
                
            # Create direct input -> output example
            direct_example = SupervisedFinetuningExample(
                input_prompt=sequence.input,
                target_completion=sequence.expected_output or "",
                domain=sequence.domain,
                difficulty=sequence.difficulty,
                task_type="generation",
                original_sequence_id=sequence.sequence_id,
                metadata={
                    "example_type": "direct",
                    "features": sequence.features
                }
            )
            examples.append(direct_example)
            
            # Create tool reasoning example if sequence has multiple actions
            if len(sequence.actions) > 1:
                # Construct a reasoning chain based on the actions
                reasoning_steps = []
                for i, action in enumerate(sequence.actions):
                    reasoning_steps.append(f"Step {i+1}: I need to {action['name']} using {action['type']} with parameters {json.dumps(action['params'])}")
                    if action.get('result'):
                        reasoning_steps.append(f"Result: {json.dumps(action['result'])}")
                
                reasoning_completion = "\n".join(reasoning_steps) + f"\n\nFinal answer: {sequence.expected_output or ''}"
                
                reasoning_example = SupervisedFinetuningExample(
                    input_prompt=f"Solve this step by step: {sequence.input}",
                    target_completion=reasoning_completion,
                    domain=sequence.domain,
                    difficulty=sequence.difficulty + 0.1,  # Slightly harder
                    task_type="chain_of_thought",
                    original_sequence_id=sequence.sequence_id,
                    metadata={
                        "example_type": "reasoning",
                        "features": sequence.features,
                        "action_count": len(sequence.actions)
                    }
                )
                examples.append(reasoning_example)
                
                # Create tool selection example
                if "tool_usage" in sequence.features:
                    tool_types = [action["type"] for action in sequence.actions]
                    tool_selection_example = SupervisedFinetuningExample(
                        input_prompt=f"What tools would you use to solve this problem? Problem: {sequence.input}",
                        target_completion=f"To solve this problem, I would use the following tools in sequence: {', '.join(tool_types)}",
                        domain=sequence.domain,
                        difficulty=sequence.difficulty - 0.1,  # Slightly easier
                        task_type="tool_selection",
                        original_sequence_id=sequence.sequence_id,
                        metadata={
                            "example_type": "tool_selection",
                            "features": sequence.features
                        }
                    )
                    examples.append(tool_selection_example)
            
        return examples
        
    async def _generate_counterfactuals(self, sequences: List[ActionSequence], dataset: TrainingDataset):
        """Generates counterfactual variations of challenging sequences"""
        # Only process sufficiently difficult sequences
        challenging_sequences = [seq for seq in sequences if seq.difficulty > 0.7]
        
        for sequence in challenging_sequences:
            # Skip if no counterfactual generator for this domain
            if sequence.domain not in self.counterfactual_generators:
                continue
                
            generator = self.counterfactual_generators[sequence.domain]
            try:
                counterfactual = await generator(sequence)
                if counterfactual:
                    # Mark as counterfactual
                    counterfactual.contains_counterfactual = True
                    # Add original sequence as dependency
                    counterfactual.interaction_dependencies.append(sequence.sequence_id)
                    # Link as variant
                    sequence.variants.append(counterfactual.sequence_id)
                    # Add to sequences
                    sequences.append(counterfactual)
            except Exception as e:
                print(f"Error generating counterfactual: {e}")
    
    def _extract_taxonomies(self, sequences: List[ActionSequence]) -> Dict[str, List[str]]:
        """Extract taxonomic categorizations from sequences"""
        # Domain taxonomy
        domains = set()
        for sequence in sequences:
            domains.add(sequence.domain)
        
        # Feature taxonomy
        features = set()
        for sequence in sequences:
            features.update(sequence.features)
        
        # Tool taxonomy
        tools = set()
        for sequence in sequences:
            for action in sequence.actions:
                if action["type"] == "tool_call":
                    tools.add(action["name"])
        
        # Challenge factor taxonomy
        challenge_factors = set()
        for sequence in sequences:
            challenge_factors.update(sequence.challenge_factors.keys())
        
        return {
            "domains": sorted(list(domains)),
            "features": sorted(list(features)),
            "tools": sorted(list(tools)),
            "challenge_factors": sorted(list(challenge_factors))
        }
        
    async def _compute_capability_coverage(self, sequences: List[ActionSequence]) -> Dict[str, Dict[str, float]]:
        """Compute capability coverage metrics for the dataset"""
        # Extract all capabilities from features
        capabilities = set()
        for sequence in sequences:
            capabilities.update(sequence.features)
        
        # Compute coverage by domain
        domains = set(seq.domain for seq in sequences)
        coverage = {}
        
        for domain in domains:
            domain_sequences = [seq for seq in sequences if seq.domain == domain]
            domain_coverage = {}
            
            for capability in capabilities:
                # Count sequences with this capability
                cap_sequences = [seq for seq in domain_sequences if capability in seq.features]
                if not domain_sequences:
                    coverage_pct = 0.0
                else:
                    coverage_pct = len(cap_sequences) / len(domain_sequences)
                
                # Count variety of difficulty levels
                difficulties = set(round(seq.difficulty * 10) / 10 for seq in cap_sequences)
                difficulty_coverage = len(difficulties) / 10  # 10 possible difficulty levels
                
                domain_coverage[capability] = {
                    "sequence_coverage": coverage_pct,
                    "difficulty_coverage": difficulty_coverage,
                    "count": len(cap_sequences)
                }
            
            coverage[domain] = domain_coverage
        
        return coverage
        
    async def _validate_dataset(self, dataset: TrainingDataset) -> Dict[str, float]:
        """Validate dataset quality and compute metrics"""
        sequences = dataset.sequences
        
        if not sequences:
            return {"error": "Empty dataset"}
        
        # Calculate diversity metrics
        domains = set(seq.domain for seq in sequences)
        domain_entropy = len(domains) / len(sequences)
        
        features = set()
        for seq in sequences:
            features.update(seq.features)
        feature_coverage = len(features) / (len(sequences) * 0.1)  # Normalized by sequence count
        
        difficulties = [seq.difficulty for seq in sequences]
        avg_difficulty = sum(difficulties) / len(difficulties)
        difficulty_std = (sum((d - avg_difficulty) ** 2 for d in difficulties) / len(difficulties)) ** 0.5
        
        # Quality metrics
        quality_scores = [seq.quality_score for seq in sequences if seq.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Curriculum coverage
        curriculum_coverage = 0.0
        if dataset.curriculum:
            total_stages = len(dataset.curriculum.stages)
            covered_stages = len(set(stage["level"] for stage in dataset.curriculum.stages))
            curriculum_coverage = covered_stages / total_stages
        
        # Supervised example coverage
        supervised_coverage = len(dataset.supervised_examples) / len(sequences) if sequences else 0.0
        
        return {
            "domain_entropy": domain_entropy,
            "feature_coverage": feature_coverage,
            "avg_difficulty": avg_difficulty,
            "difficulty_std": difficulty_std,
            "avg_quality": avg_quality,
            "curriculum_coverage": curriculum_coverage,
            "supervised_coverage": supervised_coverage,
            "total_sequences": len(sequences),
            "total_features": len(features),
            "total_domains": len(domains)
        }
    
    async def _convert_to_sequences(self, interactions: List[AgentInteraction]) -> List[ActionSequence]:
        """Converts interactions to normalized action sequences"""
        sequences = []
        
        for interaction in interactions:
            if interaction.completion_status != "completed":
                continue
                
            # Extract domain from context or tags
            domain = interaction.context_info.get("domain", "general")
            
            # Extract features
            features = []
            if len(interaction.agent_actions) > 5:
                features.append("multi_step")
            
            for action in interaction.agent_actions:
                if action.action_type == "api_call":
                    features.append("api_integration")
                elif action.action_type == "database_query":
                    features.append("database")
                elif action.action_type == "code_generation":
                    features.append("code_gen")
                elif action.action_type == "tool_call":
                    features.append("tool_usage")
                    # Extract specific tool features
                    tool_name = action.description.lower()
                    if "calculator" in tool_name:
                        features.append("math")
                    elif "search" in tool_name or "lookup" in tool_name:
                        features.append("search")
                    elif "database" in tool_name or "query" in tool_name:
                        features.append("database")
                    # Add tool-specific feature
                    features.append(f"tool:{tool_name}")
            
            features = list(set(features))  # Deduplicate
            
            # Create normalized actions
            normalized_actions = []
            for action in interaction.agent_actions:
                normalized_action = {
                    "type": action.action_type,
                    "name": action.description,
                    "params": action.parameters,
                    "result": self._normalize_result(action.result),
                    "execution_time_ms": action.execution_time_ms
                }
                normalized_actions.append(normalized_action)
            
            # Calculate difficulty (advanced heuristic)
            base_difficulty = 0.3
            complexity_factor = len(normalized_actions) * 0.08  # More steps = higher difficulty
            tool_diversity_factor = len(set([a["type"] for a in normalized_actions])) * 0.05
            error_recovery_factor = 0.0
            
            # Check for error recovery patterns (error followed by successful action)
            for i in range(len(normalized_actions) - 1):
                if "error" in str(normalized_actions[i].get("result", "")).lower():
                    error_recovery_factor += 0.1  # Bonus for error recovery
            
            # Analyze linguistic complexity of the input
            linguistic_complexity = 0.0
            words = interaction.human_input.split()
            if len(words) > 50:  # Long inputs are more complex
                linguistic_complexity += 0.1
            if len(set(words)) / max(1, len(words)) > 0.8:  # High lexical diversity
                linguistic_complexity += 0.1
            
            # Multiple intents detection
            intent_count = 1  # Default assumption
            if ";" in interaction.human_input or "and" in interaction.human_input.lower():
                # Simple heuristic for multiple intents
                intent_count += interaction.human_input.lower().count(" and ") + interaction.human_input.count(";")
                linguistic_complexity += min(0.3, intent_count * 0.1)  # Cap at 0.3
            
            # Calculate final difficulty
            difficulty = min(1.0, base_difficulty + complexity_factor + tool_diversity_factor + 
                           error_recovery_factor + linguistic_complexity)
            
            # Generate challenge factors assessment
            challenge_factors = {
                "action_complexity": min(1.0, len(normalized_actions) / 10),
                "tool_diversity": min(1.0, len(set([a["type"] for a in normalized_actions])) / 5),
                "linguistic_complexity": linguistic_complexity,
                "error_recovery": error_recovery_factor > 0,
                "multi_intent": intent_count > 1
            }
            
            # Adaptive quality scoring
            quality_score = None
            if interaction.human_feedback:
                # If we have explicit feedback, use it
                if "rating" in interaction.human_feedback:
                    quality_score = float(interaction.human_feedback["rating"]) / 5.0  # Normalize to 0-1
            else:
                # Heuristic quality estimation based on response time and errors
                response_time_factor = 0.0
                if interaction.time_to_completion_ms:
                    # Faster responses (relative to complexity) indicate higher quality
                    expected_time = 1000 * len(normalized_actions)  # Simple expectation model
                    time_ratio = expected_time / max(1, interaction.time_to_completion_ms)
                    response_time_factor = min(0.3, time_ratio * 0.3)  # Cap at 0.3
                
                error_factor = 0.0
                for action in normalized_actions:
                    if action.get("error") or "error" in str(action.get("result", "")).lower():
                        error_factor -= 0.1  # Penalize errors
                
                # Simple quality score heuristic
                quality_score = max(0.0, min(1.0, 0.7 + response_time_factor + error_factor))
            
            # Check for external context dependencies
            requires_external_context = False
            if any(kw in interaction.human_input.lower() for kw in ["previous", "before", "earlier", "last time"]):
                requires_external_context = True
            
            # Generate state tracking information
            state_tracking = {}
            if "database" in features or "persistence" in features:
                # Extract potential state changes from actions and results
                state_changes = []
                for action in normalized_actions:
                    if "create" in action["name"].lower() or "update" in action["name"].lower():
                        state_changes.append({
                            "operation": "write",
                            "target": action["name"].split("_")[-1] if "_" in action["name"] else "unknown",
                            "params": action["params"]
                        })
                state_tracking["changes"] = state_changes
            
            # Create enhanced action sequence
            sequence = ActionSequence(
                input=interaction.human_input,
                actions=normalized_actions,
                expected_output=interaction.final_response,
                difficulty=difficulty,
                domain=domain,
                features=features,
                requires_external_context=requires_external_context,
                challenge_factors=challenge_factors,
                quality_score=quality_score,
                state_tracking=state_tracking
            )
            
            # Add semantic parsing results if available
            if interaction.semantic_parser_results:
                sequence.annotations["semantic_parse"] = interaction.semantic_parser_results
            
            # Add session reference if this is part of a session
            if interaction.session_id:
                sequence.interaction_dependencies = [interaction.session_id]
            
            sequences.append(sequence)
            
            # Apply domain-specific handlers for specialized processing
            if domain in self.domain_specific_handlers:
                handler = self.domain_specific_handlers[domain]
                try:
                    enhanced_sequence = await handler(sequence, interaction)
                    if enhanced_sequence:
                        # Replace with enhanced version
                        sequences[-1] = enhanced_sequence
                except Exception as e:
                    print(f"Error in domain handler for {domain}: {e}")
        
        # Apply quality filtering
        filtered_sequences = []
        for sequence in sequences:
            keep = True
            # Apply all registered quality filters
            for filter_name, filter_func in self.quality_filters.items():
                try:
                    if not filter_func(sequence):
                        keep = False
                        break
                except Exception as e:
                    print(f"Error in quality filter {filter_name}: {e}")
            
            if keep:
                filtered_sequences.append(sequence)
        
        return filtered_sequences
    
    def _normalize_result(self, result: Any) -> Any:
        """Normalizes action results for consistency"""
        if result is None:
            return None
            
        if isinstance(result, (str, int, float, bool)):
            return result
            
        if isinstance(result, (list, dict)):
            # Remove large binary data, truncate long strings
            if isinstance(result, dict):
                normalized = {}
                for k, v in result.items():
                    if isinstance(v, str) and len(v) > 1000:
                        normalized[k] = v[:1000] + "..."
                    elif isinstance(v, bytes):
                        normalized[k] = f"<binary data, {len(v)} bytes>"
                    else:
                        normalized[k] = self._normalize_result(v)
                return normalized
            else:
                return [self._normalize_result(item) for item in result]
        
        # For other types, convert to string
        return str(result)
    
    async def _compute_dataset_statistics(self, sequences: List[ActionSequence]) -> Dict[str, Any]:
        """Computes statistics for a dataset"""
        if not sequences:
            return {}
            
        # Count by domain
        domains = {}
        for seq in sequences:
            if seq.domain not in domains:
                domains[seq.domain] = 0
            domains[seq.domain] += 1
        
        # Count by action type
        action_types = {}
        for seq in sequences:
            for action in seq.actions:
                action_type = action["type"]
                if action_type not in action_types:
                    action_types[action_type] = 0
                action_types[action_type] += 1
        
        # Count by feature
        features = {}
        for seq in sequences:
            for feature in seq.features:
                if feature not in features:
                    features[feature] = 0
                features[feature] += 1
        
        # Average sequence length
        avg_length = sum(len(seq.actions) for seq in sequences) / len(sequences)
        
        # Average difficulty
        avg_difficulty = sum(seq.difficulty for seq in sequences) / len(sequences)
        
        return {
            "total_sequences": len(sequences),
            "domains": domains,
            "action_types": action_types,
            "features": features,
            "avg_sequence_length": avg_length,
            "avg_difficulty": avg_difficulty
        }
    
    async def export_dataset(self, dataset_id: str, format: str = "jsonl") -> str:
        """Exports a dataset to the specified format"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
            
        dataset = self.datasets[dataset_id]
        
        if format == "jsonl":
            lines = []
            for sequence in dataset.sequences:
                lines.append(json.dumps(sequence.model_dump()))
            return "\n".join(lines)
        elif format == "json":
            return json.dumps(dataset.model_dump(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def register_action_template(self, action_type: str, schema: Dict[str, Any]):
        """Registers a new action template for validation"""
        self.action_templates[action_type] = schema
    
    async def validate_action(self, action_type: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates action parameters against a template"""
        if action_type not in self.action_templates:
            return False, f"Unknown action type: {action_type}"
            
        schema = self.action_templates[action_type]
        
        # Check required parameters
        for param in schema.get("required", []):
            if param not in parameters:
                return False, f"Missing required parameter: {param}"
        
        # Check parameter types
        for param, value in parameters.items():
            if param in schema.get("properties", {}):
                param_schema = schema["properties"][param]
                param_type = param_schema.get("type")
                
                if param_type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param} must be a string"
                elif param_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param} must be a number"
                elif param_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param} must be a boolean"
                elif param_type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param} must be an array"
                elif param_type == "object" and not isinstance(value, dict):
                    return False, f"Parameter {param} must be an object"
        
        return True, None
    
    async def add_interaction_observer(self, observer: Callable):
        """Adds an observer function for interactions"""
        self.interaction_observers.append(observer)
        
    # ----------------- NEURAL INTEGRATION METHODS -----------------
    
    async def create_neural_text_analysis(self, text: str, analysis_types: List[str] = ["intent", "complexity"]) -> Dict[str, Any]:
        """Performs neural-powered analysis on text input
        
        Args:
            text: The text to analyze
            analysis_types: Types of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        adapter = await self.neural_env.get_adapter()
        results = {}
        
        for analysis_type in analysis_types:
            try:
                analysis = await adapter.analyze_text(text, analysis_type)
                results[analysis_type] = analysis
            except Exception as e:
                results[f"{analysis_type}_error"] = str(e)
                
        return results
    
    async def find_similar_examples(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds similar training examples using vector similarity
        
        Args:
            query: The query text to find similar examples for
            top_k: Number of results to return
            
        Returns:
            List of similar examples with similarity scores
        """
        # Get embedding for query
        adapter = await self.neural_env.get_adapter()
        query_embedding = await adapter.get_embedding(query)
        
        # Search for similar examples
        results = await self.neural_env.search_vectors(query_embedding, top_k)
        
        # Enhance results with full example data
        enhanced_results = []
        for result in results:
            key = result["key"]
            metadata = result["metadata"]
            similarity = result["similarity"]
            
            # Try to retrieve the referenced sequence or example
            sequence = None
            for dataset in self.datasets.values():
                for seq in dataset.sequences:
                    if seq.sequence_id == key:
                        sequence = seq
                        break
                if sequence:
                    break
            
            enhanced_results.append({
                "key": key,
                "similarity": similarity,
                "metadata": metadata,
                "sequence": sequence.model_dump() if sequence else None
            })
            
        return enhanced_results
    
    async def enhance_training_data(self, dataset_id: str, enhancement_types: List[str] = ["paraphrase", "counterfactual"]) -> TrainingDataset:
        """Enhances a training dataset with neural-powered data augmentation
        
        Args:
            dataset_id: ID of the dataset to enhance
            enhancement_types: Types of enhancements to apply
            
        Returns:
            Enhanced dataset
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
            
        dataset = self.datasets[dataset_id]
        adapter = await self.neural_env.get_adapter()
        
        # Create a session for tracking this enhancement process
        session_id = await self.neural_env.start_session(
            str(uuid.uuid4()),
            {"dataset_id": dataset_id, "enhancement_types": enhancement_types}
        )
        
        # Process sequences in batches
        batch_size = 10
        processed_count = 0
        
        for i in range(0, len(dataset.sequences), batch_size):
            batch = dataset.sequences[i:i+batch_size]
            
            # Process each sequence in the batch
            for sequence in batch:
                # Skip sequences that already have variants unless we're forcing regeneration
                if sequence.variants and "force_regeneration" not in enhancement_types:
                    continue
                    
                new_variants = []
                
                # Apply each enhancement type
                for enhancement_type in enhancement_types:
                    if enhancement_type == "paraphrase":
                        # Generate paraphrased version of the input
                        try:
                            paraphrase_prompt = "Please paraphrase this text while preserving its meaning:"
                            paraphrased_input = await adapter.generate_completion(
                                "You are a helpful assistant that paraphrases text while preserving meaning.",
                                paraphrase_prompt,
                                sequence.input
                            )
                            
                            # Create variant sequence with paraphrased input
                            variant = ActionSequence(
                                input=paraphrased_input,
                                actions=sequence.actions.copy(),
                                expected_output=sequence.expected_output,
                                difficulty=sequence.difficulty,
                                domain=sequence.domain,
                                features=sequence.features + ["paraphrased"],
                                requires_external_context=sequence.requires_external_context,
                                challenge_factors=sequence.challenge_factors.copy(),
                                quality_score=sequence.quality_score
                            )
                            new_variants.append(variant)
                        except Exception as e:
                            print(f"Error generating paraphrase for {sequence.sequence_id}: {e}")
                            
                    elif enhancement_type == "counterfactual":
                        # Generate challenging counterfactual variant
                        try:
                            if sequence.domain in self.counterfactual_generators:
                                generator = self.counterfactual_generators[sequence.domain]
                                counterfactual = await generator(sequence)
                                if counterfactual:
                                    new_variants.append(counterfactual)
                        except Exception as e:
                            print(f"Error generating counterfactual for {sequence.sequence_id}: {e}")
                            
                    elif enhancement_type == "difficulty_adjustment":
                        # Generate easier or harder variants
                        try:
                            # For now, only create harder variants for sequences below 0.7 difficulty
                            if sequence.difficulty < 0.7:
                                harder_prompt = "Make this request more challenging by adding complexity or constraints:"
                                harder_input = await adapter.generate_completion(
                                    "You are a helpful assistant that makes requests more challenging.",
                                    harder_prompt,
                                    sequence.input
                                )
                                
                                harder_variant = ActionSequence(
                                    input=harder_input,
                                    actions=sequence.actions.copy(),  # Same actions for now
                                    expected_output=sequence.expected_output,
                                    difficulty=min(1.0, sequence.difficulty + 0.2),
                                    domain=sequence.domain,
                                    features=sequence.features + ["increased_difficulty"],
                                    requires_external_context=sequence.requires_external_context,
                                    challenge_factors=sequence.challenge_factors.copy(),
                                    quality_score=sequence.quality_score
                                )
                                new_variants.append(harder_variant)
                        except Exception as e:
                            print(f"Error adjusting difficulty for {sequence.sequence_id}: {e}")
                
                # Add new variants to the sequence and dataset
                for variant in new_variants:
                    # Link variants to the original sequence
                    variant.interaction_dependencies.append(sequence.sequence_id)
                    sequence.variants.append(variant.sequence_id)
                    # Add variant to dataset
                    dataset.sequences.append(variant)
                
                processed_count += 1
                
            # Update session metrics
            await self.neural_env.update_session(session_id, {
                "processed_items": batch_size,
                "new_variants": sum(len(sequence.variants) for sequence in batch),
                "current_dataset_size": len(dataset.sequences)
            })
        
        # End session
        await self.neural_env.end_session(session_id)
        
        # Update dataset statistics to reflect the new variants
        dataset.statistics = await self._compute_dataset_statistics(dataset.sequences)
        
        return dataset
        
    async def run_benchmark_evaluation(self, dataset_id: str, benchmark_id: str) -> Dict[str, Any]:
        """Runs a benchmark evaluation on a dataset
        
        Args:
            dataset_id: ID of the dataset to evaluate
            benchmark_id: ID of the benchmark to use
            
        Returns:
            Evaluation metrics
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
            
        if benchmark_id not in self.evaluation_benchmarks:
            raise ValueError(f"Benchmark with ID {benchmark_id} not found")
            
        dataset = self.datasets[dataset_id]
        benchmark = self.evaluation_benchmarks[benchmark_id]
        
        # Start evaluation session
        session_id = await self.neural_env.start_session(
            str(uuid.uuid4()),
            {"dataset_id": dataset_id, "benchmark_id": benchmark_id}
        )
        
        # Run benchmark evaluation
        metrics = {}
        
        # Extract test set according to the dataset split
        test_ratio = dataset.split.get("test", 0.1)
        test_size = int(len(dataset.sequences) * test_ratio)
        test_sequences = dataset.sequences[-test_size:] if test_size > 0 else []
        
        if not test_sequences:
            await self.neural_env.end_session(session_id, "failed")
            return {"error": "No test sequences available"}
        
        # Get neural adapter
        adapter = await self.neural_env.get_adapter()
        
        # Performance metrics
        correct = 0
        total = 0
        response_times = []
        
        # Run tests
        for sequence in test_sequences:
            start_time = datetime.datetime.now()
            
            try:
                # Generate response using the neural adapter
                response = await adapter.generate_completion(
                    benchmark.get("system_prompt", "You are a helpful assistant."),
                    benchmark.get("instruction", "Respond to the following:"),
                    sequence.input
                )
                
                # Record response time
                end_time = datetime.datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000
                response_times.append(response_time)
                
                # Basic correctness check (in a real system, this would be more sophisticated)
                expected = sequence.expected_output or ""
                correctness_score = self._calculate_similarity(response, expected)
                
                if correctness_score > benchmark.get("threshold", 0.7):
                    correct += 1
                
                total += 1
                
                # Update session metrics after every 10 tests
                if total % 10 == 0:
                    await self.neural_env.update_session(session_id, {
                        "processed_items": 10,
                        "current_accuracy": correct / total,
                        "avg_response_time": sum(response_times) / len(response_times)
                    })
                
            except Exception as e:
                print(f"Error evaluating sequence {sequence.sequence_id}: {e}")
        
        # Calculate overall metrics
        if total > 0:
            metrics["accuracy"] = correct / total
            metrics["avg_response_time_ms"] = sum(response_times) / len(response_times) if response_times else 0
            metrics["test_sequences"] = total
            metrics["benchmark"] = benchmark_id
            metrics["dataset"] = dataset_id
            
            # End session with success
            await self.neural_env.end_session(session_id, "completed")
        else:
            metrics["error"] = "No sequences were evaluated"
            await self.neural_env.end_session(session_id, "failed")
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using a simple algorithm
        
        In a real implementation, this would use more sophisticated methods.
        """
        # Simple word overlap measure for demonstration
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = words1.intersection(words2)
        similarity = len(overlap) / max(len(words1), len(words2))
        
        return similarity
        
    async def register_model(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Registers a model in the model registry
        
        Args:
            model_id: Unique identifier for the model
            model_info: Model metadata and configuration
            
        Returns:
            Updated model information
        """
        # Add registration timestamp
        model_info["registered_at"] = datetime.datetime.now().isoformat()
        
        # Add default fields if not present
        if "type" not in model_info:
            model_info["type"] = "unknown"
        if "version" not in model_info:
            model_info["version"] = "1.0.0"
        if "status" not in model_info:
            model_info["status"] = "registered"
            
        # Store in registry
        self.model_registry[model_id] = model_info
        
        return model_info
        
    async def create_simulation_environment(self, env_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a simulation environment for testing agent behaviors
        
        Args:
            env_id: Unique identifier for the environment
            config: Environment configuration
            
        Returns:
            Environment configuration
        """
        # Add creation timestamp
        config["created_at"] = datetime.datetime.now().isoformat()
        
        # Add default settings if not provided
        if "max_turns" not in config:
            config["max_turns"] = 10
        if "tools" not in config:
            config["tools"] = []
        if "scenarios" not in config:
            config["scenarios"] = []
            
        # Store environment configuration
        self.simulation_environments[env_id] = config
        
        return config
