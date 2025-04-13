import os
import re
from typing import List, Dict, Any

class ModelGenerator:
    """Generate Pydantic models and CRUD operations"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.models_dir = os.path.join(project_dir, "backend/models")
        self.ui_dir = os.path.join(project_dir, "ui")
    
    def generate_model(self, name: str, fields: List[Dict[str, Any]] = None):
        """Generate a new model with the given name and fields"""
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default fields if none provided
        if fields is None:
            fields = [
                {"name": "name", "type": "str", "required": True},
                {"name": "description", "type": "str", "required": False}
            ]
        
        # Generate model file
        model_file = os.path.join(self.models_dir, f"{name.lower()}.py")
        with open(model_file, 'w') as f:
            f.write(self._generate_model_code(name, fields))
        
        # Generate UI components (optional)
        self._generate_ui_components(name, fields)
        
        return model_file
    
    def _generate_model_code(self, name: str, fields: List[Dict[str, Any]]) -> str:
        """Generate Pydantic model code"""
        pascal_case_name = self._to_pascal_case(name)
        
        code = f"""from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
from . import FortitudeBaseModel

class {pascal_case_name}(FortitudeBaseModel):
    \"\"\"{pascal_case_name} data model\"\"\"
"""
        
        # Add fields
        for field in fields:
            field_name = field["name"]
            field_type = field["type"]
            required = field.get("required", True)
            
            if not required:
                field_type = f"Optional[{field_type}]"
                default = field.get("default", "None")
                code += f"    {field_name}: {field_type} = {default}\n"
            else:
                code += f"    {field_name}: {field_type}\n"
        
        return code
    
    def _generate_ui_components(self, name: str, fields: List[Dict[str, Any]]):
        """Generate UI components for the model"""
        # This would generate specific UI components for the model
        # For example, custom forms, views, etc.
        pass
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert a string to PascalCase"""
        # Remove non-alphanumeric characters
        name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
        # Convert to PascalCase
        return ''.join(word.capitalize() for word in name.split())