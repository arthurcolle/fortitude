from typing import Dict, Any, Type, List
import importlib.util
import sys
import os

# We need to handle relative imports differently since this will be installed as a package
try:
    from ..backend.models import FortitudeBaseModel
    from ..backend.endpoints import Endpoint, CRUDEndpoint
    from ..backend.server import FortitudeServer
except ImportError:
    # When running from a local directory, we need to use a different approach
    backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
    
    # Add backend to path if it's not there already
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    
    # Now we can import from backend
    from models import FortitudeBaseModel
    from endpoints import Endpoint, CRUDEndpoint
    from server import FortitudeServer

class FortitudeApp:
    """Main application class for Fortitude"""
    
    def __init__(self, name: str):
        self.name = name
        self.models: Dict[str, Type[FortitudeBaseModel]] = {}
        self.endpoints: Dict[str, Endpoint] = {}
        self.server = FortitudeServer(name)
    
    def register_model(self, model: Type[FortitudeBaseModel]):
        """Register a model with the application"""
        self.models[model.__name__] = model
        # Register with server
        self.server.register_model(model)
        # Automatically create CRUD endpoint
        self.register_endpoint(CRUDEndpoint(model))
        return model
    
    def register_endpoint(self, endpoint: Endpoint):
        """Register an endpoint with the application"""
        self.endpoints[endpoint.name] = endpoint
        # Register with server
        self.server.register_endpoint(endpoint)
        # Register with external tool registry
        endpoint.register()
        return endpoint
    
    def start(self, ui_port: int = 9996, api_port: int = 9997):
        """Start the UI and API servers"""
        # Start FastAPI backend server
        self.server.start(port=api_port)