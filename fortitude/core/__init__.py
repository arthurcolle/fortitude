from typing import Dict, Any, Type, List, Optional, Union, Callable
import importlib.util
import sys
import os
import asyncio

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

# Import visualization models
from .visualization import (
    PrettyRenderableModel, 
    ThoughtSegment, 
    ChainOfThought, 
    Step, 
    Plan, 
    DecomposedUserRequest,
    # Agent training data classes
    AgentAction,
    AgentInteraction,
    AgentSession,
    AgentTrainingCorpus,
    ActionSequence,
    TrainingCurriculum,
    SupervisedFinetuningExample,
    TrainingDataset,
    NeuralAdapter,
    TrainingPipeline,
    NeuralTrainingEnvironment,
    AgentTrainingManager
)

class FortitudeApp:
    """Main application class for Fortitude"""
    
    def __init__(self, name: str):
        self.name = name
        self.models: Dict[str, Type[FortitudeBaseModel]] = {}
        self.endpoints: Dict[str, Endpoint] = {}
        self.server = FortitudeServer(name)
        self.training_manager = AgentTrainingManager()
    
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
    
    async def record_agent_interaction(self, human_input: str, context_info: Dict[str, Any] = {}) -> AgentInteraction:
        """Record a new agent interaction for training data collection"""
        return await self.training_manager.record_interaction(human_input, context_info)
    
    async def create_training_corpus(self, name: str, description: str, tags: List[str] = []) -> AgentTrainingCorpus:
        """Create a new training corpus for agent interactions"""
        return await self.training_manager.create_corpus(name, description, tags)
    
    async def generate_training_dataset(self, corpus_ids: List[str], name: str, description: str) -> TrainingDataset:
        """Generate a training dataset from agent interaction corpora"""
        return await self.training_manager.create_training_dataset(corpus_ids, name, description)
    
    async def export_training_data(self, dataset_id: str, format: str = "jsonl") -> str:
        """Export training data in the specified format"""
        return await self.training_manager.export_dataset(dataset_id, format)
    
    # Neural-powered API methods
    
    async def analyze_text(self, text: str, analysis_types: List[str] = ["intent", "complexity"]) -> Dict[str, Any]:
        """Run neural text analysis on input text"""
        return await self.training_manager.create_neural_text_analysis(text, analysis_types)
    
    async def find_similar_examples(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find semantically similar training examples"""
        return await self.training_manager.find_similar_examples(query, top_k)
    
    async def enhance_dataset(self, dataset_id: str, enhancement_types: List[str] = ["paraphrase", "counterfactual"]) -> TrainingDataset:
        """Enhance a dataset with neural data augmentation"""
        return await self.training_manager.enhance_training_data(dataset_id, enhancement_types)
    
    async def run_benchmark(self, dataset_id: str, benchmark_id: str) -> Dict[str, Any]:
        """Run a benchmark evaluation on a dataset"""
        return await self.training_manager.run_benchmark_evaluation(dataset_id, benchmark_id)
    
    async def register_model(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model in the neural model registry"""
        return await self.training_manager.register_model(model_id, model_info)
    
    async def create_simulation(self, env_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simulation environment for testing"""
        return await self.training_manager.create_simulation_environment(env_id, config)
    
    def start(self, ui_port: int = 9996, api_port: int = 9997):
        """Start the UI and API servers"""
        # Start FastAPI backend server
        self.server.start(port=api_port)