#!/usr/bin/env python3

import os
import sys
import re
import json
import importlib.util
import textwrap
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from pathlib import Path

# Domain-Driven Design Templates
DDD_ENTITY_TEMPLATE = """from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class {entity_name}:
    """"{entity_name} entity in the {domain_name} domain"""
{fields}
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update(self, **kwargs):
        """Update entity attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
"""

DDD_REPOSITORY_TEMPLATE = """from typing import List, Optional, Dict, Any
from .entities import {entity_name}

class {entity_name}Repository:
    """Repository for {entity_name} entities"""
    
    def __init__(self):
        self._entities: Dict[str, {entity_name}] = {{}}
    
    async def save(self, entity: {entity_name}) -> {entity_name}:
        """Save an entity"""
        self._entities[entity.id] = entity
        return entity
    
    async def find_by_id(self, id: str) -> Optional[{entity_name}]:
        """Find an entity by ID"""
        return self._entities.get(id)
    
    async def find_all(self) -> List[{entity_name}]:
        """Find all entities"""
        return list(self._entities.values())
    
    async def delete(self, id: str) -> bool:
        """Delete an entity"""
        if id in self._entities:
            del self._entities[id]
            return True
        return False
"""

DDD_SERVICE_TEMPLATE = """from typing import List, Optional, Dict, Any
from .repositories import {entity_name}Repository
from .entities import {entity_name}
from .value_objects import {value_objects_import}
from .events import {event_bus_import}

class {entity_name}Service:
    """Service for {entity_name} operations"""
    
    def __init__(
        self, 
        repository: {entity_name}Repository,
        event_bus: {event_bus_class}
    ):
        self.repository = repository
        self.event_bus = event_bus
    
    async def create(self, data: Dict[str, Any]) -> {entity_name}:
        """Create a new {entity_name}"""
        entity = {entity_name}(**data)
        saved = await self.repository.save(entity)
        
        # Publish domain event
        await self.event_bus.publish(f"{entity_name}Created", {{"id": saved.id}})
        
        return saved
    
    async def get(self, id: str) -> Optional[{entity_name}]:
        """Get an {entity_name} by ID"""
        return await self.repository.find_by_id(id)
    
    async def list(self) -> List[{entity_name}]:
        """List all {entity_name}s"""
        return await self.repository.find_all()
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[{entity_name}]:
        """Update an {entity_name}"""
        entity = await self.repository.find_by_id(id)
        if not entity:
            return None
        
        entity.update(**data)
        saved = await self.repository.save(entity)
        
        # Publish domain event
        await self.event_bus.publish(f"{entity_name}Updated", {{"id": saved.id}})
        
        return saved
    
    async def delete(self, id: str) -> bool:
        """Delete an {entity_name}"""
        success = await self.repository.delete(id)
        
        if success:
            # Publish domain event
            await self.event_bus.publish(f"{entity_name}Deleted", {{"id": id}})
        
        return success
"""

DDD_VALUE_OBJECT_TEMPLATE = """from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class {name}:
    """"{name} value object in the {domain_name} domain"""
{fields}
    
    def __post_init__(self):
        """Validate the value object"""
        # Add validation logic here
        pass
"""

DDD_EVENT_BUS_TEMPLATE = """from typing import Dict, Any, Callable, List, Awaitable
import asyncio

class EventBus:
    """Event bus for domain events"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {{}}
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to an event"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event"""
        if event_type in self.subscribers:
            await asyncio.gather(
                *[handler(event_data) for handler in self.subscribers[event_type]]
            )
"""

# Microservice Templates
MICROSERVICE_MAIN_TEMPLATE = """#!/usr/bin/env python3

import os
import asyncio
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers
{router_imports}

# Import from shared modules
from {service_name_lower}.config import settings
from {service_name_lower}.database import init_db, close_db
from {service_name_lower}.middleware import add_middleware

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    await init_db()
    
    yield
    
    # Shutdown
    await close_db()

# Create application
app = FastAPI(
    title="{service_name} Microservice",
    description="{service_description}",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
add_middleware(app)

# Add routers
{router_registrations}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
"""

MICROSERVICE_CONFIG_TEMPLATE = """#!/usr/bin/env python3

import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # General
    APP_NAME: str = "{service_name}"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ENV: str = os.getenv("ENV", "development")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "{port}"))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:9996",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:9996",
    ]
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development_secret_key")
    TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
"""

MICROSERVICE_DATABASE_TEMPLATE = """#!/usr/bin/env python3

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from {service_name_lower}.config import settings

# Create async engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create declarative base for models
Base = declarative_base()

async def init_db():
    """Initialize database"""
    if settings.ENV == "development":
        async with engine.begin() as conn:
            # Create tables for development
            await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections"""
    await engine.dispose()

@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
"""

# GraphQL Schema Templates
GRAPHQL_SCHEMA_TEMPLATE = """# GraphQL Schema for {model_name}
type {model_name} {{
  id: ID!
  name: String!
{fields}
  createdAt: String!
  updatedAt: String!
}}

input {model_name}Input {{
  name: String!
{input_fields}
}}

type Query {{
  get{model_name}(id: ID!): {model_name}
  list{model_name}s: [{model_name}!]!
  search{model_name}s(query: String): [{model_name}!]!
}}

type Mutation {{
  create{model_name}(input: {model_name}Input!): {model_name}!
  update{model_name}(id: ID!, input: {model_name}Input!): {model_name}!
  delete{model_name}(id: ID!): Boolean!
}}
"""

GRAPHQL_RESOLVER_TEMPLATE = """# GraphQL Resolvers for {model_name}
from typing import List, Dict, Any, Optional
from backend.models.{model_file} import {model_name}
from backend.services.{service_file} import {service_name}

service = {service_name}()

async def get_{model_name_lower}(parent, info, id):
    """Get a {model_name} by ID"""
    return await service.get(id)

async def list_{model_name_lower}s(parent, info):
    """List all {model_name}s"""
    return await service.list()

async def search_{model_name_lower}s(parent, info, query):
    """Search for {model_name}s"""
    return await service.search(query)

async def create_{model_name_lower}(parent, info, input):
    """Create a new {model_name}"""
    return await service.create(input)

async def update_{model_name_lower}(parent, info, id, input):
    """Update a {model_name}"""
    return await service.update(id, input)

async def delete_{model_name_lower}(parent, info, id):
    """Delete a {model_name}"""
    return await service.delete(id)

resolvers = {{
    "Query": {{
        f"get{model_name}": get_{model_name_lower},
        f"list{model_name}s": list_{model_name_lower}s,
        f"search{model_name}s": search_{model_name_lower}s,
    }},
    "Mutation": {{
        f"create{model_name}": create_{model_name_lower},
        f"update{model_name}": update_{model_name_lower},
        f"delete{model_name}": delete_{model_name_lower},
    }}
}}
"""

def generate_domain_driven_design_scaffold(name: str, entities: List[str], output_dir: str = None) -> Dict[str, List[str]]:
    """Generate a Domain-Driven Design scaffold
    
    Args:
        name: Name of the domain
        entities: List of entities with optional field definitions (e.g. "User:name,email,age")
        output_dir: Directory to output the scaffold
        
    Returns:
        Dictionary of created files by category
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    domain_name = name.capitalize()
    domain_dir = os.path.join(output_dir, name.lower())
    
    # Create directory structure
    os.makedirs(os.path.join(domain_dir, "entities"), exist_ok=True)
    os.makedirs(os.path.join(domain_dir, "repositories"), exist_ok=True)
    os.makedirs(os.path.join(domain_dir, "services"), exist_ok=True)
    os.makedirs(os.path.join(domain_dir, "value_objects"), exist_ok=True)
    os.makedirs(os.path.join(domain_dir, "events"), exist_ok=True)
    os.makedirs(os.path.join(domain_dir, "aggregates"), exist_ok=True)
    
    created_files = {
        "entities": [],
        "repositories": [],
        "services": [],
        "value_objects": [],
        "events": [],
        "aggregates": [],
        "init_files": []
    }
    
    # Create __init__ files
    for dirpath, dirnames, filenames in os.walk(domain_dir):
        init_file = os.path.join(dirpath, "__init__.py")
        with open(init_file, 'w') as f:
            f.write(f'"""{os.path.basename(dirpath)} module for {domain_name} domain"""\n')
        created_files["init_files"].append(init_file)
    
    # Parse entities and create files
    parsed_entities = []
    for entity_def in entities:
        parts = entity_def.split(':')
        entity_name = parts[0].strip()
        
        fields = []
        if len(parts) > 1 and parts[1]:
            field_defs = parts[1].split(',')
            for field_def in field_defs:
                field_name = field_def.strip()
                field_type = "str"  # Default type
                
                if ":" in field_name:
                    field_name, field_type = field_name.split(':')
                
                fields.append((field_name.strip(), field_type.strip()))
        
        parsed_entities.append({
            "name": entity_name,
            "fields": fields
        })
    
    # Create event bus
    event_bus_path = os.path.join(domain_dir, "events", "event_bus.py")
    with open(event_bus_path, 'w') as f:
        f.write(DDD_EVENT_BUS_TEMPLATE)
    created_files["events"].append(event_bus_path)
    
    # Create entities, repositories, and services
    for entity in parsed_entities:
        entity_name = entity["name"].capitalize()
        
        # Format fields for entity
        entity_fields = []
        for field_name, field_type in entity["fields"]:
            entity_fields.append(f"    {field_name}: {field_type}")
        
        entity_fields_str = "\n".join(entity_fields)
        if entity_fields_str:
            entity_fields_str += "\n"
        
        # Create entity
        entity_path = os.path.join(domain_dir, "entities", f"{entity_name.lower()}.py")
        with open(entity_path, 'w') as f:
            f.write(DDD_ENTITY_TEMPLATE.format(
                entity_name=entity_name,
                domain_name=domain_name,
                fields=entity_fields_str
            ))
        created_files["entities"].append(entity_path)
        
        # Create repository
        repo_path = os.path.join(domain_dir, "repositories", f"{entity_name.lower()}_repository.py")
        with open(repo_path, 'w') as f:
            f.write(DDD_REPOSITORY_TEMPLATE.format(
                entity_name=entity_name
            ))
        created_files["repositories"].append(repo_path)
        
        # Create service
        service_path = os.path.join(domain_dir, "services", f"{entity_name.lower()}_service.py")
        with open(service_path, 'w') as f:
            f.write(DDD_SERVICE_TEMPLATE.format(
                entity_name=entity_name,
                value_objects_import="*",  # Import all value objects
                event_bus_import="event_bus",
                event_bus_class="event_bus.EventBus"
            ))
        created_files["services"].append(service_path)
        
        # Create value objects based on field types
        custom_types = [field_type for _, field_type in entity["fields"] 
                       if field_type not in ["str", "int", "float", "bool", "datetime", "dict", "list"]]
        
        for type_name in custom_types:
            vo_path = os.path.join(domain_dir, "value_objects", f"{type_name.lower()}.py")
            if not os.path.exists(vo_path):
                with open(vo_path, 'w') as f:
                    f.write(DDD_VALUE_OBJECT_TEMPLATE.format(
                        name=type_name,
                        domain_name=domain_name,
                        fields="    value: str"  # Default field
                    ))
                created_files["value_objects"].append(vo_path)
    
    # Create aggregates
    for entity in parsed_entities:
        entity_name = entity["name"].capitalize()
        aggregate_path = os.path.join(domain_dir, "aggregates", f"{entity_name.lower()}_aggregate.py")
        
        with open(aggregate_path, 'w') as f:
            f.write(f"""from typing import Dict, List, Any, Optional
from ..entities.{entity_name.lower()} import {entity_name}
from ..repositories.{entity_name.lower()}_repository import {entity_name}Repository
from ..services.{entity_name.lower()}_service import {entity_name}Service
from ..events.event_bus import EventBus

class {entity_name}Aggregate:
    \"\"\"Aggregate for {entity_name}\"\"\"
    
    def __init__(self):
        self.event_bus = EventBus()
        self.repository = {entity_name}Repository()
        self.service = {entity_name}Service(self.repository, self.event_bus)
""")
        created_files["aggregates"].append(aggregate_path)
    
    return created_files

def generate_microservice_scaffold(name: str, service_type: str, port: int, output_dir: str = None) -> Dict[str, List[str]]:
    """Generate a microservice scaffold
    
    Args:
        name: Name of the microservice
        service_type: Type of microservice ('api', 'worker', 'gateway', 'mcp-server')
        port: Port for the microservice
        output_dir: Directory to output the scaffold
        
    Returns:
        Dictionary of created files by category
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Format service name
    service_name = name.capitalize()
    service_name_lower = name.lower()
    service_dir = os.path.join(output_dir, service_name_lower)
    
    # Create directory structure
    os.makedirs(os.path.join(service_dir, "routers"), exist_ok=True)
    os.makedirs(os.path.join(service_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(service_dir, "services"), exist_ok=True)
    os.makedirs(os.path.join(service_dir, "dependencies"), exist_ok=True)
    os.makedirs(os.path.join(service_dir, "middleware"), exist_ok=True)
    os.makedirs(os.path.join(service_dir, "schemas"), exist_ok=True)
    
    created_files = {
        "core": [],
        "routers": [],
        "models": [],
        "services": [],
        "middleware": [],
        "dependencies": [],
        "schemas": []
    }
    
    # Create config file
    config_path = os.path.join(service_dir, "config.py")
    with open(config_path, 'w') as f:
        f.write(MICROSERVICE_CONFIG_TEMPLATE.format(
            service_name=service_name,
            port=port
        ))
    created_files["core"].append(config_path)
    
    # Create database module
    db_path = os.path.join(service_dir, "database.py")
    with open(db_path, 'w') as f:
        f.write(MICROSERVICE_DATABASE_TEMPLATE.format(
            service_name_lower=service_name_lower
        ))
    created_files["core"].append(db_path)
    
    # Create middleware module
    middleware_path = os.path.join(service_dir, "middleware.py")
    with open(middleware_path, 'w') as f:
        f.write(f"""from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time

class TimingMiddleware(BaseHTTPMiddleware):
    \"\"\"Middleware to measure request processing time\"\"\"
    
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

def add_middleware(app: FastAPI):
    \"\"\"Add custom middleware to the app\"\"\"
    app.add_middleware(TimingMiddleware)
""")
    created_files["middleware"].append(middleware_path)
    
    # Create __init__ files for each directory
    for dirpath, dirnames, filenames in os.walk(service_dir):
        if dirpath != service_dir:  # Skip the main directory
            init_file = os.path.join(dirpath, "__init__.py")
            with open(init_file, 'w') as f:
                f.write(f'"""{os.path.basename(dirpath)} module for {service_name} service"""\n')
            
            if "core" in created_files:
                created_files["core"].append(init_file)
            else:
                created_files["dependencies"].append(init_file)
    
    # Create health check router
    health_router_path = os.path.join(service_dir, "routers", "health.py")
    with open(health_router_path, 'w') as f:
        f.write(f"""from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    \"\"\"Health check endpoint\"\"\"
    return {{"status": "ok", "service": "{service_name}"}}

@router.get("/db")
async def db_health_check(db: AsyncSession = Depends(get_db)):
    \"\"\"Database health check endpoint\"\"\"
    try:
        # Execute a simple query to verify the database connection
        await db.execute("SELECT 1")
        return {{"status": "ok", "database": "connected"}}
    except Exception as e:
        return {{"status": "error", "database": "disconnected", "message": str(e)}}
""")
    created_files["routers"].append(health_router_path)
    
    # Create main.py based on service type
    main_path = os.path.join(service_dir, "main.py")
    
    router_imports = f"from {service_name_lower}.routers import health"
    router_registrations = f"app.include_router(health.router)"
    
    if service_type == "api":
        service_description = "RESTful API Service"
    elif service_type == "worker":
        service_description = "Background Worker Service"
    elif service_type == "gateway":
        service_description = "API Gateway Service"
    elif service_type == "mcp-server":
        service_description = "Model Context Protocol Server"
        router_imports += f"\nfrom {service_name_lower}.mcp import server"
        router_registrations += f"\napp.include_router(server.router)"
    else:
        service_description = "Microservice"
    
    with open(main_path, 'w') as f:
        f.write(MICROSERVICE_MAIN_TEMPLATE.format(
            service_name=service_name,
            service_name_lower=service_name_lower,
            service_description=service_description,
            router_imports=router_imports,
            router_registrations=router_registrations
        ))
    created_files["core"].append(main_path)
    
    # Create Dockerfile
    dockerfile_path = os.path.join(service_dir, "Dockerfile")
    with open(dockerfile_path, 'w') as f:
        f.write(f"""FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT={port}
ENV ENV=production

# Expose port
EXPOSE {port}

# Start the application
CMD ["python", "-m", "{service_name_lower}.main"]
""")
    created_files["core"].append(dockerfile_path)
    
    # Create docker-compose.yml
    compose_path = os.path.join(service_dir, "docker-compose.yml")
    with open(compose_path, 'w') as f:
        f.write(f"""version: '3'

services:
  {service_name_lower}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - PORT={port}
      - DATABASE_URL=postgresql://${{DB_USER}}:${{DB_PASSWORD}}@db:5432/${{DB_NAME}}
      - SECRET_KEY=${{SECRET_KEY}}
      - ENV=production
    depends_on:
      - db
    volumes:
      - .:/app

  db:
    image: postgres:14-alpine
    environment:
      - POSTGRES_USER=${{DB_USER}}
      - POSTGRES_PASSWORD=${{DB_PASSWORD}}
      - POSTGRES_DB=${{DB_NAME}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
""")
    created_files["core"].append(compose_path)
    
    # Create requirements.txt
    req_path = os.path.join(service_dir, "requirements.txt")
    with open(req_path, 'w') as f:
        f.write(f"""fastapi>=0.103.1
uvicorn>=0.23.2
pydantic>=2.4.2
pydantic-settings>=2.0.3
sqlalchemy>=2.0.22
alembic>=1.12.0
asyncpg>=0.28.0
psycopg2-binary>=2.9.9
python-multipart>=0.0.6
python-dotenv>=1.0.0
httpx>=0.25.0
""")
    created_files["core"].append(req_path)
    
    # If it's an MCP server, create MCP module
    if service_type == "mcp-server":
        os.makedirs(os.path.join(service_dir, "mcp"), exist_ok=True)
        
        mcp_init_path = os.path.join(service_dir, "mcp", "__init__.py")
        with open(mcp_init_path, 'w') as f:
            f.write(f'"""MCP module for {service_name} service"""\n')
        created_files["core"].append(mcp_init_path)
        
        mcp_server_path = os.path.join(service_dir, "mcp", "server.py")
        with open(mcp_server_path, 'w') as f:
            f.write(f"""from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/mcp", tags=["mcp"])

@router.post("/sample")
async def sample():
    \"\"\"Sample endpoint for MCP\"\"\"
    return {{
        "model": "{service_name}",
        "role": "assistant",
        "content": {{
            "type": "text",
            "text": "This is a sample response from the {service_name} MCP server."
        }}
    }}

@router.post("/tools/{{tool_id}}")
async def invoke_tool(tool_id: str, params: Dict[str, Any]):
    \"\"\"Invoke a tool\"\"\"
    # Implement tool invocation logic
    return {{
        "result": f"Tool {{tool_id}} invoked with params {{params}}"
    }}

@router.get("/resources/{{resource_id}}")
async def get_resource(resource_id: str):
    \"\"\"Get a resource\"\"\"
    # Implement resource retrieval logic
    return {{
        "id": resource_id,
        "data": "Sample resource data"
    }}
""")
        created_files["core"].append(mcp_server_path)
    
    return created_files

def generate_graphql_schemas(models: List[str], output_file: str = None) -> Dict[str, str]:
    """Generate GraphQL schemas from models
    
    Args:
        models: List of model names to generate schemas for
        output_file: Output file for the GraphQL schema
        
    Returns:
        Dictionary of created files
    """
    # Find the models in the project
    model_classes = {}
    model_fields = {}
    
    for model_name in models:
        # Check in backend/models directory
        model_file = os.path.join(os.getcwd(), 'backend', 'models', f"{model_name.lower()}.py")
        if not os.path.exists(model_file):
            print(f"Warning: Model file {model_file} not found")
            continue
        
        # Import the module
        try:
            spec = importlib.util.spec_from_file_location(f"backend.models.{model_name.lower()}", model_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for the model class
                model_attr = model_name.capitalize()
                if hasattr(module, model_attr):
                    model_class = getattr(module, model_attr)
                    model_classes[model_name] = model_class
                    
                    # Extract fields if it's a Pydantic model
                    if hasattr(model_class, "model_fields"):
                        model_fields[model_name] = model_class.model_fields
                    elif hasattr(model_class, "__annotations__"):
                        model_fields[model_name] = model_class.__annotations__
        except Exception as e:
            print(f"Error importing model {model_name}: {str(e)}")
    
    if not model_classes:
        print("No valid models found to generate GraphQL schemas")
        return {}
    
    created_files = {}
    
    # Create GraphQL directory if needed
    graphql_dir = os.path.join(os.getcwd(), 'graphql')
    os.makedirs(graphql_dir, exist_ok=True)
    
    # Create schema for each model
    for model_name, model_class in model_classes.items():
        # Convert model fields to GraphQL fields
        graphql_fields = []
        graphql_input_fields = []
        
        fields = model_fields.get(model_name, {})
        for field_name, field_type in fields.items():
            # Skip id field (added automatically)
            if field_name == "id":
                continue
            
            # Skip created_at and updated_at fields (added automatically)
            if field_name in ["created_at", "updated_at"]:
                continue
            
            # Convert Python types to GraphQL types
            graphql_type = "String"  # Default type
            is_required = True
            
            # Try to get the type as a string
            type_str = str(field_type)
            if "Optional[" in type_str:
                is_required = False
                type_str = type_str.replace("Optional[", "").replace("]", "")
            
            if "str" in type_str:
                graphql_type = "String"
            elif "int" in type_str:
                graphql_type = "Int"
            elif "float" in type_str:
                graphql_type = "Float"
            elif "bool" in type_str:
                graphql_type = "Boolean"
            elif "List[" in type_str:
                graphql_type = "[String]"  # Default to string list
                if "int" in type_str:
                    graphql_type = "[Int]"
                elif "float" in type_str:
                    graphql_type = "[Float]"
                elif "bool" in type_str:
                    graphql_type = "[Boolean]"
            elif "Dict[" in type_str or "dict" in type_str:
                graphql_type = "JSON"  # JSON type for dictionaries
            
            # Add required indicator
            if is_required:
                graphql_type += "!"
            
            # Add field to schema
            graphql_fields.append(f"  {field_name}: {graphql_type}")
            
            # Add to input type if it's not a complex field
            if not ("List[" in type_str or "Dict[" in type_str or "dict" in type_str):
                input_type = graphql_type
                if input_type.endswith("!"):
                    input_type = input_type[:-1]  # Remove required indicator for input fields
                graphql_input_fields.append(f"  {field_name}: {input_type}")
        
        # Create schema file
        schema_content = GRAPHQL_SCHEMA_TEMPLATE.format(
            model_name=model_name.capitalize(),
            fields="\n".join(graphql_fields),
            input_fields="\n".join(graphql_input_fields)
        )
        
        schema_file = output_file or os.path.join(graphql_dir, f"{model_name.lower()}.graphql")
        with open(schema_file, "w") as f:
            f.write(schema_content)
        created_files[schema_file] = schema_content
        
        # Create resolver file
        resolver_content = GRAPHQL_RESOLVER_TEMPLATE.format(
            model_name=model_name.capitalize(),
            model_name_lower=model_name.lower(),
            model_file=model_name.lower(),
            service_name=f"{model_name.capitalize()}Service",
            service_file=f"{model_name.lower()}_service"
        )
        
        resolver_file = os.path.join(graphql_dir, f"{model_name.lower()}_resolvers.py")
        with open(resolver_file, "w") as f:
            f.write(resolver_content)
        created_files[resolver_file] = resolver_content
    
    # Create merged schema if multiple models
    if len(model_classes) > 1 and not output_file:
        merged_schema = """# Combined GraphQL Schema
type Query {
"""
        for model_name in model_classes:
            model_name_cap = model_name.capitalize()
            merged_schema += f"  get{model_name_cap}(id: ID!): {model_name_cap}\n"
            merged_schema += f"  list{model_name_cap}s: [{model_name_cap}!]!\n"
            merged_schema += f"  search{model_name_cap}s(query: String): [{model_name_cap}!]!\n"
        
        merged_schema += "}\n\ntype Mutation {\n"
        
        for model_name in model_classes:
            model_name_cap = model_name.capitalize()
            merged_schema += f"  create{model_name_cap}(input: {model_name_cap}Input!): {model_name_cap}!\n"
            merged_schema += f"  update{model_name_cap}(id: ID!, input: {model_name_cap}Input!): {model_name_cap}!\n"
            merged_schema += f"  delete{model_name_cap}(id: ID!): Boolean!\n"
        
        merged_schema += "}\n"
        
        merged_file = os.path.join(graphql_dir, "schema.graphql")
        with open(merged_file, "w") as f:
            f.write(merged_schema)
        created_files[merged_file] = merged_schema
    
    return created_files