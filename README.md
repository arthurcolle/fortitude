# Fortitude Framework

Fortitude is a web framework that enables server-side components defined as Pydantic models, which can be used as input and output schemas for endpoints registered as tools in an external registry.

## Features

- Define data models with Pydantic
- Automatic CRUD API endpoints
- Server-side NextJS UI components
- Tool registry integration
- CLI for project management
- MCP (Model Context Protocol) integration for LLM sampling
- Advanced Rails-like scaffolding for rapid development
- Domain-driven design support
- Microservices architecture
- Database migrations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fortitude.git
cd fortitude

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Create a new project
fort new myproject
cd myproject

# Install dependencies
pip install -r requirements.txt
cd ui && npm install && cd ..

# Generate a model
fort model User

# Start the servers
fort start
```

## Project Structure

```
myproject/
├── ui/                    # NextJS UI server
│   ├── app/               # App router
│   ├── components/        # React components
│   ├── public/            # Static assets
│   └── package.json
├── backend/               # Backend API server
│   ├── models/            # Pydantic models
│   ├── endpoints/         # API endpoints
│   └── mcp/               # MCP integration
├── main.py                # Main application entry point
└── requirements.txt
```

## Creating Models

Models are defined as Pydantic classes that inherit from `FortitudeBaseModel`:

```python
from fortitude.backend.models import FortitudeBaseModel
from pydantic import Field
from typing import Optional

class User(FortitudeBaseModel):
    name: str
    email: str
    age: Optional[int] = None
```

Register the model in your main application:

```python
from backend.models.user import User
app.register_model(User)
```

## Advanced Rails-like Scaffolding

Fortitude offers powerful Rails-like scaffolding capabilities with both standard and advanced options:

### Basic Resource Generation

```bash
# Generate a complete resource scaffold
fort resource Product
```

This creates:
- Pydantic model
- CRUD controller
- Service layer
- UI components (list, form, detail views)
- MCP integrations
- Tests

### Advanced Resource Generation

```bash
# Generate an advanced resource scaffold with relationships, validation, and more
fort resource Product --advanced
```

The advanced scaffold includes:
- Models with relationships, validations, and computed properties
- Full-featured controllers with authentication
- Service layer with caching, transactions, and error handling
- Advanced UI components with filtering, sorting, pagination
- Comprehensive test suite
- Deployment configurations

### Domain-Driven Design

```bash
# Generate a domain-driven design scaffold
fort domain Store User:name,email,age Product:name,price,stock Order
```

This creates a complete domain structure with:
- Core domain models
- Value objects
- Repositories
- Domain services
- Application services
- Aggregates
- Entity relationships

### Microservices

```bash
# Generate microservices
fort microservice auth --type api
fort microservice worker --type worker
fort microservice gateway --type gateway
fort microservice ai-assistant --type mcp-server
```

Generates specialized microservices with:
- Containerization
- API gateways
- Service discovery
- Health checks
- Message queues

### Custom Scaffolds

For maximum flexibility, generate and customize your own scaffolds:

```bash
# Generate a scaffold configuration (standard or advanced)
fort generate Task --advanced --output task_scaffold.py

# Edit the configuration file to customize everything
# Then create the scaffold
fort scaffold task_scaffold.py --advanced
```

## MCP Integration

Fortitude provides comprehensive MCP support with both client and server capabilities:

### Running an MCP Server

```bash
# Start an MCP server
fort mcp-server --name "My MCP Server" --port 8888

# Create a custom sampling handler
fort mcp-handler my_handler --type sampling

# Create a tool handler
fort mcp-handler my_tool --type tool

# Create a resource handler
fort mcp-handler my_resource --type resource
```

### Using MCP with Models

```bash
# Generate an MCP client for a model
fort model-mcp-client User --endpoint http://localhost:8888/mcp

# Generate an MCP server for a model
fort model-mcp-server User --port 8889
```

### Sampling from LLMs

```python
# Sample from LLM using MCP
result = await endpoint.sample_llm(
    prompt="What insights can you provide about this data?",
    system_prompt="You are a data analysis expert",
    max_tokens=1000
)
```

The client will display the sample request and handle user approval through the MCP protocol.

## Accessing the UI

Once the servers are running:

- UI Server: http://localhost:9996
- API Server: http://localhost:9997
- API Documentation: http://localhost:9997/docs
- MCP Server: http://localhost:8888/mcp (when running)

## Registry Integration

Endpoints are automatically registered as tools in the external registry at https://arthurcolle--registry.modal.run, making them available for agents to use.

## License

MIT