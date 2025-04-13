#!/usr/bin/env python3

import argparse
import os
import sys
from .commands import create_new_project, start_servers, generate_model
from .mcp_commands import start_mcp_server, scaffold_mcp_client, scaffold_mcp_handler
from .model_mcp_commands import generate_mcp_model_client, generate_mcp_model_server
from .scaffolding import generate_scaffold_config, create_scaffold_from_config, generate_resource_scaffold
from .advanced_scaffolding import generate_advanced_scaffold_config, create_advanced_scaffold_from_config, generate_advanced_resource_scaffold

def main():
    parser = argparse.ArgumentParser(description='Fortitude CLI (fort)')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create new project command
    new_parser = subparsers.add_parser('new', help='Create a new Fortitude project')
    new_parser.add_argument('name', help='Name of the project')
    
    # Start servers command
    start_parser = subparsers.add_parser('start', help='Start Fortitude servers')
    start_parser.add_argument('--ui-port', type=int, default=9996, help='UI server port (default: 9996)')
    start_parser.add_argument('--api-port', type=int, default=9997, help='API server port (default: 9997)')
    
    # Generate model command
    model_parser = subparsers.add_parser('model', help='Generate Pydantic model and CRUD operations')
    model_parser.add_argument('name', help='Name of the model')
    
    # MCP server command
    mcp_server_parser = subparsers.add_parser('mcp-server', help='Start an MCP server')
    mcp_server_parser.add_argument('--name', default='Fortitude MCP Server', help='Name of the MCP server')
    mcp_server_parser.add_argument('--port', type=int, default=8888, help='Port to run the MCP server on')
    
    # MCP client command
    mcp_client_parser = subparsers.add_parser('mcp-client', help='Scaffold an MCP client')
    mcp_client_parser.add_argument('name', help='Name of the MCP client')
    mcp_client_parser.add_argument('--endpoint', default='http://localhost:8888/mcp', help='MCP endpoint URL')
    
    # MCP handler command
    mcp_handler_parser = subparsers.add_parser('mcp-handler', help='Scaffold an MCP handler')
    mcp_handler_parser.add_argument('name', help='Name of the MCP handler')
    mcp_handler_parser.add_argument('--type', choices=['sampling', 'resource', 'tool'], default='sampling', help='Type of handler to scaffold')
    
    # Model MCP client command
    model_mcp_client_parser = subparsers.add_parser('model-mcp-client', help='Generate an MCP client for a model')
    model_mcp_client_parser.add_argument('model', help='Name of the model')
    model_mcp_client_parser.add_argument('--endpoint', default='http://localhost:8888/mcp', help='MCP endpoint URL')
    
    # Model MCP server command
    model_mcp_server_parser = subparsers.add_parser('model-mcp-server', help='Generate an MCP server for a model')
    model_mcp_server_parser.add_argument('model', help='Name of the model')
    model_mcp_server_parser.add_argument('--port', type=int, default=8888, help='Port to run the MCP server on')
    
    # Scaffold generator command
    scaffold_generator_parser = subparsers.add_parser('generate', help='Generate a scaffold configuration')
    scaffold_generator_parser.add_argument('name', help='Name of the scaffold')
    scaffold_generator_parser.add_argument('--output', help='Output file for the scaffold config')
    scaffold_generator_parser.add_argument('--advanced', action='store_true', help='Generate advanced scaffold config')
    
    # Scaffold command
    scaffold_parser = subparsers.add_parser('scaffold', help='Create a scaffold from a configuration file')
    scaffold_parser.add_argument('config', help='Path to the scaffold config file')
    scaffold_parser.add_argument('--output-dir', help='Directory to output the scaffold')
    scaffold_parser.add_argument('--advanced', action='store_true', help='Use advanced scaffolding')
    
    # Resource scaffold command
    resource_scaffold_parser = subparsers.add_parser('resource', help='Generate a resource scaffold')
    resource_scaffold_parser.add_argument('name', help='Name of the resource')
    resource_scaffold_parser.add_argument('--output-dir', help='Directory to output the scaffold')
    resource_scaffold_parser.add_argument('--advanced', action='store_true', help='Generate advanced resource scaffold')
    
    # Domain-driven scaffold command
    ddd_parser = subparsers.add_parser('domain', help='Generate a domain-driven design scaffold')
    ddd_parser.add_argument('name', help='Name of the domain')
    ddd_parser.add_argument('entities', nargs='+', help='Entities in the domain (comma-separated field definitions can be provided like User:name,email,age)')
    ddd_parser.add_argument('--output-dir', help='Directory to output the scaffold')
    
    # Microservice scaffold command
    microservice_parser = subparsers.add_parser('microservice', help='Generate a microservice scaffold')
    microservice_parser.add_argument('name', help='Name of the microservice')
    microservice_parser.add_argument('--type', choices=['api', 'worker', 'gateway', 'mcp-server'], default='api', help='Type of microservice')
    microservice_parser.add_argument('--port', type=int, default=9997, help='Port for the microservice')
    microservice_parser.add_argument('--output-dir', help='Directory to output the scaffold')
    
    # Database migration command
    migration_parser = subparsers.add_parser('migration', help='Generate a database migration')
    migration_parser.add_argument('name', help='Name of the migration')
    migration_parser.add_argument('--model', help='Model to generate migration for')
    
    # GraphQL schema generator
    graphql_parser = subparsers.add_parser('graphql', help='Generate GraphQL schemas from models')
    graphql_parser.add_argument('models', nargs='+', help='Models to generate GraphQL schemas for')
    graphql_parser.add_argument('--output', help='Output file for the GraphQL schema')
    
    args = parser.parse_args()
    
    if args.command == 'new':
        create_new_project(args.name)
    elif args.command == 'start':
        start_servers(args.ui_port, args.api_port)
    elif args.command == 'model':
        generate_model(args.name)
    elif args.command == 'mcp-server':
        start_mcp_server(args.name, args.port)
    elif args.command == 'mcp-client':
        scaffold_mcp_client(args.name, args.endpoint)
    elif args.command == 'mcp-handler':
        scaffold_mcp_handler(args.name, args.type)
    elif args.command == 'model-mcp-client':
        generate_mcp_model_client(args.model, args.endpoint)
    elif args.command == 'model-mcp-server':
        generate_mcp_model_server(args.model, args.port)
    elif args.command == 'generate':
        output = args.output or f"{args.name.lower()}_scaffold_config.py"
        
        if args.advanced:
            config_path = generate_advanced_scaffold_config(args.name, output)
            print(f"Generated advanced scaffold config: {config_path}")
        else:
            config_path = generate_scaffold_config(args.name, output)
            print(f"Generated scaffold config: {config_path}")
            
        print(f"Edit this file to customize your scaffold, then run:")
        print(f"  fort scaffold {config_path}" + (" --advanced" if args.advanced else ""))
    elif args.command == 'scaffold':
        if args.advanced:
            created_files = create_advanced_scaffold_from_config(args.config, args.output_dir)
            total_files = sum(len(files) for files in created_files.values())
            print(f"Created {total_files} files across {len(created_files)} categories")
            for category, files in created_files.items():
                if files:
                    print(f"  - {category}: {len(files)} files")
        else:
            created_files = create_scaffold_from_config(args.config, args.output_dir)
            print(f"Created {len(created_files)} files")
            
        print("Scaffold created successfully")
    elif args.command == 'resource':
        if args.advanced:
            created_files = generate_advanced_resource_scaffold(args.name, args.output_dir)
        else:
            created_files = generate_resource_scaffold(args.name, args.output_dir)
            
        print(f"Created {len(created_files)} files for resource: {args.name}")
        print("Resource scaffold created successfully")
    elif args.command == 'domain':
        # TODO: Implement domain-driven design scaffolding
        print(f"Domain-driven design scaffolding not yet implemented")
    elif args.command == 'microservice':
        # TODO: Implement microservice scaffolding
        print(f"Microservice scaffolding not yet implemented")
    elif args.command == 'migration':
        # TODO: Implement database migration
        print(f"Database migration not yet implemented")
    elif args.command == 'graphql':
        # TODO: Implement GraphQL schema generation
        print(f"GraphQL schema generation not yet implemented")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()