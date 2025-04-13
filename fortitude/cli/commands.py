import os
import sys
import subprocess
from .model_generator import ModelGenerator

def create_new_project(name: str):
    """Create a new Fortitude project"""
    # Create project directory
    project_dir = os.path.join(os.getcwd(), name)
    if os.path.exists(project_dir):
        print(f"Error: Directory {project_dir} already exists")
        sys.exit(1)
    
    os.makedirs(project_dir)
    
    # Create project structure
    dirs = [
        "ui",
        "ui/app",
        "ui/components",
        "ui/public",
        "backend",
        "backend/models",
        "backend/endpoints",
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(project_dir, d), exist_ok=True)
    
    # Create package.json for UI
    package_json = os.path.join(project_dir, "ui/package.json")
    with open(package_json, 'w') as f:
        f.write("""{
  "name": "fortitude-ui",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 9996",
    "build": "next build",
    "start": "next start -p 9996",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.0.0"
  }
}
""")
    
    # Create requirements.txt for backend
    requirements = os.path.join(project_dir, "requirements.txt")
    with open(requirements, 'w') as f:
        f.write("""fastapi>=0.103.1
uvicorn>=0.23.2
pydantic>=2.4.2
asyncio>=3.4.3
aiohttp>=3.8.5
""")
    
    # Create main application file
    main_py = os.path.join(project_dir, "main.py")
    with open(main_py, 'w') as f:
        f.write(f"""from fortitude.core import FortitudeApp

app = FortitudeApp("{name}")

# Define your models here
# Example:
# from fortitude.backend.models import FortitudeBaseModel
# from pydantic import Field
# from typing import Optional
#
# class User(FortitudeBaseModel):
#     name: str
#     email: str
#     age: Optional[int] = None
#
# app.register_model(User)

def start():
    app.start(ui_port=9996, api_port=9997)

if __name__ == "__main__":
    start()
""")
    
    print(f"Created new Fortitude project: {name}")
    print(f"\nTo get started:")
    print(f"  cd {name}")
    print(f"  pip install -r requirements.txt")
    print(f"  cd ui && npm install")
    print(f"  cd .. && python -m fortitude.cli.fort start")

def start_servers(ui_port: int = 9996, api_port: int = 9997):
    """Start UI and API servers"""
    # Check if we're in a Fortitude project
    if not (os.path.exists("ui") and os.path.exists("backend")):
        print("Error: Not in a Fortitude project directory")
        sys.exit(1)
    
    print(f"Starting UI server on port {ui_port}")
    ui_process = subprocess.Popen(
        ["npm", "run", "dev", "--", "-p", str(ui_port)],
        cwd="ui"
    )
    
    print(f"Starting API server on port {api_port}")
    if os.path.exists("main.py"):
        api_process = subprocess.Popen(
            [sys.executable, "main.py"]
        )
    else:
        print("Error: main.py not found")
        ui_process.terminate()
        sys.exit(1)
    
    try:
        ui_process.wait()
        api_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        ui_process.terminate()
        api_process.terminate()

def generate_model(name: str):
    """Generate a new model"""
    generator = ModelGenerator(os.getcwd())
    model_file = generator.generate_model(name)
    
    print(f"Generated model: {model_file}")
    print(f"Don't forget to register it in your main.py file:")
    print(f"\nfrom backend.models.{name.lower()} import {name.capitalize()}")
    print(f"app.register_model({name.capitalize()})")
