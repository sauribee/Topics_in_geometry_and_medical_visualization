#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source geometric_project_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete. Virtual environment is activated."
echo "To deactivate the virtual environment when done, type 'deactivate'." 