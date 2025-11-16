#!/bin/bash

# Credit Default Prediction - Setup Script
# This script sets up the development environment

echo "========================================"
echo "Credit Default Prediction - Setup"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo -e "\n${YELLOW}Installing requirements...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

# Create directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p data/raw data/processed models logs results notebooks tests config examples

echo -e "${GREEN}✓ Directories created${NC}"

# Check if data exists
echo -e "\n${YELLOW}Checking for training data...${NC}"
if [ -f "data/raw/cs-training.csv" ]; then
    echo -e "${GREEN}✓ Training data found${NC}"
else
    echo -e "${YELLOW}⚠ Training data not found${NC}"
    echo "Please place your training data in: data/raw/cs-training.csv"
fi

# Display next steps
echo -e "\n========================================"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "========================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place your training data in:"
echo "   data/raw/cs-training.csv"
echo ""
echo "3. Run the training pipeline:"
echo "   cd src"
echo "   python train.py"
echo ""
echo "4. View results in:"
echo "   models/ - Trained models"
echo "   results/ - Evaluation results and plots"
echo "   logs/ - Training logs"
echo ""
echo "For more information, see README.md"
echo "========================================"
