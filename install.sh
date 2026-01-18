#!/bin/bash
set -e

pip install --extra-index-url https://download.pytorch.org/whl/cu126 .

# coqui-tts claims transformers>=4.43 but actually needs <4.40 (LogitsWarper removed in 4.40+)
pip install 'transformers>=4.39,<4.40'

RED='\033[0;31m'
NC='\033[0m'
echo -e "${RED}-------------------------------------------------${NC}"
echo -e "${RED}Don't worry about the error you have been shown.${NC}"
echo -e "${RED}-------------------------------------------------${NC}"

echo "Done. Run with: python -m wyoming_xtts"
