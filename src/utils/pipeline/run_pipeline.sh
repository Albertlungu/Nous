#!/bin/bash

# Nous Corpus Data Pipeline Runner
# Runs the complete pipeline to generate nous_corpus.txt and nous_corpus.pkl

echo "================================================================================"
echo "NOUS CORPUS DATA PIPELINE"
echo "================================================================================"
echo "Target: 100B tokens for 9B parameter model"
echo "Output: training_data/nous_corpus.txt and nous_corpus.pkl"
echo "================================================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "Step 1/2: Combining all datasets into nous_corpus.txt..."
echo "--------------------------------------------------------------------------------"
python src/utils/pipeline/combine_datasets.py

if [ $? -ne 0 ]; then
    echo "✗ Error: Dataset combination failed"
    exit 1
fi

echo ""
echo "Step 2/2: Tokenizing corpus to nous_corpus.pkl..."
echo "--------------------------------------------------------------------------------"
python src/utils/pipeline/tokenize_corpus.py

if [ $? -ne 0 ]; then
    echo "✗ Error: Tokenization failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo "Output files:"
echo "  - training_data/nous_corpus.txt (text corpus)"
echo "  - training_data/nous_corpus.pkl (tokenized corpus)"
echo "================================================================================"
echo "Ready for training!"
echo "================================================================================"
