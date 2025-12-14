echo "=== νοῦς (nous) Metal Setup Script ==="
echo ""

echo "Installing PyEnv..."
curl https://pyenv.run | bash
pyenv --version

echo ""
echo "Installing Python 3.10"
pyenv install 3.10
pyenv local 3.10

# Run this to add the setup code to both ~/.zshrc and ~/.zprofile
grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' ~/.zshrc || cat << 'EOF' >> ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
EOF

grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' ~/.zprofile || cat << 'EOF' >> ~/.zprofile
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
EOF


pyenv shell 3.10
python --version # Should return Python 3.10.x

echo ""
echo "Setting Up Virtual Environment..."
python -m venv venv
source venv/bin/activate
which python # Should return "/Users/[your_user]/[something]/Nous/venv/bin/python"

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setting up Node.js and Electron ==="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install node
else
    echo "Node.js is already installed: $(node --version)"
fi

echo ""
echo "Installing Electron dependencies..."
cd electron-app
npm install
cd ..

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start the app:"
echo "  1. In one terminal, start the Flask server:"
echo "     source venv/bin/activate"
echo "     python api/server.py"
echo ""
echo "  2. In another terminal, start the Electron app:"
echo "     cd electron-app"
echo "     npm start"
echo ""
echo "Or use the convenience script:"
echo "     ./start_app.sh"

echo "This application has been made by Albert Lungu. Enjoy!