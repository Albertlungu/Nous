echo "Starting PyGPT..."

if [ ! -d "venv" ]; then
    echo "Error: virtual environment not found. Please run ./metal_setup.sh first."
    exit 1
fi

echo "Starting Flask server..."
source venv/bin/activate
python api/server.py &
FLASK_PID=$!

sleep 2

echo "Starting Electron app..."
cd electron-app
npm start

echo "Shutting down Flask..."
kill $FLASK_PID