try {
    require('electron-reloader')(module, {
      debug: true,
      watchRenderer: true
    });
  } catch (_) { }

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { APIConnectionError } = require('openai');
const { create } = require('domain');
const { transformedClearcoatNormalView } = require('three/src/nodes/TSL.js');

let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        },
        backgroundColor: '#1a1a1a',
        titleBarStyle: 'hiddenInset'
    });

    mainWindow.loadFile('renderer/index.html');

    mainWindow.webContents.openDevTools();

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startPythonServer() {
    // Start flask API server
    const pythonPath = 'python3';
    const scriptPath = path.join(__dirname, '..', 'api', 'server.py');

    console.log(`Attempting to start python server at ${scriptPath}`)

    try {
        pythonProcess = spawn(pythonPath, [scriptPath]);

        if (!pythonProcess) {
            console.error(`Failed to spawn python process`);
            return;
        }

        pythonPath.stdout.on('data', (data) => {
            console.log(`Python: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python Error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code: ${code}`);
        });

        pythonProcess.on('error', (err) => {
            console.error(`Failed to start Python: ${err.message}`);
        });
    } catch (err) {
        console.error(`Error spawning Python: ${err.message}`);
    }
}

app.whenReady().then(() => {
    startPythonServer();

    setTimeout(() => {
        createWindow();
    }, 2000);

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }

    if (process.platform != 'darwin') {
        app.quit();
    }
});

app.on('quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});