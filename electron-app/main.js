try {
    require('electron-reloader')(module, {
      debug: true,
      watchRenderer: true
    });
  } catch (_) { }

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

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
    // Determine if running in production (packaged) or development
    const isPackaged = app.isPackaged;

    let pythonPath;
    let scriptPath;
    let dataPath;

    if (isPackaged) {
        // Production: use bundled Python from venv
        const resourcesPath = process.resourcesPath;
        if (process.platform === 'win32') {
            pythonPath = path.join(resourcesPath, 'venv', 'Scripts', 'python.exe');
            dataPath = path.join(app.getPath('appData'), 'nous');
        } else {
            pythonPath = path.join(resourcesPath, 'venv', 'bin', 'python');
            dataPath = path.join(app.getPath('appData'), 'nous');
        }
        scriptPath = path.join(resourcesPath, 'api', 'server.py');
    } else {
        // Development: use system Python and project directories
        pythonPath = process.platform === 'win32' ? 'python' : 'python3';
        scriptPath = path.join(__dirname, '..', 'api', 'server.py');
        dataPath = path.join(__dirname, '..');
    }

    console.log(`Python path: ${pythonPath}`);
    console.log(`Script path: ${scriptPath}`);
    console.log(`Data path: ${dataPath}`);
    console.log(`Is packaged: ${isPackaged}`);

    try {
        pythonProcess = spawn(pythonPath, [scriptPath], {
            env: {
                ...process.env,
                NOUS_DATA_PATH: dataPath
            }
        });

        if (!pythonProcess) {
            console.error(`Failed to spawn python process`);
            return;
        }

        pythonProcess.stdout.on('data', (data) => {
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
