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
    const fs = require('fs');

    // Determine if running in production (packaged) or development
    const isPackaged = app.isPackaged;

    let pythonPath;
    let scriptPath;
    let dataPath;
    let logPath;

    if (isPackaged) {
        // Production: use bundled Python from venv and Resources directory for data
        const resourcesPath = process.resourcesPath;
        if (process.platform === 'win32') {
            pythonPath = path.join(resourcesPath, 'venv', 'Scripts', 'python.exe');
            dataPath = resourcesPath;
        } else {
            pythonPath = path.join(resourcesPath, 'venv', 'bin', 'python');
            dataPath = resourcesPath;
        }
        scriptPath = path.join(resourcesPath, 'api', 'server.py');
        logPath = path.join(dataPath, 'nous-debug.log');
    } else {
        // Development: use system Python and project directories
        pythonPath = process.platform === 'win32' ? 'python' : 'python3';
        scriptPath = path.join(__dirname, '..', 'api', 'server.py');
        dataPath = path.join(__dirname, '..');
        logPath = path.join(dataPath, 'nous-debug.log');
    }

    // Create data directory if it doesn't exist
    if (!fs.existsSync(dataPath)) {
        fs.mkdirSync(dataPath, { recursive: true });
    }

    const logStream = fs.createWriteStream(logPath, { flags: 'a' });
    const log = (msg) => {
        const timestamp = new Date().toISOString();
        const logMsg = `[${timestamp}] ${msg}\n`;
        console.log(msg);
        logStream.write(logMsg);
    };

    log(`Python path: ${pythonPath}`);
    log(`Script path: ${scriptPath}`);
    log(`Data path: ${dataPath}`);
    log(`Is packaged: ${isPackaged}`);
    log(`Log path: ${logPath}`);

    try {
        pythonProcess = spawn(pythonPath, [scriptPath], {
            env: {
                ...process.env,
                NOUS_DATA_PATH: dataPath
            }
        });

        if (!pythonProcess) {
            log(`Failed to spawn python process`);
            return;
        }

        pythonProcess.stdout.on('data', (data) => {
            log(`Python stdout: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            log(`Python stderr: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            log(`Python process exited with code: ${code}`);
        });

        pythonProcess.on('error', (err) => {
            log(`Failed to start Python: ${err.message}`);
        });
    } catch (err) {
        log(`Error spawning Python: ${err.message}`);
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
