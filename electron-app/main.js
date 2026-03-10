try {
    require('electron-reloader')(module, {
      debug: true,
      watchRenderer: true
    });
  } catch (_) { }

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const net = require('net');
const http = require('http');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;
let apiPort = 5000;

function isPortAvailable(port) {
    return new Promise((resolve) => {
        const server = net.createServer();

        server.once('error', () => resolve(false));
        server.once('listening', () => {
            server.close(() => resolve(true));
        });

        server.listen(port, '127.0.0.1');
    });
}

async function findAvailablePort(startPort = 5000, maxAttempts = 200) {
    for (let offset = 0; offset < maxAttempts; offset += 1) {
        const candidate = startPort + offset;
        if (await isPortAvailable(candidate)) {
            return candidate;
        }
    }

    throw new Error(`No available port found from ${startPort} to ${startPort + maxAttempts - 1}`);
}

function waitForBackend(port, timeoutMs = 20000, intervalMs = 250) {
    return new Promise((resolve) => {
        const start = Date.now();

        const tryHealth = () => {
            const req = http.get(
                {
                    hostname: '127.0.0.1',
                    port,
                    path: '/api/health',
                    timeout: 1500
                },
                (res) => {
                    if (res.statusCode === 200) {
                        res.resume();
                        resolve(true);
                        return;
                    }

                    res.resume();
                    if (Date.now() - start >= timeoutMs) {
                        resolve(false);
                    } else {
                        setTimeout(tryHealth, intervalMs);
                    }
                }
            );

            req.on('error', () => {
                if (Date.now() - start >= timeoutMs) {
                    resolve(false);
                } else {
                    setTimeout(tryHealth, intervalMs);
                }
            });

            req.on('timeout', () => {
                req.destroy();
            });
        };

        tryHealth();
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            additionalArguments: [`--nous-api-port=${apiPort}`]
        },
        backgroundColor: '#1a1a1a',
        titleBarStyle: 'hiddenInset'
    });

    mainWindow.loadFile('renderer/index.html');

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startPythonServer() {
    const fs = require('fs');
    const fsPromises = fs.promises;

    // Determine if running in production (packaged) or development
    const isPackaged = app.isPackaged;

    let pythonPath;
    let scriptPath;
    let backendCommand;
    let backendArgs = [];
    let dataPath;
    let logPath;

    if (isPackaged) {
        // Production: prefer a bundled backend executable on macOS.
        const resourcesPath = process.resourcesPath;
        const userDataPath = app.getPath('userData');
        if (process.platform === 'darwin') {
            backendCommand = path.join(resourcesPath, 'backend', 'nous-api-server');
            scriptPath = path.join(resourcesPath, 'api', 'server.py');

            if (fs.existsSync(backendCommand)) {
                try {
                    fs.chmodSync(backendCommand, 0o755);
                } catch (_) {
                    // If chmod fails, spawn will report the error.
                }
            } else {
                // Last resort fallback for local testing if backend binary is missing.
                backendCommand = 'python3';
                backendArgs = [scriptPath];
            }
        } else if (process.platform === 'win32') {
            pythonPath = 'python';
            scriptPath = path.join(resourcesPath, 'api', 'server.py');
        } else {
            pythonPath = 'python3';
            scriptPath = path.join(resourcesPath, 'api', 'server.py');
        }
        dataPath = userDataPath;
        logPath = path.join(userDataPath, 'nous-debug.log');
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

    if (backendCommand) {
        log(`Backend command: ${backendCommand}`);
        log(`Backend args: ${backendArgs.join(' ') || '(none)'}`);
    } else {
        log(`Python path: ${pythonPath}`);
        log(`Script path: ${scriptPath}`);
    }
    log(`Data path: ${dataPath}`);
    log(`Is packaged: ${isPackaged}`);
    log(`Log path: ${logPath}`);

    const copyDirIfMissing = async (sourceDir, targetDir) => {
        if (!fs.existsSync(sourceDir) || fs.existsSync(targetDir)) {
            return;
        }

        await fsPromises.mkdir(path.dirname(targetDir), { recursive: true });
        await fsPromises.cp(sourceDir, targetDir, { recursive: true });
        log(`Seeded data directory from resources: ${sourceDir} -> ${targetDir}`);
    };

    const launchPython = async () => {
        try {
            apiPort = await findAvailablePort(5000, 200);
            log(`Selected API port: ${apiPort}`);

            const command = backendCommand || pythonPath;
            const args = backendCommand ? backendArgs : [scriptPath];

            pythonProcess = spawn(command, args, {
                env: {
                    ...process.env,
                    NOUS_DATA_PATH: dataPath,
                    NOUS_API_PORT: String(apiPort)
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
                log(`Failed to start backend: ${err.message}`);
            });

            const ready = await waitForBackend(apiPort, 20000, 250);
            if (!ready) {
                log(`Backend health check timed out on port ${apiPort}`);
            } else {
                log(`Backend ready on port ${apiPort}`);
            }

            if (!mainWindow) {
                createWindow();
            }
        } catch (err) {
            log(`Error spawning Python: ${err.message}`);
            if (!mainWindow) {
                createWindow();
            }
        }
    };

    if (isPackaged) {
        const resourcesPath = process.resourcesPath;
        copyDirIfMissing(path.join(resourcesPath, 'artifacts'), path.join(dataPath, 'artifacts'))
            .then(() => copyDirIfMissing(path.join(resourcesPath, 'training_data'), path.join(dataPath, 'training_data')))
            .then(launchPython)
            .catch((err) => {
                log(`Failed to seed user data: ${err.message}`);
                launchPython();
            });
    } else {
        launchPython();
    }
}

app.whenReady().then(() => {
    startPythonServer();

    ipcMain.handle('nous:get-api-base-url', async () => `http://127.0.0.1:${apiPort}`);

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

    ipcMain.removeHandler('nous:get-api-base-url');
});
