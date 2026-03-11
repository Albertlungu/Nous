const { contextBridge, ipcRenderer } = require('electron');

const portArg = process.argv.find((arg) => arg.startsWith('--nous-api-port='));
const fallbackPort = (portArg && portArg.split('=')[1]) || '5000';

contextBridge.exposeInMainWorld('electronAPI', {
    platform: process.platform,
    apiBaseUrl: `http://127.0.0.1:${fallbackPort}`,
    getApiBaseUrl: () => ipcRenderer.invoke('nous:get-api-base-url')
});
