const FALLBACK_API_BASE = 'http://127.0.0.1:5000';

const configuredBase = window.electronAPI?.apiBaseUrl || FALLBACK_API_BASE;
window.NOUS_API_BASE = configuredBase;

if (typeof window.electronAPI?.getApiBaseUrl === 'function') {
    window.electronAPI.getApiBaseUrl()
        .then((baseUrl) => {
            if (typeof baseUrl === 'string' && baseUrl.startsWith('http')) {
                window.NOUS_API_BASE = baseUrl;
            }
        })
        .catch(() => {
            // Keep fallback base URL if IPC lookup fails.
        });
}

const nativeFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
    if (typeof input === 'string' && input.startsWith('/api/')) {
        return nativeFetch(`${window.NOUS_API_BASE}${input}`, init);
    }

    return nativeFetch(input, init);
};
