/**
 * BYOK AI Chat - Background Service Worker
 * Handles API calls (avoids CORS), screenshots, page context extraction,
 * context menu, and message routing between side panel and content scripts.
 */

// Import API module via importScripts (service worker context)
importScripts('js/api.js');
importScripts('js/searx-api.js');

// --- Side Panel Setup ---
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error);

// --- Context Menu ---
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'send-to-ai-chat',
        title: 'Send to BYOK AI Chat',
        contexts: ['selection'],
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'send-to-ai-chat' && info.selectionText) {
        // Open side panel and send selected text
        chrome.sidePanel.open({ windowId: tab.windowId }).then(() => {
            // Small delay to let panel load
            setTimeout(() => {
                chrome.runtime.sendMessage({
                    type: 'CONTEXT_MENU_TEXT',
                    text: info.selectionText,
                    sourceUrl: tab.url,
                    sourceTitle: tab.title,
                }).catch(() => { /* panel may not be ready */ });
            }, 500);
        });
    }
});

// --- Service Worker Keep-Alive ---
// MV3 service workers are terminated after ~30 seconds of inactivity.
// Long provider calls (reasoning models, local servers still loading a
// model, multi-step web search) can easily exceed that, and when the worker
// dies mid-request the side panel sees "Connection to service worker lost".
// While any request is in flight we call an extension API on a short
// interval, which resets the idle timer and keeps the worker alive.
const KEEPALIVE_INTERVAL_MS = 20000;
let keepAliveTimer = null;
let activeRequests = 0;

function startKeepAlive() {
    activeRequests++;
    if (keepAliveTimer) return;
    keepAliveTimer = setInterval(() => {
        // Any extension API call resets the service worker idle timer.
        chrome.runtime.getPlatformInfo(() => {});
    }, KEEPALIVE_INTERVAL_MS);
}

function stopKeepAlive() {
    activeRequests = Math.max(0, activeRequests - 1);
    if (activeRequests === 0 && keepAliveTimer) {
        clearInterval(keepAliveTimer);
        keepAliveTimer = null;
    }
}

// Run a handler with the keep-alive heartbeat active until it responds.
function runWithKeepAlive(fn, sendResponse) {
    startKeepAlive();
    Promise.resolve()
        .then(fn)
        .catch((err) => {
            try { sendResponse({ success: false, ok: false, error: err?.message || String(err) }); } catch (e) { /* channel closed */ }
        })
        .finally(() => stopKeepAlive());
    return true; // keep the message channel open for the async response
}

// --- Message Handler ---
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // Heartbeat from the side panel. Receiving a message resets the worker's
    // idle timer, so pings every <30s keep it alive during long operations.
    if (message.type === 'PING') {
        sendResponse({ ok: true });
        return false;
    }

    switch (message.type) {
        case 'CAPTURE_SCREENSHOT':
            return runWithKeepAlive(() => handleCaptureScreenshot(sender, sendResponse), sendResponse);

        case 'GET_PAGE_CONTENT':
            return runWithKeepAlive(() => handleGetPageContent(message, sender, sendResponse), sendResponse);

        case 'LIST_MODELS':
            return runWithKeepAlive(() => handleListModels(message, sendResponse), sendResponse);

        case 'GENERATE_TITLE':
            return runWithKeepAlive(() => handleGenerateTitle(message, sendResponse), sendResponse);

        case 'TEST_CONNECTION':
            return runWithKeepAlive(() => handleTestConnection(message, sendResponse), sendResponse);

        case 'GET_SETTINGS':
            return runWithKeepAlive(() => handleGetSettings(sendResponse), sendResponse);

        case 'SAVE_SETTINGS':
            return runWithKeepAlive(() => handleSaveSettings(message, sendResponse), sendResponse);

        case 'WEB_SEARCH':
            return runWithKeepAlive(() => handleWebSearch(message, sendResponse), sendResponse);

        case 'VISIT_WEBSITE':
            return runWithKeepAlive(() => handleVisitWebsite(message, sendResponse), sendResponse);

        case 'WEB_SEARCH_TEST':
            return runWithKeepAlive(() => handleWebSearchTest(sendResponse), sendResponse);

        default:
            return false;
    }
});

// --- Port-based Streaming Handler ---
// The side panel opens a port via chrome.runtime.connect() for streaming.
// This is the ONLY reliable way to stream from service worker → side panel.
// A keep-alive heartbeat runs for the whole stream so the service worker is
// never terminated mid-generation ("Connection to service worker lost").
chrome.runtime.onConnect.addListener((port) => {
    if (port.name !== 'ai-stream') return;

    // Per-connection abort controller so the panel can stop generation
    let abortController = null;

    port.onMessage.addListener((message) => {
        // Panel asked us to stop generating
        if (message.type === 'ABORT_STREAM') {
            try { abortController?.abort(); } catch (e) {}
            return;
        }
        if (message.type !== 'SEND_MESSAGE') return;

        const { messages, settings } = message;
        const options = { ...message.options };
        abortController = new AbortController();
        options.abortSignal = abortController.signal;

        startKeepAlive();
        (async () => {
            try {
                const stream = API.sendMessage(messages, options, settings);
                let fullContent = '';

                for await (const chunk of stream) {
                    fullContent += chunk;
                    // Send streaming chunks back through the port
                    try {
                        port.postMessage({
                            type: 'STREAM_CHUNK',
                            content: chunk,
                            fullContent,
                            provider: options.provider,
                            model: options.model,
                        });
                    } catch (e) {
                        // Port disconnected — panel was closed
                        break;
                    }
                }

                // Send completion signal
                try {
                    port.postMessage({
                        type: 'STREAM_DONE',
                        content: fullContent,
                        aborted: abortController.signal.aborted,
                        provider: options.provider,
                        model: options.model,
                    });
                } catch (e) { /* port closed */ }
            } catch (error) {
                if (error?.name === 'AbortError') {
                    try {
                        port.postMessage({
                            type: 'STREAM_DONE',
                            content: '',
                            aborted: true,
                            provider: options.provider,
                            model: options.model,
                        });
                    } catch (e) { /* port closed */ }
                } else {
                    try {
                        port.postMessage({
                            type: 'STREAM_ERROR',
                            error: error.message,
                            provider: options.provider,
                            model: options.model,
                        });
                    } catch (e) { /* port closed */ }
                }
            } finally {
                abortController = null;
                stopKeepAlive();
            }
        })();
    });

    // Clean up when the port disconnects (panel closed or stream stopped)
    port.onDisconnect.addListener(() => {
        try { abortController?.abort(); } catch (e) {}
        abortController = null;
    });
});

// --- Handler: Capture Screenshot ---
async function handleCaptureScreenshot(sender, sendResponse) {
    try {
        // Get the active tab in the current window
        const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
        if (!tab) {
            sendResponse({ success: false, error: 'No active tab found' });
            return;
        }

        const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
            format: 'png',
            quality: 90,
        });

        sendResponse({ success: true, dataUrl, tabTitle: tab.title, tabUrl: tab.url });
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// --- Handler: Get Page Content ---
async function handleGetPageContent(message, sender, sendResponse) {
    try {
        const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
        if (!tab) {
            sendResponse({ success: false, error: 'No active tab found' });
            return;
        }

        const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: extractPageContent,
            args: [message.maxTextLength || 4000],
        });

        const pageContext = results?.[0]?.result || null;
        sendResponse({ success: true, context: pageContext, tabTitle: tab.title, tabUrl: tab.url });
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// Function that runs in the page context
function extractPageContent(maxTextLength) {
    const article = document.querySelector('article') || document.querySelector('main') || document.querySelector('[role="main"]');
    let text = '';

    if (article) {
        text = article.innerText || article.textContent || '';
    }
    if (text.length < 200) {
        text = document.body?.innerText || document.body?.textContent || '';
    }

    text = text.replace(/\s+/g, ' ').trim();
    if (text.length > maxTextLength) {
        text = text.substring(0, maxTextLength) + '... [truncated]';
    }

    const selection = window.getSelection();
    const selectedText = selection ? selection.toString().trim() : '';

    const meta = document.querySelector('meta[name="description"]');

    return {
        title: document.title || '',
        url: window.location.href || '',
        selectedText,
        pageText: text,
        metaDescription: meta?.content || '',
        language: document.documentElement.lang || '',
    };
}

// --- Handler: List Models ---
async function handleListModels(message, sendResponse) {
    try {
        const { provider, settings } = message;
        const adapter = API.getProvider(provider);

        if (adapter?.listModels) {
            const models = await adapter.listModels(settings);
            sendResponse({ success: true, models });
        } else {
            const models = API.getModelsForProvider(provider);
            sendResponse({ success: true, models });
        }
    } catch (error) {
        sendResponse({ success: false, error: error.message, models: API.getModelsForProvider(message.provider) });
    }
}

// --- Handler: Generate Title ---
async function handleGenerateTitle(message, sendResponse) {
    try {
        const { messages, provider, model, settings } = message;
        const title = await API.generateTitle(messages, provider, model, settings);
        sendResponse({ success: true, title });
    } catch (error) {
        sendResponse({ success: false, title: 'New Chat' });
    }
}

// --- Handler: Test Connection ---
async function handleTestConnection(message, sendResponse) {
    try {
        const { provider, settings } = message;
        const result = await API.testConnection(provider, settings);
        sendResponse(result);
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// --- Handler: Get Settings ---
async function handleGetSettings(sendResponse) {
    try {
        const result = await chrome.storage.local.get('aichat_settings');
        const settings = result.aichat_settings || getDefaultSettings();
        sendResponse({ success: true, settings });
    } catch (error) {
        sendResponse({ success: false, error: error.message, settings: getDefaultSettings() });
    }
}

// --- Handler: Save Settings ---
async function handleSaveSettings(message, sendResponse) {
    try {
        await chrome.storage.local.set({ aichat_settings: message.settings });
        sendResponse({ success: true });
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// --- Default Settings ---
function getDefaultSettings() {
    return {
        providers: {
            openai: { apiKey: '', baseUrl: 'https://api.openai.com/v1' },
            anthropic: { apiKey: '', baseUrl: 'https://api.anthropic.com' },
            google: { apiKey: '', baseUrl: '' },
            ollama: { apiKey: '', baseUrl: 'http://localhost:11434' },
            lmstudio: { apiKey: '', baseUrl: 'http://localhost:1234/v1' },
            custom: { apiKey: '', baseUrl: '', model: '' },
        },
        defaultProvider: 'openai',
        defaultModel: 'gpt-4o',
        defaultSystemPrompt: 'You are a helpful assistant.',
        defaultTemperature: 0.7,
        defaultTopP: 1,
        contextMode: 'truncate',
        maxContextMessages: 50,
        webSearchEnabled: false,
        webSearchMode: 'off',
    };
}

// --- Handler: Web Search ---
async function handleWebSearch(message, sendResponse) {
    try {
        const { query } = message;
        const result = await WebSearchAPI.search(query);
        sendResponse({ success: true, data: result });
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// --- Handler: Visit Website ---
async function handleVisitWebsite(message, sendResponse) {
    try {
        const { url, maxTextLength } = message;
        const result = await WebSearchAPI.visit(url, {
            maxTextLength: maxTextLength || 6000,
            timeout: 15000,
        });
        sendResponse({ success: true, data: result });
    } catch (error) {
        sendResponse({ success: false, error: error.message });
    }
}

// --- Handler: Web Search Test ---
async function handleWebSearchTest(sendResponse) {
    try {
        const result = await WebSearchAPI.testConnection();
        sendResponse(result);
    } catch (error) {
        sendResponse({ ok: false, error: error.message });
    }
}
