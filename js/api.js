/**
 * BYOK AI Chat - API Module
 * Unified provider abstraction with streaming support for:
 * OpenAI, Anthropic, Google Gemini, Ollama, LM Studio, Custom API
 * 
 * This module runs in the service worker (background.js).
 * It does NOT have access to DOM or Storage module directly.
 * Settings are passed in via message from the side panel.
 */

const API = (() => {
    // --- Provider Models ---
    const PROVIDER_MODELS = {
        openai: [
            { id: 'gpt-4o', name: 'GPT-4o' },
            { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
            { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
            { id: 'gpt-4', name: 'GPT-4' },
            { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
            { id: 'o1-preview', name: 'o1 Preview' },
            { id: 'o1-mini', name: 'o1 Mini' },
            { id: 'o3-mini', name: 'o3 Mini' },
        ],
        anthropic: [
            { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
            { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet' },
            { id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku' },
            { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus' },
        ],
        google: [
            { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
            { id: 'gemini-2.0-flash-lite', name: 'Gemini 2.0 Flash Lite' },
            { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
            { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
        ],
        ollama: [
            { id: 'llama3.1', name: 'Llama 3.1' },
            { id: 'llama3', name: 'Llama 3' },
            { id: 'mistral', name: 'Mistral' },
            { id: 'codellama', name: 'Code Llama' },
            { id: 'phi3', name: 'Phi-3' },
            { id: 'gemma2', name: 'Gemma 2' },
            { id: 'qwen2', name: 'Qwen 2' },
            { id: 'deepseek-coder-v2', name: 'DeepSeek Coder V2' },
        ],
        lmstudio: [
            { id: 'local-model', name: 'Local Model' },
        ],
        custom: [],
    };

    // --- Vision Capability Detection (from provider metadata) ---
    // GET /models on OpenAI-compatible endpoints usually says nothing about
    // image support, but some routers (OpenRouter, Ollama, vLLM, LM Studio)
    // advertise it. Returns true/false when the endpoint says something,
    // null when the capability is unknown (caller decides the fallback).
    function extractVisionCapability(m) {
        if (!m || typeof m !== 'object') return null;

        if (typeof m.vision === 'boolean') return m.vision;
        if (typeof m.supports_images === 'boolean') return m.supports_images;
        if (typeof m.image_input === 'boolean') return m.image_input;

        const caps = m.capabilities || m.supported_modalities || m.modalities || m.supported_inputs;
        if (Array.isArray(caps) && caps.length) {
            return caps.some(c => /image|vision/i.test(String(c)));
        }

        const inputs = m.architecture?.input_modalities
            || m.properties?.input_modalities
            || m.inputs
            || m.input_modalities;
        if (Array.isArray(inputs) && inputs.length) {
            return inputs.some(c => /image|vision/i.test(String(c)));
        }

        if (typeof m.modality === 'string' && m.modality.includes('->')) {
            return /image|vision/i.test(m.modality); // e.g. "text+image->text"
        }

        const families = m.details?.families || m.families;
        if (Array.isArray(families) && families.length) {
            return families.some(f => /clip|mllama|llava|vision|qwen2vl|internvl/i.test(String(f)));
        }

        return null;
    }

    // Attach `vision` to a listed model only when the endpoint advertised it.
    function withVision(model, meta) {
        const vision = extractVisionCapability(meta);
        return vision === null ? model : { ...model, vision };
    }

    // Keep the configured default model reachable even if the endpoint's
    // model list no longer contains it (renamed/retired snapshot, filtered
    // list, ...). Used by every adapter that loads models live.
    function withConfiguredDefault(models, configuredModel) {
        if (configuredModel && !models.some(m => m.id === configuredModel)) {
            models.unshift({ id: configuredModel, name: configuredModel });
        }
        return models;
    }

    // Anthropic's /models response carries no modality metadata, so the
    // capability is derived from the model generation: every Claude from
    // generation 3 on accepts image input, Claude 1/2 does not. Returns null
    // for ids we don't recognise, letting the client heuristics decide.
    function anthropicVision(id) {
        const s = String(id).toLowerCase();
        if (!s.includes('claude')) return null;
        if (/^claude-[3456789](\D|$)/.test(s) || /-[3456789]-\d/.test(s)) return true;  // claude-3-…, claude-sonnet-4-…, claude-haiku-4-5-…
        if (/^claude-(instant|[12])(\D|$)/.test(s)) return false;                        // pre-vision models
        return null;
    }

    // --- OpenAI-compatible file parts (text + images) ---
    // Shared by the OpenAI and Custom adapters, both of which speak the
    // { type: 'text' | 'image_url' } content format. The message's own text
    // is kept as the first part so typed text is never dropped.
    function processOpenAIFiles(files, messageText) {
        const textFiles = files.filter(f => f.type?.startsWith('text/') || f.name?.match(/\.(txt|md|json|csv|py|js|html|css|xml|yaml|yml|log)$/i));
        const imageFiles = files.filter(f => f.type?.startsWith('image/'));

        const content = [];
        if (messageText) content.push({ type: 'text', text: messageText });
        for (const file of textFiles) {
            content.push({ type: 'text', text: `--- File: ${file.name} ---\n${file.content}\n--- End of ${file.name} ---` });
        }
        for (const img of imageFiles) {
            content.push({ type: 'image_url', image_url: { url: img.data, detail: 'auto' } });
        }

        if (content.length === 0) return {};
        return { content };
    }

    // --- Token Estimation (approximate) ---
    function estimateTokens(text) {
        if (!text) return 0;
        return Math.ceil(text.length / 3.5);
    }

    function estimateConversationTokens(messages) {
        let total = 0;
        for (const msg of messages) {
            total += estimateTokens(msg.content) + 4;
            if (msg.files) {
                for (const file of msg.files) {
                    if (file.type?.startsWith('text/') || file.name?.match(/\.(txt|md|json|csv|py|js|html|css|xml|yaml|yml|log)$/i)) {
                        total += estimateTokens(file.content || '') + 10;
                    } else if (file.type?.startsWith('image/')) {
                        total += 85;
                    }
                }
            }
        }
        return total;
    }

    // --- OpenAI Adapter ---
    const OpenAIAdapter = {
        async *stream(messages, options, settings) {
            const apiKey = settings.providers?.openai?.apiKey || options.apiKey;
            const baseUrl = (settings.providers?.openai?.baseUrl || 'https://api.openai.com/v1').replace(/\/+$/, '');

            const body = {
                model: options.model || 'gpt-4o',
                messages: messages.map(m => ({
                    role: m.role,
                    content: m.content,
                    ...(m.files?.length ? processOpenAIFiles(m.files, m.content) : {}),
                })),
                stream: true,
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 1,
                frequency_penalty: options.frequencyPenalty ?? 0,
            };

            if (options.maxTokens) body.max_tokens = options.maxTokens;

            const response = await fetch(`${baseUrl}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                },
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
                throw new Error(error.error?.message || `OpenAI API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);
                    if (data === '[DONE]') return;

                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content) yield content;
                    } catch (e) { /* skip */ }
                }
            }
        },

        _processFiles(files, messageText) {
            return processOpenAIFiles(files, messageText);
        },

        async listModels(settings) {
            const apiKey = settings.providers?.openai?.apiKey;
            if (!apiKey) return PROVIDER_MODELS.openai;
            const baseUrl = (settings.providers?.openai?.baseUrl || 'https://api.openai.com/v1').replace(/\/+$/, '');
            try {
                const resp = await fetch(`${baseUrl}/models`, {
                    headers: { 'Authorization': `Bearer ${apiKey}` },
                });
                if (resp.ok) {
                    const data = await resp.json();
                    const models = data.data
                        .filter(m => m.id.includes('gpt') || m.id.includes('o1') || m.id.includes('o3') || m.id.includes('chat'))
                        .map(m => withVision({ id: m.id, name: m.id }, m))
                        .sort((a, b) => a.name.localeCompare(b.name));
                    return models.length > 0 ? models : PROVIDER_MODELS.openai;
                }
            } catch (e) { /* fallback */ }
            return PROVIDER_MODELS.openai;
        },
    };

    // --- Anthropic Adapter ---
    const AnthropicAdapter = {
        async *stream(messages, options, settings) {
            const apiKey = settings.providers?.anthropic?.apiKey || options.apiKey;
            const baseUrl = (settings.providers?.anthropic?.baseUrl || 'https://api.anthropic.com').replace(/\/+$/, '');

            let systemPrompt = '';
            const filteredMessages = messages.filter(m => {
                if (m.role === 'system') {
                    systemPrompt += (systemPrompt ? '\n\n' : '') + m.content;
                    return false;
                }
                return true;
            });

            const body = {
                model: options.model || 'claude-sonnet-4-20250514',
                max_tokens: options.maxTokens || 128000,
                messages: filteredMessages.map(m => {
                    const msg = { role: m.role, content: m.content };
                    if (m.files?.length) {
                        const content = [];
                        const imageFiles = m.files.filter(f => f.type?.startsWith('image/'));
                        const textFiles = m.files.filter(f => f.type?.startsWith('text/') || f.name?.match(/\.(txt|md|json|csv|py|js|html|css|xml|yaml|yml|log)$/i));
                        for (const tf of textFiles) {
                            content.push({ type: 'text', text: `--- File: ${tf.name} ---\n${tf.content}\n--- End of ${tf.name} ---` });
                        }
                        for (const img of imageFiles) {
                            const mediaType = img.type || 'image/png';
                            const base64Data = img.data?.replace(/^data:[^;]+;base64,/, '') || '';
                            content.push({ type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Data } });
                        }
                        if (m.content) content.unshift({ type: 'text', text: m.content });
                        msg.content = content;
                    }
                    return msg;
                }),
                stream: true,
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 1,
            };

            if (systemPrompt) body.system = systemPrompt;

            const response = await fetch(`${baseUrl}/v1/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                    'anthropic-dangerous-direct-browser-access': 'true',
                },
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
                throw new Error(error.error?.message || `Anthropic API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);

                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
                            yield parsed.delta.text;
                        }
                    } catch (e) { /* skip */ }
                }
            }
        },

        // Live model list from GET {baseUrl}/v1/models. The response is
        // paginated with an `after_id` cursor, so follow it for a few pages
        // and deduplicate. Like Custom, opts.strict (used by
        // TEST_CONNECTION) turns missing credentials/fetch/HTTP failures into
        // a real error instead of the static fallback list.
        async listModels(settings, opts = {}) {
            const strict = !!opts.strict;
            const apiKey = settings.providers?.anthropic?.apiKey;
            const baseUrl = (settings.providers?.anthropic?.baseUrl || 'https://api.anthropic.com').replace(/\/+$/, '');
            // The saved model (only meaningful when Anthropic is the
            // selected provider) must stay reachable in the dropdown.
            const configuredModel = settings.defaultProvider === 'anthropic'
                ? (settings.defaultModel || '').trim()
                : '';

            if (!apiKey) {
                if (strict) throw new Error('Anthropic: API key is required. Configure it in Settings.');
                return PROVIDER_MODELS.anthropic;
            }

            const headers = {
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
                'anthropic-dangerous-direct-browser-access': 'true',
            };

            try {
                const models = [];
                let url = `${baseUrl}/v1/models?limit=1000`;

                for (let page = 0; page < 5 && url; page++) {
                    const resp = await fetch(url, { headers });
                    if (!resp.ok) {
                        if (strict) throw new Error(`Anthropic: GET /v1/models failed (${resp.status} ${resp.statusText})`);
                        break;
                    }

                    const data = await resp.json();
                    for (const m of data.data || []) {
                        const id = m?.id;
                        if (!id || models.some(x => x.id === id)) continue;
                        const vision = anthropicVision(id);
                        models.push({ id, name: m.display_name || id, ...(vision === null ? {} : { vision }) });
                    }

                    url = data.has_more && data.last_id
                        ? `${baseUrl}/v1/models?limit=1000&after_id=${encodeURIComponent(data.last_id)}`
                        : null;
                }

                if (models.length) {
                    return withConfiguredDefault(
                        models.sort((a, b) => a.name.localeCompare(b.name)),
                        configuredModel
                    );
                }
            } catch (e) {
                if (strict) throw e; // surface the real error to the caller
            }
            return PROVIDER_MODELS.anthropic;
        },
    };

    // --- Google Gemini Adapter ---
    const GoogleAdapter = {
        async *stream(messages, options, settings) {
            const apiKey = settings.providers?.google?.apiKey || options.apiKey;
            const model = options.model || 'gemini-2.0-flash';

            const contents = [];
            let systemInstruction = '';

            for (const m of messages) {
                if (m.role === 'system') {
                    systemInstruction += (systemInstruction ? '\n\n' : '') + m.content;
                    continue;
                }

                const parts = [];
                if (m.content) {
                    parts.push({ text: m.content });
                }
                if (m.files?.length) {
                    for (const file of m.files) {
                        if (file.type?.startsWith('image/')) {
                            const base64Data = file.data?.replace(/^data:[^;]+;base64,/, '') || '';
                            parts.push({
                                inline_data: {
                                    mime_type: file.type,
                                    data: base64Data,
                                }
                            });
                        } else {
                            parts.push({ text: `--- File: ${file.name} ---\n${file.content}\n--- End of ${file.name} ---` });
                        }
                    }
                }

                contents.push({
                    role: m.role === 'assistant' ? 'model' : 'user',
                    parts,
                });
            }

            const body = {
                contents,
                generationConfig: {
                    temperature: options.temperature ?? 0.7,
                    topP: options.topP ?? 1,
                    ...(options.maxTokens ? { maxOutputTokens: options.maxTokens } : {}),
                },
            };

            if (systemInstruction) {
                body.systemInstruction = { parts: [{ text: systemInstruction }] };
            }

            const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
                throw new Error(error.error?.message || `Google API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);

                    try {
                        const parsed = JSON.parse(data);
                        const text = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
                        if (text) yield text;
                    } catch (e) { /* skip */ }
                }
            }
        },

        // Live model list from GET /v1beta/models. Only models that support
        // generateContent are chat-usable, which drops embedding/aqa/imagen
        // entries. The API carries no image-input metadata, so `vision` is
        // left for the client heuristics (Gemini family => images allowed).
        // opts.strict (used by TEST_CONNECTION) reports missing credentials
        // and fetch/HTTP failures instead of the static fallback list.
        async listModels(settings, opts = {}) {
            const strict = !!opts.strict;
            const apiKey = settings.providers?.google?.apiKey;
            const configuredModel = settings.defaultProvider === 'google'
                ? (settings.defaultModel || '').trim()
                : '';

            if (!apiKey) {
                if (strict) throw new Error('Google: API key is required. Configure it in Settings.');
                return PROVIDER_MODELS.google;
            }

            try {
                const models = [];
                let pageToken = '';

                // The list is paginated (default page size is small); follow
                // the token for a few pages, which covers every model list
                // Google returns for a single API key.
                for (let page = 0; page < 5; page++) {
                    const url = `https://generativelanguage.googleapis.com/v1beta/models`
                        + `?key=${encodeURIComponent(apiKey)}${pageToken ? `&pageToken=${encodeURIComponent(pageToken)}` : ''}`;
                    const resp = await fetch(url);
                    if (!resp.ok) {
                        if (strict) throw new Error(`Google: GET /v1beta/models failed (${resp.status} ${resp.statusText})`);
                        break;
                    }

                    const data = await resp.json();
                    for (const m of data.models || []) {
                        if (!(m.supportedGenerationMethods || []).includes('generateContent')) continue; // drops embedding/aqa/imagen
                        const id = String(m.name || '').replace(/^models\//, '');
                        if (id && !models.some(x => x.id === id)) {
                            models.push({ id, name: m.displayName || id });
                        }
                    }

                    pageToken = data.nextPageToken || '';
                    if (!pageToken) break;
                }

                if (models.length > 0) {
                    return withConfiguredDefault(
                        models.sort((a, b) => a.name.localeCompare(b.name)),
                        configuredModel
                    );
                }
            } catch (e) {
                if (strict) throw e; // surface the real error to the caller
            }
            return PROVIDER_MODELS.google;
        },
    };

    // --- Ollama Adapter ---
    const OllamaAdapter = {
        async *stream(messages, options, settings) {
            const baseUrl = (settings.providers?.ollama?.baseUrl || 'http://localhost:11434').replace(/\/+$/, '');

            const body = {
                model: options.model || 'llama3.1',
                messages: messages.map(m => {
                    const msg = { role: m.role, content: m.content };
                    if (m.files?.length) {
                        const imageFiles = m.files.filter(f => f.type?.startsWith('image/'));
                        if (imageFiles.length > 0) {
                            msg.images = imageFiles.map(f => f.data?.replace(/^data:[^;]+;base64,/, '') || '');
                        }
                        const textFiles = m.files.filter(f => f.type?.startsWith('text/') || f.name?.match(/\.(txt|md|json|csv|py|js|html|css|xml|yaml|yml|log)$/i));
                        if (textFiles.length > 0) {
                            const fileTexts = textFiles.map(f => `--- File: ${f.name} ---\n${f.content}\n--- End of ${f.name} ---`);
                            msg.content = fileTexts.join('\n\n') + '\n\n' + (m.content || '');
                        }
                    }
                    return msg;
                }),
                stream: true,
                options: {
                    temperature: options.temperature ?? 0.7,
                    top_p: options.topP ?? 1,
                    ...(options.maxTokens ? { num_predict: options.maxTokens } : {}),
                },
            };

            const response = await fetch(`${baseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: response.statusText }));
                throw new Error(error.error || `Ollama API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;

                    try {
                        const parsed = JSON.parse(trimmed);
                        if (parsed.message?.content) yield parsed.message.content;
                        if (parsed.done) return;
                    } catch (e) { /* skip */ }
                }
            }
        },

        async listModels(settings) {
            const baseUrl = (settings.providers?.ollama?.baseUrl || 'http://localhost:11434').replace(/\/+$/, '');
            try {
                const resp = await fetch(`${baseUrl}/api/tags`);
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.models?.length) {
                        return data.models.map(m => withVision({ id: m.name, name: m.name }, m));
                    }
                }
            } catch (e) { /* fallback */ }
            return PROVIDER_MODELS.ollama;
        },
    };

    // --- LM Studio Adapter (OpenAI-compatible) ---
    const LMStudioAdapter = {
        async *stream(messages, options, settings) {
            const baseUrl = (settings.providers?.lmstudio?.baseUrl || 'http://localhost:1234/v1').replace(/\/+$/, '');

            const body = {
                model: options.model || 'local-model',
                messages: messages.map(m => ({
                    role: m.role,
                    content: m.content,
                })),
                stream: true,
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 1,
            };

            if (options.maxTokens) body.max_tokens = options.maxTokens;

            const response = await fetch(`${baseUrl}/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
                throw new Error(error.error?.message || `LM Studio API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);
                    if (data === '[DONE]') return;

                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content) yield content;
                    } catch (e) { /* skip */ }
                }
            }
        },

        async listModels(settings) {
            const baseUrl = (settings.providers?.lmstudio?.baseUrl || 'http://localhost:1234/v1').replace(/\/+$/, '');
            try {
                const resp = await fetch(`${baseUrl}/models`);
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.data?.length) {
                        return data.data.map(m => withVision({ id: m.id, name: m.id }, m));
                    }
                }
            } catch (e) { /* fallback */ }
            return PROVIDER_MODELS.lmstudio;
        },
    };

    // --- Custom API Adapter (OpenAI-compatible format) ---
    // Behaves exactly like the OpenAI adapter: model comes from the model
    // dropdown, images/text files are sent as OpenAI content parts, and the
    // model list is loaded live from GET {baseUrl}/models.
    const CustomAdapter = {
        async *stream(messages, options, settings) {
            const apiKey = settings.providers?.custom?.apiKey || options.apiKey;
            const baseUrl = (settings.providers?.custom?.baseUrl || '').replace(/\/+$/, '');
            const model = options.model || settings.providers?.custom?.model || 'custom-model';

            if (!baseUrl) {
                throw new Error('Custom API: Base URL is required. Configure it in Settings.');
            }

            const body = {
                model,
                messages: messages.map(m => ({
                    role: m.role,
                    content: m.content,
                    ...(m.files?.length ? processOpenAIFiles(m.files, m.content) : {}),
                })),
                stream: true,
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 1,
            };

            if (options.maxTokens) body.max_tokens = options.maxTokens;

            const headers = { 'Content-Type': 'application/json' };
            if (apiKey) {
                headers['Authorization'] = `Bearer ${apiKey}`;
            }

            const response = await fetch(`${baseUrl}/chat/completions`, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
                signal: options.abortSignal,
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
                throw new Error(error.error?.message || `Custom API error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);
                    if (data === '[DONE]') return;

                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content) yield content;
                    } catch (e) { /* skip */ }
                }
            }
        },

        // Load the model list exactly like the OpenAI adapter does:
        // GET {baseUrl}/models, sorted, with the configured default model
        // guaranteed to appear in the dropdown.
        // opts.strict (used by TEST_CONNECTION) turns fetch/HTTP failures into
        // a thrown error so "Test Connection" can honestly report failure;
        // the dropdown path (no opts) always falls back gracefully.
        async listModels(settings, opts = {}) {
            const strict = !!opts.strict;
            const apiKey = settings.providers?.custom?.apiKey;
            const baseUrl = (settings.providers?.custom?.baseUrl || '').replace(/\/+$/, '');
            const configuredModel = settings.providers?.custom?.model?.trim();
            const fallback = configuredModel ? [{ id: configuredModel, name: configuredModel }] : [];

            if (!baseUrl) {
                if (strict) throw new Error('Custom API: Base URL is required. Configure it in Settings.');
                return fallback;
            }

            try {
                const headers = {};
                if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

                const resp = await fetch(`${baseUrl}/models`, { headers });
                if (!resp.ok) {
                    if (strict) throw new Error(`Custom API: GET /models failed (${resp.status} ${resp.statusText})`);
                } else {
                    const data = await resp.json();
                    const list = Array.isArray(data.data) ? data.data
                        : Array.isArray(data.models) ? data.models
                        : [];

                    const models = list
                        .map(m => m?.id || m?.name)
                        .filter(Boolean)
                        .map(id => withVision({ id, name: id }, list.find(m => (m.id || m.name) === id)))
                        .sort((a, b) => a.name.localeCompare(b.name));

                    if (models.length > 0) {
                        // Keep the configured default reachable even if the
                        // endpoint doesn't list it.
                        return withConfiguredDefault(models, configuredModel);
                    }
                }
            } catch (e) {
                if (strict) throw e; // surface the real error to the caller
            }
            return fallback;
        },
    };

    // --- Provider Registry ---
    const providers = {
        openai: OpenAIAdapter,
        anthropic: AnthropicAdapter,
        google: GoogleAdapter,
        ollama: OllamaAdapter,
        lmstudio: LMStudioAdapter,
        custom: CustomAdapter,
    };

    function getProvider(name) {
        return providers[name] || null;
    }

    function getModelsForProvider(providerName) {
        return PROVIDER_MODELS[providerName] || [];
    }

    // --- Unified Send Function ---
    async function* sendMessage(messages, options, settings) {
        const provider = getProvider(options.provider);
        if (!provider) {
            throw new Error(`Unknown provider: ${options.provider}`);
        }
        yield* provider.stream(messages, options, settings);
    }

    // --- Auto-title generation ---
    async function generateTitle(messages, provider, model, settings) {
        try {
            const titlePrompt = [
                { role: 'system', content: 'Generate a very short title (3-6 words) for this conversation. Reply with ONLY the title, nothing else. No quotes, no punctuation at the end.' },
                { role: 'user', content: messages.slice(0, 2).map(m => m.content).join('\n\n') },
            ];

            let title = '';
            for await (const chunk of sendMessage(titlePrompt, {
                provider,
                model,
                temperature: 0.5,
                maxTokens: 30,
            }, settings)) {
                title += chunk;
            }

            return title.trim().replace(/^["']|["']$/g, '') || 'New Chat';
        } catch (e) {
            return 'New Chat';
        }
    }

    // --- Connection Test ---
    async function testConnection(providerName, settings) {
        try {
            const provider = getProvider(providerName);
            if (!provider) return { success: false, error: 'Unknown provider' };

            if (provider.listModels) {
                // strict: adapters that can (Custom) surface real fetch/HTTP
                // errors instead of silently returning their fallback list.
                const models = await provider.listModels(settings, { strict: true });
                return { success: true, models };
            }

            // For providers without listModels, try a minimal request
            const testMessages = [{ role: 'user', content: 'Hi' }];
            let response = '';
            for await (const chunk of provider.stream(testMessages, {
                model: getModelsForProvider(providerName)[0]?.id || 'test',
                temperature: 0.1,
                maxTokens: 5,
            }, settings)) {
                response += chunk;
                break; // Just need one chunk to confirm connection
            }
            return { success: true };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    return {
        providers,
        getProvider,
        getModelsForProvider,
        sendMessage,
        generateTitle,
        testConnection,
        estimateTokens,
        estimateConversationTokens,
        PROVIDER_MODELS,
    };
})();