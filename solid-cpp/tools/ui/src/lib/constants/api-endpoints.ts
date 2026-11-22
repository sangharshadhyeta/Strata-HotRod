export const API_MODELS = {
	LIST: '/v1/models',
	LOAD: '/models/load',
	SSE: '/models/sse',
	UNLOAD: '/models/unload'
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const API_CHAT = {
	COMPLETIONS: './v1/chat/completions',
	CONTROL: './v1/chat/completions/control'
};

// slot introspection, requires the --slots flag on the server
export const API_SLOTS = {
	LIST: './slots'
};

// live per-expert moe-cache tier/heat map, for the Brain view. Present
// whenever the server is built with CUDA moe-cache support; returns an
// empty grid (rows/cols 0) rather than an error before anything has been
// cached yet.
export const API_EXPERTS = {
	LIST: './experts'
};

export const API_TOOLS = {
	EXECUTE: '/tools',
	LIST: '/tools'
};

// resumable stream routes, the conv::model identity travels as the conv_id query param
// because model names can contain slashes that a path segment cannot carry
// resume retry cadence while the owning model is still loading (server answers 503)
export const STREAM_RESUME_RETRY_MS = 2000;

export const API_STREAM = {
	BASE: './v1/stream',
	LOOKUP: './v1/streams/lookup'
};

/** CORS proxy endpoint path */
export const CORS_PROXY_ENDPOINT = '/cors-proxy';
