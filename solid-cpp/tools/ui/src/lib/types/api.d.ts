import type { ChatMessagePromptProgress, ChatRole } from './chat';
import type {
	ContentPartType,
	FileTypeAudio,
	ServerModelsSseEventType,
	ServerModelStatus,
	ServerRole
} from '$lib/enums';

export type AudioInputFormat = FileTypeAudio.WAV | FileTypeAudio.MP3;

export interface ApiChatCompletionToolFunction {
	name: string;
	description?: string;
	parameters: Record<string, unknown>;
}

export interface ApiChatCompletionTool {
	type: 'function';
	function: ApiChatCompletionToolFunction;
}

export interface ApiChatMessageContentPart {
	type: ContentPartType;
	text?: string;
	image_url?: {
		url: string;
	};
	input_audio?: {
		data: string;
		format: AudioInputFormat;
	};
	input_video?: {
		data: string;
		format: 'mp4' | 'ogg' | 'auto';
	};
}

export interface ApiContextSizeError {
	code: number;
	message: string;
	type: 'exceed_context_size_error';
	n_prompt_tokens: number;
	n_ctx: number;
}

export interface ApiErrorResponse {
	error:
		| ApiContextSizeError
		| {
				code: number;
				message: string;
				type?: string;
		  };
}

export interface ApiChatMessageData {
	role: ChatRole;
	content: string | ApiChatMessageContentPart[];
	reasoning_content?: string;
	tool_calls?: ApiChatCompletionToolCall[];
	tool_call_id?: string;
	timestamp?: number;
}

/**
 * Model status object from /models endpoint
 */
export interface ApiModelStatus {
	/** Status value: loaded, unloaded, loading, sleeping, failed */
	value: ServerModelStatus;
	/** Command line arguments used when loading (only for loaded models) */
	args?: string[];
}

/**
 * Model entry from /models endpoint (ROUTER mode)
 * Based on actual API response structure
 */
export interface ApiModelDataEntry {
	/** Model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF:latest") */
	id: string;
	/** Model name (optional, usually same as id - not always returned by API) */
	name?: string;
	/** Object type, always "model" */
	object: string;
	/** Owner, usually "llamacpp" */
	owned_by: string;
	/** Creation timestamp */
	created: number;
	/** Whether model files are in HuggingFace cache */
	in_cache: boolean;
	/** Path to model manifest file */
	path: string;
	/** Current status of the model */
	status: ApiModelStatus;
	/** Alternative names that resolve to this model */
	aliases?: string[];
	/** Informational tags for this model */
	tags?: string[];
	/** Modality capabilities, reported by the router for every model regardless of load state */
	architecture?: ApiModelArchitecture;
	/** Legacy meta field (may be present in older responses) */
	meta?: Record<string, unknown> | null;
}

/**
 * Modality capabilities of a model, as advertised by the ROUTER /models endpoint.
 * Read from the model manifest, so it is available before the model is loaded.
 */
export interface ApiModelArchitecture {
	/** Accepted input modalities, always contains "text" */
	input_modalities: string[];
}

/**
 * Load stage reported by the /models/sse feed, in load order.
 */
export type ApiModelLoadStage = 'text_model' | 'spec_model' | 'mmproj_model';

/**
 * Load progress snapshot: the full ordered stage plan, the active stage,
 * and its fractional value (0.0 -> 1.0).
 */
export interface ApiModelsSseProgress {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
}

/**
 * Status payload carried by a /models/sse envelope.
 * exit_code appears on unload.
 */
export interface ApiModelsSseData {
	status: ServerModelStatus;
	progress?: ApiModelsSseProgress;
	exit_code?: number;
}

/**
 * Event kind multiplexed on the /models/sse feed.
 * Only the status_* events carry a status payload, models_reload signals a
 * full list refresh, model_remove drops a row, download_* drive download UI.
 */
/**
 * One /models/sse record. event discriminates the kind, model names the
 * target instance, data carries the status payload when present.
 */
export interface ApiModelsSseEvent {
	model: string;
	event: ServerModelsSseEventType;
	data: ApiModelsSseData;
}

export interface ApiModelDetails {
	name: string;
	model: string;
	modified_at?: string;
	size?: string | number;
	digest?: string;
	type?: string;
	description?: string;
	tags?: string[];
	capabilities?: string[];
	parameters?: string;
	details?: {
		parent_model?: string;
		format?: string;
		family?: string;
		families?: string[];
		parameter_size?: string;
		quantization_level?: string;
	};
}

export interface ApiModelListResponse {
	object: string;
	data: ApiModelDataEntry[];
	models?: ApiModelDetails[];
}

/** Live per-(layer,expert) moe-cache tier/heat snapshot for the Brain view.
 * `map`/`hits` are hex-packed byte strings - see
 * docs/moe-cache-colibri-notes.md for the exact wire format (ported from
 * Colibri). rows/cols are 0 when nothing has been cached yet. */
/** Aggregate moe-cache health, summed across every live cache device.
 * Empty object when no cache pool has been allocated yet. */
export interface ApiExpertMapStats {
	hit_rate?: number;
	hits?: number;
	misses?: number;
	evictions?: number;
	fill_failures?: number;
	admission_skips?: number;
	slots_used?: number;
	slots_total?: number;
	protected_slots?: number;
	avg_heat?: number;
	allocated_mib?: number;
	budget_mib?: number;
	/** Real router-rank misses served by a resident stand-in
	 * (GGML_CUDA_MOE_CACHE_SUBSTITUTE) instead of CPU fallback - not folded
	 * into `hits`/`misses` above (a substitution is a miss that was made
	 * cheap, not a hit). declined = a substitution was attempted but no
	 * resident stand-in existed. */
	substitutions?: number;
	substitute_declined?: number;
	/** Track 1 step 0/1 (docs/plan.md): live, decaying centroid of where the
	 * current request is trending in the same (x,y) topic-affinity space
	 * the atlas cells below use. Absent until at least one atlas-covered
	 * expert has actually been selected - not present at all rather than
	 * a stale (0,0) placeholder. */
	req_dir?: { x: number; y: number };
	co_activation_within_layer?: ApiCoActivationEdge[];
	co_activation_cross_layer?: ApiCoActivationEdge[];
}

/** One co-activation edge - see Track 1 steps 4a/4b in docs/plan.md.
 * within_layer: two experts selected together in one routing decision
 * (undirected). cross_layer: layer_from's top pick -> layer_to's top pick
 * (directed) - note a known gap, not yet fixed server-side: this doesn't
 * distinguish a genuinely later layer from a different tensor of the SAME
 * logical layer (gate_up_exps -> down_exps share one router decision), so
 * some cross-layer edges are same-layer artifacts (layer_from === layer_to
 * is the tell). */
export interface ApiCoActivationEdge {
	layer_from: number;
	expert_from: number;
	layer_to: number;
	expert_to: number;
	count: number;
	/** Strength relative to the strongest edge in the SAME layer, 0..1.
	 * Raw counts are not comparable across layers - they differ by an order of
	 * magnitude - so scaling line opacity by a global max made every drawn edge
	 * look identical. Optional: absent from servers predating per-layer edge
	 * sampling. */
	weight?: number;
}

/** One measured (layer,expert) topic-affinity cell from llama-expert-atlas.
 * x/y come from the discovered co-activation embedding, projected to a disc by
 * scripts/moe-atlas-evolve.py (the static 9-topic generator this used to cite
 * was removed). Magnitude reflects how specialized the expert is (near 0,0 =
 * generalist, near the rim = a topic specialist); the layer component is
 * removed from the layout deliberately, since the Grid panel already shows it. */
export interface ApiExpertAtlasCell {
	layer: number;
	expert: number;
	n: number;
	spec: number;
	x: number;
	y: number;
	/** Per-category probability vector, sparse (categories this expert never
	 * fired on are omitted). Absent in atlas files generated before this
	 * field existed - consumers must fall back to the 2D x/y in that case.
	 * x/y are this same vector collapsed onto a circle, which is lossy: an
	 * expert firing on exactly two categories lands on a straight chord
	 * between their anchors (22% of experts on a real Ornith atlas), and
	 * anything firing on all of them collapses to the origin alongside
	 * experts that fired on none. Keeping the vector lets a consumer project
	 * into 3D (or re-project after the category set changes) instead. */
	cats?: Record<string, number>;
}

export interface ApiExpertAtlas {
	/** True when the axes were FACTORED OUT of observed co-activation rather
	 * than measured against the hardcoded k_probes[] category list. The two
	 * are not interchangeable to a viewer: probe axes are named topics and a
	 * cell's x/y is a convex combination of category anchors, while discovered
	 * axes are eigenvector components with no name and no anchor geometry -
	 * so the category ring must not be drawn over them. */
	discovered?: boolean;
	categories: string[];
	n_layer: number;
	n_probes: number;
	cells: ApiExpertAtlasCell[];
}

export interface ApiExpertMapResponse {
	rows: number;
	cols: number;
	map: string;
	hits: string;
	/** Bitset (1 bit/cell, same shape as `map`) of cells that served as a
	 * resident substitution stand-in since the last poll - read-and-cleared
	 * server-side, so unlike `hits` this needs no client-side diffing.
	 * Empty string when the provider has nothing to report (older builds,
	 * or a mid-poll shape mismatch) - treat as "no substitutions this poll". */
	substitutions: string;
	/** Bitset (1 bit/cell, same shape as `map`) of cells belonging to the
	 * speculative draft rather than the target. A live snapshot, not
	 * read-and-cleared: a tensor's role never changes. Absent entirely when no
	 * draft is loaded, or when the draft's experts are GPU-resident and so never
	 * enter the expert cache at all. The draft keeps its own pools, heat and knob
	 * values, so these cells are shown in their own hue family. */
	draft?: string;
	/** Live per-cell dead-neuron-fraction proxy (0.0-1.0, -1.0 = no data
	 * yet), same rows x cols shape as `map`, row-major flat array. Only
	 * present when the server has GGML_CUDA_MOE_CACHE_NEURON_HEAT enabled
	 * - absent (not an empty array) otherwise, so "not measured" and "flat
	 * zero" stay distinguishable. Unlike `substitutions`, this is a live
	 * snapshot (not read-and-cleared), so re-polling without new traffic
	 * returns the same values. */
	neuron_concentration?: number[];
	seq: number;
	stats: ApiExpertMapStats;
	atlas?: ApiExpertAtlas;
}

export interface ApiLlamaCppServerProps {
	default_generation_settings: {
		id: number;
		id_task: number;
		n_ctx: number;
		speculative: boolean;
		is_processing: boolean;
		params: {
			n_predict: number;
			seed: number;
			temperature: number;
			dynatemp_range: number;
			dynatemp_exponent: number;
			top_k: number;
			top_p: number;
			min_p: number;
			top_n_sigma: number;
			xtc_probability: number;
			xtc_threshold: number;
			typ_p: number;
			repeat_last_n: number;
			repeat_penalty: number;
			presence_penalty: number;
			frequency_penalty: number;
			dry_multiplier: number;
			dry_base: number;
			dry_allowed_length: number;
			dry_penalty_last_n: number;
			dry_sequence_breakers: string[];
			mirostat: number;
			mirostat_tau: number;
			mirostat_eta: number;
			stop: string[];
			max_tokens: number;
			n_keep: number;
			n_discard: number;
			ignore_eos: boolean;
			stream: boolean;
			logit_bias: Array<[number, number]>;
			n_probs: number;
			min_keep: number;
			grammar: string;
			grammar_lazy: boolean;
			grammar_triggers: string[];
			preserved_tokens: number[];
			chat_format: string;
			reasoning_format: string;
			reasoning_in_content: boolean;
			generation_prompt: string;
			samplers: string[];
			backend_sampling: boolean;
			'speculative.n_max': number;
			'speculative.n_min': number;
			'speculative.p_min': number;
			timings_per_token: boolean;
			post_sampling_probs: boolean;
			lora: Array<{ name: string; scale: number }>;
		};
		prompt: string;
		next_token: {
			has_next_token: boolean;
			has_new_line: boolean;
			n_remain: number;
			n_decoded: number;
			stopping_word: string;
		};
	};
	total_slots: number;
	model_path: string;
	role: ServerRole;
	modalities: {
		vision: boolean;
		audio: boolean;
		video: boolean;
	};
	chat_template: string;
	bos_token: string;
	eos_token: string;
	build_info: string;
	/** @deprecated Use {@link ui_settings} instead */
	webui_settings?: Record<string, string | number | boolean>;
	ui_settings?: Record<string, string | number | boolean>;
	cors_proxy_enabled?: boolean;
	/** What this deployment's own placement logic actually decided (e.g. via --moe-cache), not just
	 * what was requested on the command line. Live cache performance (hit rate, VRAM used, etc.)
	 * belongs in the Brain view instead - this is deployment configuration, not a live metric. */
	solid_cpp?: {
		/** Experts per MoE layer; absent/0 for a dense (non-MoE) model. */
		n_expert?: number;
		/** Number of layers whose MoE experts are offloaded to CPU (moe-cache tracked). */
		n_cpu_moe?: number;
		/** What the placement search decided, and what it was decided against. */
		placement?: {
			/** -ncmoe as requested on the command line (-1 when not specified). */
			n_cpu_moe_requested?: number;
			/** -ncmoe actually used, after raising it to fit the requested context. */
			n_cpu_moe_final?: number;
			/** Total layers in the model, for "N of M" framing. */
			n_layer?: number;
			/** Context size as requested, before any fit reduction. */
			n_ctx_requested?: number;
		};
	};
}

export interface ApiChatCompletionRequest {
	messages: Array<{
		role: ChatRole;
		content: string | ApiChatMessageContentPart[];
		reasoning_content?: string;
		tool_calls?: ApiChatCompletionToolCall[];
		tool_call_id?: string;
	}>;
	stream?: boolean;
	model?: string;
	return_progress?: boolean;
	sse_ping_interval?: number;
	tools?: ApiChatCompletionTool[];
	// Reasoning parameters
	reasoning_format?: string;
	// Generation parameters
	temperature?: number;
	max_tokens?: number;
	// Sampling parameters
	dynatemp_range?: number;
	dynatemp_exponent?: number;
	top_k?: number;
	top_p?: number;
	min_p?: number;
	xtc_probability?: number;
	xtc_threshold?: number;
	typ_p?: number;
	// Penalty parameters
	repeat_last_n?: number;
	repeat_penalty?: number;
	presence_penalty?: number;
	frequency_penalty?: number;
	dry_multiplier?: number;
	dry_base?: number;
	dry_allowed_length?: number;
	dry_penalty_last_n?: number;
	// Sampler configuration
	samplers?: string[];
	backend_sampling?: boolean;
	// Custom parameters (JSON string)
	custom?: Record<string, unknown>;
	timings_per_token?: boolean;
	// Continuation control (vLLM compat)
	add_generation_prompt?: boolean;
	continue_final_message?: boolean;
}

export interface ApiChatCompletionToolCallFunctionDelta {
	name?: string;
	arguments?: string;
}

export interface ApiChatCompletionToolCallDelta {
	index?: number;
	id?: string;
	type?: string;
	function?: ApiChatCompletionToolCallFunctionDelta;
}

export interface ApiChatCompletionToolCall extends ApiChatCompletionToolCallDelta {
	function?: ApiChatCompletionToolCallFunctionDelta & { arguments?: string };
}

export interface ApiChatCompletionStreamChunk {
	id?: string;
	object?: string;
	model?: string;
	choices: Array<{
		model?: string;
		metadata?: { model?: string };
		delta: {
			content?: string;
			reasoning_content?: string;
			model?: string;
			tool_calls?: ApiChatCompletionToolCallDelta[];
		};
		finish_reason?: string | null;
	}>;
	timings?: {
		prompt_n?: number;
		prompt_ms?: number;
		predicted_n?: number;
		predicted_ms?: number;
		cache_n?: number;
	};
	prompt_progress?: ChatMessagePromptProgress;
}

export interface ApiChatCompletionResponse {
	model?: string;
	choices: Array<{
		model?: string;
		metadata?: { model?: string };
		message: {
			content: string;
			reasoning_content?: string;
			model?: string;
			tool_calls?: ApiChatCompletionToolCall[];
		};
		finish_reason?: string | null;
	}>;
}

export interface ApiSlotData {
	id: number;
	id_task: number;
	n_ctx: number;
	speculative: boolean;
	is_processing: boolean;
	params: {
		n_predict: number;
		seed: number;
		temperature: number;
		dynatemp_range: number;
		dynatemp_exponent: number;
		top_k: number;
		top_p: number;
		min_p: number;
		top_n_sigma: number;
		xtc_probability: number;
		xtc_threshold: number;
		typical_p: number;
		repeat_last_n: number;
		repeat_penalty: number;
		presence_penalty: number;
		frequency_penalty: number;
		dry_multiplier: number;
		dry_base: number;
		dry_allowed_length: number;
		dry_penalty_last_n: number;
		mirostat: number;
		mirostat_tau: number;
		mirostat_eta: number;
		max_tokens: number;
		n_keep: number;
		n_discard: number;
		ignore_eos: boolean;
		stream: boolean;
		n_probs: number;
		min_keep: number;
		chat_format: string;
		reasoning_format: string;
		reasoning_in_content: boolean;
		generation_prompt: string;
		samplers: string[];
		backend_sampling: boolean;
		'speculative.n_max': number;
		'speculative.n_min': number;
		'speculative.p_min': number;
		timings_per_token: boolean;
		post_sampling_probs: boolean;
		lora: Array<{ name: string; scale: number }>;
	};
	next_token: {
		has_next_token: boolean;
		has_new_line: boolean;
		n_remain: number;
		n_decoded: number;
	};
}

export interface ApiProcessingState {
	status: 'initializing' | 'generating' | 'preparing' | 'idle';
	tokensDecoded: number;
	tokensRemaining: number;
	contextUsed: number;
	contextTotal: number | null;
	outputTokensUsed: number; // Total output tokens (thinking + regular content)
	outputTokensMax: number; // Max output tokens allowed
	temperature: number;
	topP: number;
	speculative: boolean;
	hasNextToken: boolean;
	tokensPerSecond?: number;
	// Progress information from prompt_progress
	progressPercent?: number;
	promptProgress?: ChatMessagePromptProgress;
	promptTokens?: number;
	promptMs?: number;
	cacheTokens?: number;
}

/**
 * Router model metadata - extended from ApiModelDataEntry with additional router-specific fields
 * @deprecated Use ApiModelDataEntry instead - the /models endpoint returns this structure directly
 */
export interface ApiRouterModelMeta {
	/** Model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF:latest") */
	name: string;
	/** Path to model file or manifest */
	path: string;
	/** Optional path to multimodal projector */
	path_mmproj?: string;
	/** Whether model is in HuggingFace cache */
	in_cache: boolean;
	/** Port where model instance is running (0 if not loaded) */
	port?: number;
	/** Current status of the model */
	status: ApiModelStatus;
	/** Error message if status is FAILED */
	error?: string;
}

/**
 * Request to load a model
 */
export interface ApiRouterModelsLoadRequest {
	model: string;
}

/**
 * Response from loading a model
 */
export interface ApiRouterModelsLoadResponse {
	success: boolean;
	error?: string;
}

/**
 * Request to check model status
 */
export interface ApiRouterModelsStatusRequest {
	model: string;
}

/**
 * Response with model status
 */
export interface ApiRouterModelsStatusResponse {
	model: string;
	status: ModelStatus;
	port?: number;
	error?: string;
}

/**
 * Response with list of all models from /models endpoint
 * Note: This is the same as ApiModelListResponse - the endpoint returns the same structure
 * regardless of server mode (MODEL or ROUTER)
 */
export interface ApiRouterModelsListResponse {
	object: string;
	data: ApiModelDataEntry[];
}

/**
 * Request to unload a model
 */
export interface ApiRouterModelsUnloadRequest {
	model: string;
}

/**
 * Response from unloading a model
 */
export interface ApiRouterModelsUnloadResponse {
	success: boolean;
	error?: string;
}

/**
 * Entry returned by POST /v1/streams/lookup. The client passes the conv ids it owns in the body
 * and the server returns one entry per matching live or recently completed background streaming
 * session, keyed by conversation_id. The WebUI uses this at mount and on visibilitychange to
 * populate sidebar spinners and to reattach to an ongoing inference for the active conversation.
 * The server never lists ids the client did not ask about, so foreign random UUIDs stay private.
 */
export interface ApiStreamSession {
	conversation_id: string;
	is_done: boolean;
	total_bytes: number;
	started_at: number;
	completed_at: number;
}
