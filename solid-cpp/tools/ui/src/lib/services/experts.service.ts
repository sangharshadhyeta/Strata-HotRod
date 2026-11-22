import { API_EXPERTS } from '$lib/constants';
import type { ApiExpertMapResponse } from '$lib/types/api';
import { apiFetch } from '$lib/utils';

export class ExpertsService {
	/**
	 * Fetch the live per-(layer,expert) moe-cache tier/heat snapshot for the
	 * Brain view. `map`/`hits` are hex-packed byte strings (see
	 * docs/moe-cache-colibri-notes.md for the wire format, ported from
	 * Colibri); rows/cols are 0 when nothing has been cached yet.
	 */
	static async get(): Promise<ApiExpertMapResponse> {
		return apiFetch<ApiExpertMapResponse>(API_EXPERTS.LIST);
	}
}
