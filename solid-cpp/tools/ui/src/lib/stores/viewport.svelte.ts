import { browser } from '$app/environment';
import { DEFAULT_MOBILE_BREAKPOINT } from '$lib/constants/viewport';
import { MediaQuery } from 'svelte/reactivity';

export const viewport = $state({
	width: browser ? window.innerWidth : 0
});

export const isMobile = new MediaQuery(`max-width: ${DEFAULT_MOBILE_BREAKPOINT - 1}px`);

/** Whether the mobile sidebar panel is currently expanded (full-width overlay with its own
 * close button) - read by anything else fixed near the top-right corner so it can get out of
 * the way instead of overlapping the sidebar's close button. */
export const sidebarState = $state({ expanded: false });
