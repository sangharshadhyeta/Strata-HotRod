/**
 *
 * Brain
 *
 * Live per-expert moe-cache tier/heat map, ported from Colibri's Brain.tsx
 * (see docs/moe-cache-colibri-notes.md). Polls GET /experts and renders a
 * Canvas 2D grid: colour = cache tier, brightness = heat, flash-on-hit
 * pulse animation for cells that were just written to.
 *
 */
export { default as Brain } from './Brain.svelte';
