<script lang="ts">
	let { children } = $props();
</script>

<div class="min-h-full">
	{@render children?.()}
</div>
