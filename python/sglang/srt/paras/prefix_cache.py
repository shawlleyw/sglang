"""Radix ownership for live prefixes imported by a ParaS switch."""

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class ParaSPrefixCache(RadixCache):
    # Physical identities are private to a migration. Identical logical keys
    # from different EP ranks can have distinct physical KV at the destination.
    _migrated = object()

    def adopt_live_prefix(self, req, slots, slot_key):
        key = RadixKey(slot_key, self._migrated)
        self.insert(key, slots)
        match = self.match_prefix(key)
        node = match.last_device_node
        self.inc_lock_ref(node)
        req.last_node = req.last_host_node = node
        req.prefix_indices = slots
        req.last_matched_prefix_len = len(slot_key)

    def cache_finished_req(self, req, is_insert=True):
        node = req.last_node
        migrated = node is not None and node.key.extra_key is self._migrated
        super().cache_finished_req(req, is_insert=is_insert and not migrated)
        if migrated:
            # The parent routine freed only the request-private tail and
            # released the radix lock. Reclaim newly unreferenced migrated KV.
            while (
                node is not self.root_node and not node.lock_ref and not node.children
            ):
                parent = node.parent
                self.token_to_kv_pool_allocator.free(node.value)
                self._delete_leaf(node)
                node = parent
