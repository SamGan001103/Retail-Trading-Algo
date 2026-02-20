"""
Compatibility surface for older imports.

New code should use broker adapters (`broker.create_stream`) instead of importing a concrete
realtime stream class from runtime directly.
"""

from trading_algo.broker.projectx_realtime import JsonHubProtocolRS, ProjectXRealtimeStream, RTState

# Backward-compat alias for existing imports.
RealtimeClient = ProjectXRealtimeStream

__all__ = ["JsonHubProtocolRS", "RTState", "RealtimeClient", "ProjectXRealtimeStream"]
