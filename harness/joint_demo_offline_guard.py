"""Fail-closed credential and network boundary for offline validation only."""
from __future__ import annotations
from contextlib import contextmanager,ExitStack
import os,socket
from unittest.mock import patch

@contextmanager
def offline_boundary():
    """Deny all socket egress, real httpx transports and credential fallback.

    MockTransport remains functional. This is not a malicious-code sandbox.
    It prevents accidental real network calls in the reviewed offline suite.
    """
    blocked=[]
    def deny(*a,**kw):
        blocked.append('network_attempt')
        raise RuntimeError('offline_network_denied')
    names=[k for k in os.environ if any(x in k.upper() for x in (
        'API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','AUTHORIZATION'))
        or k in ('CARTPROOF_SHOPIFY_LIVE','CARTPROOF_JOINT_PARENT_LAUNCH')]
    stripped={k:os.environ[k] for k in names}
    for k in names:os.environ.pop(k,None)
    try:
        with ExitStack() as stack:
            for name in ('connect','connect_ex','sendto'):
                stack.enter_context(patch.object(socket.socket,name,deny))
            stack.enter_context(patch.object(socket,'create_connection',deny))
            stack.enter_context(patch.object(socket,'getaddrinfo',deny))
            import httpx
            stack.enter_context(patch.object(httpx.HTTPTransport,'handle_request',deny))
            stack.enter_context(patch.object(httpx.AsyncHTTPTransport,'handle_async_request',deny))
            from agents import env_loader
            stack.enter_context(patch.object(env_loader,'load_root_dotenv',
                side_effect=RuntimeError('offline_credential_fallback_denied')))
            from harness import shopify_live_backend
            stack.enter_context(patch.object(shopify_live_backend,'resolve_token',return_value=''))
            yield blocked
    finally:
        for k in names:os.environ.pop(k,None)
        os.environ.update(stripped)
