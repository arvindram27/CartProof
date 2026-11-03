"""Hosted slice: the public "Verify my cart" page and its security middleware.

Everything in this package is NEW hosting code. It composes the existing
storefront (over its own HTTP API) with the deterministic verification core,
the way harness/run_task.py composes adapter + core. core/ never imports
store/; nothing here changes that.
"""
