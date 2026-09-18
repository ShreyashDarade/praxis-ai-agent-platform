# praxis/agents/specialists/__init__.py
"""Specialist sub-agent implementations.

One module per specialist, each ending in a `register_agent(...)`
call - the same self-registering pattern used by connectors, parsers,
and skills. `praxis.agents.subagent.discover_agents()` imports every
module in this package to trigger those registrations, so adding a
specialist is one new file with zero edits to orchestration code.
"""
