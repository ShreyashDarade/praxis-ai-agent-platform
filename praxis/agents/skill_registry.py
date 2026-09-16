# praxis/agents/skill_registry.py
"""Self-registering skill registry - mirrors `praxis/connectors/factory.py`
and `praxis/ingestion/parsers/registry.py` exactly (spec §3/§7's "adding
a new capability = new file; core code is never edited to add one").

Each module in `praxis/agents/skills/` calls `register_skill(...)` once,
at import time, with a fully-constructed `Skill` instance. `discover_skills()`
package-scans that subpackage (flat modules, same layout as
`praxis.ingestion.parsers`) to trigger every module's self-registration.
Adding skill #3 is one new file ending in a `register_skill(...)` call;
nothing here or anywhere else changes.
"""
from __future__ import annotations

import importlib
import pkgutil

from praxis.agents.skill import Skill

_SKILLS: dict[str, Skill] = {}
_DISCOVERED = False


def register_skill(skill: Skill) -> None:
    """Called by a skill module at import time to register itself.

    Two skills claiming the same name is a programming error, not a
    runtime condition to degrade gracefully from - fail loudly, same
    posture as `register_connector_factory`/`register_parser`.
    """
    if skill.name in _SKILLS:
        raise ValueError(f"a skill named '{skill.name}' is already registered")
    _SKILLS[skill.name] = skill


def get_skill(name: str) -> Skill:
    try:
        return _SKILLS[name]
    except KeyError:
        raise KeyError(f"no skill named '{name}' is registered") from None


def all_skills() -> list[Skill]:
    return list(_SKILLS.values())


def discover_skills() -> None:
    """Import every flat module in `praxis.agents.skills` once.

    Idempotent and safe to call repeatedly (e.g. once per test) -
    Python's own import cache plus the `_DISCOVERED` flag mean a second
    call is a no-op, so no skill is ever registered twice.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    import praxis.agents.skills as _skills_pkg

    for module_info in pkgutil.iter_modules(_skills_pkg.__path__):
        if module_info.ispkg:
            continue
        importlib.import_module(f"praxis.agents.skills.{module_info.name}")
    _DISCOVERED = True
