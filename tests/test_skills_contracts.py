from agent_workbench.settings import Settings
from agent_workbench.skills import SkillsRegistry, SkillContext


def test_skills_registry_allows_and_validates():
    settings = Settings.load()
    reg = SkillsRegistry(settings)
    reg.load_builtins()
    assert "fs.write" in reg.list()
    ctx = SkillContext(session_id="t", settings=settings)
    ok = reg.execute("fs.write", ctx, {"path": "test_contract.txt", "content": "x"})
    assert ok.get("success") is True
    bad = reg.execute("fs.write", ctx, {"path": 123})
    assert bad.get("success") is False
