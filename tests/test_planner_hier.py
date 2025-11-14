from agent_workbench.planner_hier import Manager


def test_hierarchical_planner_builds_dag():
    m = Manager(["web.fetch", "python.run", "fs.write"], concurrency=2)
    g = m.build_plan("Research and save")
    assert len(g.nodes) >= 1
    # Ensure edges reflect sequence
    if len(g.nodes) > 1:
        assert len(list(g.edges)) >= 1
