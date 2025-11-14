from agent_workbench.trace import TraceWriter, TraceReader


def test_trace_write_and_read(tmp_path):
    tw = TraceWriter(str(tmp_path))
    rid = tw.new_run()
    tw.append(rid, {"type": "test", "value": 1})
    tr = TraceReader(str(tmp_path))
    evs = list(tr.read(rid))
    assert evs and evs[0]["type"] == "test"
