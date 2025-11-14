from agent_workbench.hitl import GLOBAL_APPROVAL_STORE


def test_hitl_store_create_approve_reject():
    store = GLOBAL_APPROVAL_STORE
    item = store.create("fs.write", "mutation")
    assert item.status == "pending"
    assert store.approve(item.id) is True
    assert store.get(item.id).status == "approved"
    item2 = store.create("web.fetch", "network")
    assert store.reject(item2.id) is True
    assert store.get(item2.id).status == "rejected"
