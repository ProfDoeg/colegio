"""tags — chain-state of a quipu root's auxiliary outputs (the node's own datum).

A quipu root's outputs are either strands (their first spend carries an
OP_RETURN — the strand's first knot) or tags (unspent, or spent by a tx with no
OP_RETURN — a deliberate event: a sale claim, a tripwire, a release). The tag
state is pure chain-data — the edition / correction-thread status of a textile,
the one thing only the node knows. The classifier is index-agnostic: spend_of
and get_tx are injected, so the same logic runs over RPC now and the fork's
spentindex later (it ports straight into C++ unchanged in shape).
"""

import pydoge


def _has_op_return(tx):
    return any(o.get("script", "").startswith("6a") for o in tx.get("outs", []))


def classify_root_outputs(root_tx, spend_of, get_tx):
    """Classify every output of a quipu root as strand or tag.

    root_tx   root transaction (pydoge dict with "outs"), or raw hex
    spend_of  callable(txid, vout) -> spending txid or None (None = unspent)
    get_tx    callable(txid) -> tx dict with "outs" (or raw hex)

    A STRAND output's spending tx carries an OP_RETURN (it is the strand's first
    knot). A TAG output is unspent, or spent by a transaction with no OP_RETURN
    (the tag's event — specialization, claim, trip, release).

    Returns one entry per output:
      {"vout", "value", "script", "kind": "strand"|"tag",
       "spent_by": txid|None, "state": "strand"|"intact"|"SPENT"}
    """
    if isinstance(root_tx, str):
        root_tx = pydoge.deserialize(root_tx)
    root_txid = pydoge.txhash(pydoge.serialize(root_tx))
    out = []
    for i, o in enumerate(root_tx["outs"]):
        spender = spend_of(root_txid, i)
        if spender is None:
            out.append({"vout": i, "value": o["value"], "script": o["script"],
                        "kind": "tag", "spent_by": None, "state": "intact"})
            continue
        stx = get_tx(spender)
        if isinstance(stx, str):
            stx = pydoge.deserialize(stx)
        if _has_op_return(stx):
            out.append({"vout": i, "value": o["value"], "script": o["script"],
                        "kind": "strand", "spent_by": spender, "state": "strand"})
        else:
            out.append({"vout": i, "value": o["value"], "script": o["script"],
                        "kind": "tag", "spent_by": spender, "state": "SPENT"})
    return out


def find_tags(root_tx, spend_of, get_tx):
    """Just the tags from classify_root_outputs."""
    return [c for c in classify_root_outputs(root_tx, spend_of, get_tx)
            if c["kind"] == "tag"]
