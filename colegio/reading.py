"""reading — the pre-scan quipu reader.

Reads multi-strand quipu off chain by walking self-spending OP_RETURN chains.
Two backends throughout: a fast dataframe walker over a pre-scanned account set
(`scan_accounts` → df_transactions, df_outputs), and an RPC walker that derives
what it needs live from the node. No keys, no signing — pure read path.

Workflow (dataframe mode):
    1. scan_accounts({addr: label, ...})       -> (df_tx, df_out)   slow, do once
    2. find_quipu_roots(addr, df_tx, df_out)    -> [txid, ...]       list quipus
    3. read_quipu(txid, df_out)                 -> (header, body)    decode one

The dataframes can be cached / saved to disk and reloaded later. `read_quipu`
and `fetch_quipu_bytes` also work with no dataframe (RPC mode) for one-off reads.
"""

import pandas as pd

from . import node, envelope, tags


def get_all_transactions(account_name, batch_size=10000):
    """Page through listtransactions for a wallet account label."""
    transactions = []
    offset = 0
    while True:
        batch = node.rpc_request(
            "listtransactions", [account_name, batch_size, offset, True]
        )
        if not batch:
            break
        transactions.extend(batch)
        offset += batch_size
    return transactions


def extract_op_return(vout):
    """Return the OP_RETURN payload hex from a vout dict, or None."""
    if vout.get("scriptPubKey", {}).get("type") != "nulldata":
        return None
    hex_data = vout["scriptPubKey"]["hex"]
    if not hex_data.startswith("6a"):
        return None
    length_byte = int(hex_data[2:4], 16)
    if length_byte <= 75:
        return hex_data[4:4 + length_byte * 2]
    if length_byte == 0x4c:  # OP_PUSHDATA1
        n = int(hex_data[4:6], 16)
        return hex_data[6:6 + n * 2]
    if length_byte == 0x4d:  # OP_PUSHDATA2
        n = int(hex_data[4:8], 16)
        return hex_data[8:8 + n * 2]
    if length_byte == 0x4e:  # OP_PUSHDATA4
        n = int(hex_data[4:12], 16)
        return hex_data[12:12 + n * 2]
    return None


def _process_transaction_row(row):
    """Expand one tx row into one row per output."""
    txid = row["txid"]
    return [
        {
            "txout": f"{txid}:{n}",
            "spent_in": None,
            "value": row["values"][n],
            "op_return": row["op_return"],
            "blockheight": row["blockheight"],
            "blocktime": row["blocktime"],
            "txid": txid,
            "n": n,
        }
        for n in range(row["num_outputs"])
    ]


def scan_accounts(accounts):
    """Build (df_transactions, df_outputs) for a dict of {address: account_label}.

    df_outputs has one row per (txid, vout_n) with 'spent_in' filled in for
    outputs we observed being spent within the same account set.
    """
    all_tx = []
    for _addr, label in accounts.items():
        for tx in get_all_transactions(label):
            tx["account_label"] = label
            all_tx.append(tx)

    detailed = []
    seen = set()
    for tx in all_tx:
        txid = tx["txid"]
        if txid in seen:
            continue
        seen.add(txid)
        # gettransaction (wallet RPC) works in pruned mode for wallet-relevant
        # txs. We decode the hex field to get the same shape getrawtransaction
        # would have returned, then merge in the block metadata from the wallet
        # view. (getrawtransaction would require -txindex, which is incompatible
        # with prune.)
        wallet_tx = node.rpc_request("gettransaction", [txid, True])
        raw = node.rpc_request("decoderawtransaction", [wallet_tx["hex"]])
        raw["blockhash"] = wallet_tx.get("blockhash")
        raw["blocktime"] = wallet_tx.get("blocktime")
        block = node.rpc_request("getblockheader", [raw["blockhash"]])
        last_vout = raw["vout"][-1]
        op = (
            extract_op_return(last_vout)
            if last_vout["scriptPubKey"]["type"] == "nulldata"
            else None
        )
        detailed.append({
            "txid": txid,
            "blockhash": raw["blockhash"],
            "blocktime": raw["blocktime"],
            "blockheight": block["height"],
            "inputs": [f"{vin['txid']}:{vin['vout']}" for vin in raw["vin"]],
            "values": [vo["value"] for vo in raw["vout"]],
            "num_inputs": len(raw["vin"]),
            "num_outputs": len(raw["vout"]),
            "op_return": op,
            "addresses_in_outputs": [
                vo["scriptPubKey"].get("addresses", [])
                for vo in raw["vout"]
            ],
        })

    df_tx = pd.DataFrame(detailed).sort_values(by=["blockheight", "blocktime"])

    output_rows = []
    for _, row in df_tx.iterrows():
        output_rows.extend(_process_transaction_row(row))
    df_out = pd.DataFrame(output_rows).sort_values(by=["blockheight", "blocktime"])

    # Fill spent_in by matching each tx's inputs to existing outputs
    for _, tx in df_tx.iterrows():
        for input_ref in tx["inputs"]:
            mask = df_out["txout"] == input_ref
            if mask.any():
                df_out.loc[mask, "spent_in"] = tx["txid"]

    return df_tx, df_out


def _build_spender_index_rpc(address):
    """For an address loaded in the wallet, build a {txout -> spender_txid} dict
    by scanning the wallet's tx history at that address. Used by read_strand
    when no df_outputs is provided (RPC mode)."""
    spent_map = {}
    # Pull a generous slice of wallet txs (multiple labels possible — listtransactions
    # is filterable by label/account, but we just grab all and filter by address).
    txs = node.rpc_request("listtransactions", ["*", 10000, 0, True])
    seen_txids = set()
    for entry in txs:
        if entry.get("address") != address:
            continue
        txid = entry["txid"]
        if txid in seen_txids:
            continue
        seen_txids.add(txid)
        raw = node.rpc_request("getrawtransaction", [txid, 1])
        for vin in raw.get("vin", []):
            prev = vin.get("txid")
            if prev is None:
                continue
            spent_map[f"{prev}:{vin['vout']}"] = txid
    return spent_map


def outputs_walk_index(df_outputs):
    """O(1) lookup maps for the dataframe walkers, memoized on df.attrs.

    Returns {"txout": {txout -> (spent_in, op_return_of_that_row)},
             "txid_op": {txid -> op_return}}.

    Built once per dataframe (O(n)) and cached in df.attrs; every walker
    lookup is then a dict hit instead of a full-frame boolean scan — the
    difference between minutes and hours once the frame holds ~10^5 rows.
    The cache is keyed on len(df); if you mutate spent_in/op_return in
    place WITHOUT changing the row count, delete
    df.attrs["_quipu_walk_index"] to force a rebuild."""
    cached = df_outputs.attrs.get("_quipu_walk_index")
    if cached is not None and cached["n"] == len(df_outputs):
        return cached
    txout_map, txid_op = {}, {}
    for txout, spent_in, op, txid in zip(df_outputs["txout"].to_numpy(),
                                         df_outputs["spent_in"].to_numpy(),
                                         df_outputs["op_return"].to_numpy(),
                                         df_outputs["txid"].to_numpy()):
        txout_map[txout] = (spent_in, op)
        if txid not in txid_op:
            txid_op[txid] = op
    cached = {"n": len(df_outputs), "txout": txout_map, "txid_op": txid_op}
    df_outputs.attrs["_quipu_walk_index"] = cached
    return cached


def read_strand(txout, df_outputs=None, spender_map=None):
    """Walk a strand iteratively, collecting OP_RETURN payload bytes along the way.

    Two backends:
      - If df_outputs is given, walk using the pre-scanned dataframe (fast, bulk).
      - Otherwise walk via RPC. spender_map ({txout -> spender_txid}) must be
        provided; build it once per address with _build_spender_index_rpc.

    Returns hex-string of concatenated OP_RETURN payloads (empty if no strand)."""
    idx = outputs_walk_index(df_outputs) if df_outputs is not None else None
    out = ""
    cur = txout
    while True:
        if idx is not None:
            hit = idx["txout"].get(cur)
            if hit is None: return out
            spend_tx = hit[0]
            if not spend_tx: return out
            spend_hit = idx["txout"].get(f"{spend_tx}:0")
            if spend_hit is None: return out
            op_data = spend_hit[1]
            if not op_data: return out
        else:
            spend_tx = (spender_map or {}).get(cur)
            if spend_tx is None: return out
            raw = node.rpc_request("getrawtransaction", [spend_tx, 1])
            op_data = None
            for v in raw.get("vout", []):
                d = extract_op_return(v)
                if d:
                    op_data = d
                    break
            if not op_data: return out
        out += op_data
        cur = f"{spend_tx}:0"


def read_quipu(tx, df_outputs=None):
    """Read a multi-strand quipu from its root txid. Strand 0 is the header
    (cabeza); strands 1..N are body chunks (cuerpos), concatenated in order.

    Two backends, auto-selected:
      - If df_outputs is given → dataframe walker (fast for bulk reads).
      - Otherwise → RPC walker. Derives the address from the root tx's first
        output, builds a spender index once, then walks each strand."""
    if df_outputs is None:
        # RPC mode — derive address from root tx, build spender map once
        root = node.rpc_request("getrawtransaction", [tx, 1])
        first_out = root["vout"][0]
        addrs = first_out.get("scriptPubKey", {}).get("addresses", [])
        if not addrs:
            raise ValueError(f"can't derive address from {tx} output 0")
        address = addrs[0]
        spender_map = _build_spender_index_rpc(address)
        kwargs = {"spender_map": spender_map}
    else:
        kwargs = {"df_outputs": df_outputs}

    header = read_strand(f"{tx}:0", **kwargs)
    body_parts = []
    idx = 1
    while True:
        strand = read_strand(f"{tx}:{idx}", **kwargs)
        if strand == "":
            break
        body_parts.append(strand)
        idx += 1
    return header, "".join(body_parts)


def fetch_quipu_bytes(txid, max_walk=64):
    """Fetch the concatenated header+body bytes of any quipu given its root
    OR its join txid.

    A diamond's root has N≥2 outputs (one per strand) and NO OP_RETURN
    outputs (its outputs fund strand starters). Strand txs carry the
    OP_RETURN-bearing knots; the join has 1 regular output collecting
    every strand terminus. Given a join, walks back via first input
    until a tx with ≥2 outputs and no OP_RETURNs is found — that's the
    root. Then defers to read_quipu to walk forward through all strands.

    Args:
        txid: hex string — either a quipu's root or its join txid
        max_walk: safety bound on the back-walk depth

    Returns:
        bytes — concatenated header + body, suitable for resolve_ref's
        fetcher contract and for the canonical readers.
    """
    def _looks_like_root(tx):
        vout = tx.get("vout", [])
        if len(vout) < 2:
            return False
        for o in vout:
            spk = o.get("scriptPubKey", {})
            asm = spk.get("asm", "")
            if asm.startswith("OP_RETURN"):
                return False
        return True

    tx = node.rpc_request("getrawtransaction", [txid, 1])
    if _looks_like_root(tx):
        root = txid
    else:
        cur = txid
        root = None
        for _ in range(max_walk):
            t = node.rpc_request("getrawtransaction", [cur, 1])
            if _looks_like_root(t):
                root = cur
                break
            vin = t.get("vin", [])
            if not vin or "txid" not in vin[0]:
                raise ValueError(f"hit a coinbase or malformed tx walking back from {txid}")
            cur = vin[0]["txid"]
        if root is None:
            raise ValueError(
                f"could not find diamond root within {max_walk} hops from {txid} "
                f"(is this really a quipu join/root?)"
            )

    header_hex, body_hex = read_quipu(root)
    return bytes.fromhex(header_hex + body_hex)


def quipuread(txid, df_outputs=None):
    """The read contract: {header, body, tags} for a quipu given its root txid.

    This is the Python reference for the fork's native `quipuread` RPC — the
    executable spec and the differential oracle. Returns:
      - header: the universal envelope {magic, version, type, tone, raw} — the
        node parses only these; the client decodes title/fields per type.
      - body:   the assembled body-strand bytes (hex), opaque to the node.
      - tags:   chain-state of the root's tag outputs (see tags.classify).

    df_outputs (a pre-scanned frame) accelerates the strand walk and the
    spend lookups; without it everything is derived live from the node.
    """
    header_hex, body_hex = read_quipu(txid, df_outputs)
    header = envelope.parse_envelope(bytes.fromhex(header_hex))

    # tags need the root tx's output scripts/values (not in the frame) — one
    # call — plus the spend-state of each output.
    root_hex = node.rpc_request("getrawtransaction", [txid, 0])
    if df_outputs is not None:
        idx = outputs_walk_index(df_outputs)

        def spend_of(t, v):
            hit = idx["txout"].get(f"{t}:{v}")
            return hit[0] if hit and hit[0] else None

        def get_tx(spender):
            # only _has_op_return is consulted; a 1-output shim suffices
            return {"outs": [{"script": "6a"}]} if idx["txid_op"].get(spender) \
                else {"outs": []}
    else:
        root_v = node.rpc_request("getrawtransaction", [txid, 1])
        addrs = root_v["vout"][0].get("scriptPubKey", {}).get("addresses", [])
        smap = _build_spender_index_rpc(addrs[0]) if addrs else {}

        def spend_of(t, v):
            return smap.get(f"{t}:{v}")

        def get_tx(spender):
            return node.rpc_request("getrawtransaction", [spender, 0])

    classified = tags.classify_root_outputs(root_hex, spend_of, get_tx)
    return {
        "header": header,
        "body": body_hex,
        "tags": [c for c in classified if c["kind"] == "tag"],
    }


def identify_quipus(df_transactions, df_outputs):
    """Return quipu-root txids. The criterion (per the author):

        a root transaction carries no OP_RETURN of its own, and the
        spend of its 0th output carries the c1dd magic.

    Output :0 starts the cabeza strand, so its spender's OP_RETURN is the
    first header knot — the magic bytes themselves. Nothing else in a
    wallet's graph has this shape: knots carry their own OP_RETURN,
    splitters/joins feed txs without one, and mid-strand knots never
    spend a :0 that begins a header.

    Two earlier criteria proved wrong in turn:
      - "every output spent by a tx with an OP_RETURN" — rejects 2022-era
        roots (e.g. the 1ec0… certificate node), whose change outputs are
        spent by ordinary txs;
      - "every output spent by an in-frame tx" — rejects any root with a
        tag output (quipu_tags vout≥1 convention), where UNSPENT is the
        steady state meaning 'current edition' (the 34316f64… healing
        catalog was invisible under this rule)."""
    idx = outputs_walk_index(df_outputs)
    if "op_return" in df_transactions.columns:
        own_op = df_transactions["op_return"].fillna("").to_numpy()
    else:
        own_op = [""] * len(df_transactions)
    results = []
    for txid, op in zip(df_transactions["txid"].to_numpy(), own_op):
        if op:
            continue
        hit = idx["txout"].get(f"{txid}:0")
        if hit is None or not hit[0]:
            continue
        spender_op = idx["txid_op"].get(hit[0]) or ""
        if spender_op.startswith("c1dd"):
            results.append(txid)
    return results


def find_quipu_roots(address, df_transactions, df_outputs):
    """Return quipu-root txids whose first output pays `address`.

    A "quipu root" is a transaction with N outputs, each of which is then
    spent in a tx that carries an OP_RETURN. This is the test in
    identify_quipus(); find_quipu_roots adds an address filter so you can
    ask "which quipus did this wallet originate or hold?"
    """
    all_roots = identify_quipus(df_transactions, df_outputs)
    out = []
    for txid in all_roots:
        rows = df_transactions[df_transactions["txid"] == txid]
        if rows.empty:
            continue
        addrs_per_out = rows.iloc[0]["addresses_in_outputs"]
        # Match if the address appears in any output
        if any(address in addrs for addrs in addrs_per_out):
            out.append(txid)
    return out


def find_pre_funded_quipu_roots(address, df_transactions, df_outputs):
    """Find candidate quipu-root txs that haven't been inscribed yet —
    "broomhead" roots ready to write to. Heuristic:
      - ≥2 outputs to `address`
      - all those outputs currently unspent (spent_in is null in df_outputs)
      - tx itself isn't a quipu-strand step (no OP_RETURN of its own)

    These show up as quipu nodes with N unspent tendrils in the topology
    view, distinct from fully-inscribed quipus (find_quipu_roots).
    """
    out = []
    for _, tx in df_transactions.iterrows():
        txid = tx["txid"]
        # Skip if the tx itself carries an OP_RETURN (it's a strand step,
        # not a root)
        if tx.get("op_return"):
            continue
        addrs_per_out = tx["addresses_in_outputs"]
        out_indices = [
            i for i, addrs in enumerate(addrs_per_out) if address in addrs
        ]
        if len(out_indices) < 2:
            continue
        all_unspent = True
        for i in out_indices:
            rows = df_outputs[df_outputs["txout"] == f"{txid}:{i}"]
            if rows.empty:
                all_unspent = False
                break
            sp = rows.iloc[0]["spent_in"]
            if sp and not (isinstance(sp, float) and sp != sp):
                all_unspent = False
                break
        if all_unspent:
            out.append(txid)
    return out
