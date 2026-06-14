"""Groundtruth gate: colegio ≡ canonical Colegio_Invisible ≡ on-chain bytes.

Reconstructs the dataframes from the cached on-chain scan (data/cache/rawtx.jsonl
== df_tx) and, for every quipu in the canonical catalog (data/quipu_data.csv),
asserts the colegio read path reproduces the chain byte-for-byte and agrees with
the monolith. Skips cleanly when the sibling Colegio_Invisible repo isn't present
(so the package stays hermetic elsewhere). Offline — no node required.
"""
import csv
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CINV = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
_CACHE = os.path.join(_CINV, "data", "cache", "rawtx.jsonl")
_CATALOG = os.path.join(_CINV, "data", "quipu_data.csv")


@pytest.fixture(scope="module")
def corpus():
    if not (os.path.exists(_CACHE) and os.path.exists(_CATALOG)):
        pytest.skip("canonical Colegio_Invisible corpus not present")
    pd = pytest.importorskip("pandas")
    if _CINV not in sys.path:
        sys.path.insert(0, _CINV)
    try:
        import colegio_tools as old
    except Exception as exc:
        pytest.skip(f"monolith colegio_tools not importable: {exc}")
    from colegio import reading

    recs = [json.loads(line) for line in open(_CACHE)]
    df_tx = pd.DataFrame(recs)
    out_rows = []
    for r in recs:
        out_rows += reading._process_transaction_row(r)
    df_out = pd.DataFrame(out_rows)
    spender = {}
    for r in recs:
        for inp in r["inputs"]:
            spender[inp] = r["txid"]
    df_out["spent_in"] = df_out["txout"].map(spender)
    cat = list(csv.DictReader(open(_CATALOG)))
    return {"df_tx": df_tx, "df_out": df_out, "cat": cat, "old": old, "reading": reading}


def test_identify_quipus_matches_catalog_and_monolith(corpus):
    ids = set(corpus["reading"].identify_quipus(corpus["df_tx"], corpus["df_out"]))
    old_ids = set(corpus["old"].identify_quipus(corpus["df_tx"], corpus["df_out"]))
    assert ids == old_ids                                       # code parity
    cat_roots = set(r["root_txid"] for r in corpus["cat"])
    missing = cat_roots - ids
    assert not missing, f"catalog roots not identified: {sorted(missing)}"


def test_read_path_byte_exact_vs_chain_and_catalog(corpus):
    from colegio import envelope
    reading, df_out = corpus["reading"], corpus["df_out"]
    for r in corpus["cat"]:
        root = r["root_txid"]
        h, b = reading.read_quipu(root, df_out)
        assembled = bytes.fromhex(h + b)
        body_path = os.path.join(_CINV, "data", r["body_file"])
        assert assembled == open(body_path, "rb").read(), f"{root}: body != on-chain"
        if r["total_bytes"]:
            assert len(assembled) == int(r["total_bytes"]), f"{root}: length"
        env = envelope.parse_envelope(bytes.fromhex(h))
        assert env["type"] == int(r["type_byte"], 16), f"{root}: type"
        assert env["tone"] == int(r["tone"], 16), f"{root}: tone"


def test_read_quipu_equals_monolith_on_real_corpus(corpus):
    reading, old, df_out = corpus["reading"], corpus["old"], corpus["df_out"]
    for r in corpus["cat"]:
        root = r["root_txid"]
        assert reading.read_quipu(root, df_out) == old.read_quipu(root, df_out), root
