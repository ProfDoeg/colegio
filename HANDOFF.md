# Packagify handoff — finishing `colegio_tools.py` → the `colegio` package

Porting the 1,499-line monolith `colegio_tools.py` (in the `Colegio_Invisible`
repo) into this clean `colegio` package, **swapping `cryptos` → `pydoge`**, each
module tested. This doc is self-contained so a fresh context (or a resident
Claude on another machine) can finish it.

## Done ✅
- `colegio/node.py` — RPC client (rpc_request, current_block_height, unspent,
  only_conf, add_address_to_watch, get_address_utxos) + MEMPOOL_ANCESTOR_LIMIT.
- `colegio/keys.py` — keyfile/pubkey/addr I/O, make_qr, gen_save_keys_addr
  (uses `pydoge.pubtoaddr`). No cryptos.
- `colegio/imaging.py` — the image bit-codec + `read_image_data`. No cryptos.
- `tests/test_pydoge_vs_cryptos.py` — proves pydoge ≡ cryptos byte-for-byte.

## Remaining modules (port in this order)

### 1. `colegio/inscriptions.py`  ← colegio_tools.py lines ~141-478  (MONEY CODE)
Port: the `STATE_*` constants (141-153), `mk_opreturn` (159-189), and classes
`Cadena`, `CadenaMulti`, `CadenaAtom`, `CadenaMultiAtom` (195-478), plus
`_txid_of_serial` (search for it ~line 480). Uses `node.rpc_request` for
broadcast. **This is the diamond's foundation — port carefully, test hardest.**

### 2. `colegio/crypto.py`  ← colegio_tools.py lines ~999-1373
ECIES/AES box crypto: `shared_key`, `get_txn_pub_from_node`, `_strip_pub_prefix`,
`get_address_pubkeys`, `_parse_multisig_redeem`, `array_dec_from_txn`, the `aes_*`
helpers, `build/read_aes_sealed_quipu`, `build/read_broadcast_quipu`, and the
keydrop functions. Uses ecies/pycryptodome (keep) + `node` + a little cryptos.

### 3. `colegio/reading.py`  ← colegio_tools.py lines ~502-901
The pre-scan reader: `get_all_transactions`, `extract_op_return`,
`_process_transaction_row`, `scan_accounts`, `_build_spender_index_rpc`,
`outputs_walk_index`, `read_strand`, `read_quipu`, `fetch_quipu_bytes`,
`identify_quipus`, `find_quipu_roots`, `find_pre_funded_quipu_roots`. Uses
`node.rpc_request` + pandas. Minimal/no cryptos.

## The cryptos → pydoge map (worked out; apply mechanically)

| cryptos (old)                              | pydoge (new)                                        |
|--------------------------------------------|-----------------------------------------------------|
| `cryptos.Doge()` object                    | (none — call pydoge functions directly)             |
| `doge.mktx(ins, outs)`                     | `pydoge.mktx(ins, outs)`                            |
| `cryptos.serialize` / `deserialize`        | `pydoge.serialize` / `pydoge.deserialize`          |
| `doge.signall(tx, prv)`                    | `pydoge.signall(tx, prv)`  (default uncompressed)   |
| `doge.privtoaddr(prv)`                     | `pydoge.privtoaddr(prv, compressed=False)`          |
| `doge.privtopub(prv)`                      | `pydoge.privtopub(prv, compressed=False)`           |
| `doge.mk_multisig_address(*pubs, num_required=n)` → `(script, addr)` | `script=pydoge.mk_multisig_script(pubs, n)`; `addr=pydoge.mk_multisig_address(pubs, n)` |
| `doge.multisign(tx, i, script, pk)`        | `pydoge.multisign(tx, i, script, pk)`               |
| `cryptos.apply_multisignatures(tx,0,script,*sigs)` | `pydoge.apply_multisignatures(tx,0,script, sigs)` (LIST, not *args) |
| `cryptos.compress` / `privtopub` / `random_key` | `pydoge.compress` / `privtopub` / `random_key` |
| `cryptos.bin_hash160(b)`                   | `pydoge.hash160(b)`                                 |
| `cryptos.bin_to_b58check(data, magic)`     | `pydoge.b58check_encode(magic, data)`  (ARG ORDER!) |
| `cryptos.Bitcoin().pubtoaddr(pub)`         | `pydoge.pubtoaddr(pub, version=0x00)`               |
| `_txid_of_serial(hex)`                     | `pydoge.txhash(hex)`                                |
| `from_int_to_byte(n)`                      | `bytes([n])`                                        |
| `safe_hexlify(b)`                          | `b.hex()`                                           |
| `from cryptos.py3specials import ...`      | (drop — use stdlib)                                 |

Note `mk_opreturn`: pydoge has `mk_opreturn(data)->scripthex`. Keep the old
rawtx-append behavior using `pydoge.deserialize/serialize`.

## Verification (the gates — non-negotiable for money code)
- Each module imports cleanly and its public names match colegio_tools.
- **Byte-identity:** extend `tests/test_pydoge_vs_cryptos.py` — for a `Cadena`
  built+signed via the new `inscriptions` vs the old cryptos path, assert the
  serialized signed tx is identical. The on-chain corpus
  (`Colegio_Invisible/data/bodies/*.bin`, and real knot/join txs) is the oracle.
- `pytest` green. Run via a venv with `pydoge` (editable), `cryptos==2.0.9`,
  `coincurve`, `pytest`, `numpy`, `pandas`, `ecies`, `pycryptodome`, `eth-keys`.

## Backward-compat (optional, later)
A thin `colegio_tools` shim that re-exports from `colegio.*` keeps the old
monorepo's notebooks/scripts working. Not required for the package itself.
