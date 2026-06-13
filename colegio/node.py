"""node — RPC client to a Dogecoin Core (or quipu-fork) daemon.

Loads RPC credentials from the environment (.env in the working directory).
A module-level session keeps HTTP keep-alive so the reader's thousands of
calls don't exhaust ephemeral ports.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

RPC_USER = os.getenv("RPC_USER", "drdoeg")
RPC_PASSWORD = os.getenv("RPC_PASSWORD", "password")
RPC_HOST = os.getenv("RPC_HOST", "127.0.0.1")
RPC_PORT = int(os.getenv("RPC_PORT", "22555"))
RPC_URL = f"http://{RPC_USER}:{RPC_PASSWORD}@{RPC_HOST}:{RPC_PORT}"

# Mempool policy: Dogecoin Core's default is 25 unconfirmed ancestors per chain.
MEMPOOL_ANCESTOR_LIMIT = 25

_RPC_SESSION = requests.Session()
_RPC_SESSION.headers.update({"content-type": "application/json"})


def rpc_request(method, params=None):
    """Call a Dogecoin Core RPC method. Returns the 'result' field."""
    if params is None:
        params = []
    payload = json.dumps({"method": method, "params": params,
                          "jsonrpc": "2.0", "id": 0})
    response = _RPC_SESSION.post(RPC_URL, data=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"RPC request failed ({response.status_code}): {response.text}")
    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"RPC error: {body['error']}")
    return body["result"]


def current_block_height():
    return rpc_request("getblockcount")


def unspent(address):
    """List unspent outputs for an address in pydoge/cryptos-compatible format."""
    raw = rpc_request("listunspent", [0, 9999999, [address]])
    return [
        {"value": int(out["amount"] * 100_000_000),     # DOGE → satoshis
         "output": f"{out['txid']}:{out['vout']}"}
        for out in raw
    ]


def only_conf(utxos):
    """Keep only utxos whose funding tx has at least one confirmation."""
    out = []
    for u in utxos:
        txid = u["output"].split(":")[0]
        info = rpc_request("gettransaction", [txid])
        if info.get("confirmations", 0) > 0:
            out.append(u)
    return out


def add_address_to_watch(address, label="watch"):
    """Register an address for the wallet to watch — NEVER rescans (idempotent).
    For deep historical scans use scantxoutset from the CLI deliberately."""
    return rpc_request("importaddress", [address, label, False])


def get_address_utxos(address):
    """Current spendable UTXOs at a watched address. Fast, no scan."""
    return rpc_request("listunspent", [0, 9999999, [address]])
