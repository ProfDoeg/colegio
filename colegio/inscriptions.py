"""inscriptions — the Cadena writers, the diamond's foundation.

Splits arbitrary bytes into 80-byte OP_RETURN chunks and posts them as a chain
of self-spending transactions (a "strand"). Two key modes — single-key
(`Cadena`) and m-of-n multisig (`CadenaMulti`) — each with a step-by-step
node-backed variant and an atomic precompute-then-broadcast variant.

This is the money code: it builds and signs real spends. It is built on the
owned `pydoge` transaction layer (swapped in for the fragile `cryptos`
dependency) and is proven byte-identical to the old cryptos path in tests/.

Lifecycle:
    Cadena / CadenaMulti (step-by-step)
        CONF → make_tx() → READY → broadcast() → SENT → update() → CONF/DONE
    CadenaAtom / CadenaMultiAtom (atomic)
        INIT → precompute() → PRECOMPUTED → broadcast() → BROADCAST
             → confirm() → CONFIRMED
"""

import time

import pydoge

from . import node

# --- State constants --------------------------------------------------------
# Step-by-step lifecycle (Cadena, CadenaMulti):
STATE_CONF  = "CONF"   # last broadcast tx is confirmed (or no broadcast yet)
STATE_READY = "READY"  # next tx is built & signed, awaiting broadcast
STATE_SENT  = "SENT"   # broadcast, awaiting confirmation
STATE_DONE  = "DONE"   # full chain complete

# Atomic lifecycle (CadenaAtom, CadenaMultiAtom):
STATE_INIT        = "INIT"         # constructed, nothing computed yet
STATE_PRECOMPUTED = "PRECOMPUTED"  # all txs built and signed, none broadcast
STATE_BROADCAST   = "BROADCAST"    # all txs in mempool, none yet confirmed
STATE_CONFIRMED   = "CONFIRMED"    # final tx in chain has at least 1 confirmation


# --- OP_RETURN primitive ----------------------------------------------------
def mk_opreturn(msg, rawtx=None, json_out=False):
    """Build an OP_RETURN script hex (or attach it to an existing rawtx).

    msg: bytes or str — the data to embed.
    rawtx: optional hex tx; if given, returns a new rawtx with the OP_RETURN
           output appended.
    json_out: if True and rawtx is None, returns {'script': ..., 'value': 0}.

    The script itself comes from pydoge.mk_opreturn; the rawtx-append behavior
    is kept here (the diamond builds strands by appending an OP_RETURN to a
    freshly made spend, then signing).
    """
    msg_bytes = msg if isinstance(msg, bytes) else msg.encode()
    orhex = pydoge.mk_opreturn(msg_bytes)
    orjson = {"script": orhex, "value": 0}
    if rawtx is not None:
        txo = pydoge.deserialize(rawtx)
        if "outs" not in txo:
            raise ValueError("OP_RETURN cannot be the sole output")
        txo["outs"].append(orjson)
        return pydoge.serialize(txo)
    return orjson if json_out else orhex


# --- Cadena (single-key, node-backed) ---------------------------------------
class Cadena:
    """A single-strand inscription chain.

    Splits `data` into 80-byte chunks and posts them in a chain of
    self-spending OP_RETURN transactions.
    """

    def __init__(self, prvkey, data, utxo_dct, tip):
        self.data = data
        self.clip = [data[i:i + 80] for i in range(0, len(data), 80)]
        self.og_len = len(self.clip)
        self.state = STATE_CONF
        self.utxo = utxo_dct
        self.head_utxo = utxo_dct
        self.txn_ids = [utxo_dct["output"].split(":")[0]]
        self.prv = prvkey
        self.addr = pydoge.privtoaddr(prvkey, compressed=False)
        self.tip = tip
        self.index = 0
        self.signed_inscribed_tx = None

    def make_tx(self):
        tx = pydoge.mktx(
            [self.head_utxo],
            [{"value": self.head_utxo["value"] - self.tip, "address": self.addr}],
        )
        serial = mk_opreturn(self.clip[self.index], pydoge.serialize(tx))
        inscribed = pydoge.deserialize(serial)
        self.signed_inscribed_tx = pydoge.signall(inscribed, self.prv)
        self.state = STATE_READY

    def broadcast(self):
        raw_hex = pydoge.serialize(self.signed_inscribed_tx)
        cast_txid = node.rpc_request("sendrawtransaction", [raw_hex])
        self.txn_ids.append(cast_txid)
        self.head_utxo = {
            "output": f"{cast_txid}:0",
            "value": self.head_utxo["value"] - self.tip,
        }
        self.index += 1
        self.state = STATE_SENT

    def update(self):
        txid = self.head_utxo["output"].split(":")[0]
        info = node.rpc_request("gettransaction", [txid])
        if info.get("confirmations", 0) > 0:
            self.state = STATE_CONF
            if self.index == self.og_len:
                self.state = STATE_DONE


# --- CadenaMulti (multisig, node-backed) ------------------------------------
class CadenaMulti:
    """Multisig variant of Cadena. All listed private keys must sign each tx."""

    DOGE_P2SH_MAGIC = 22  # Dogecoin mainnet P2SH version byte

    def __init__(self, prvkeys, data, utxo_dct, tip):
        self.data = data
        self.clip = [data[i:i + 80] for i in range(0, len(data), 80)]
        self.og_len = len(self.clip)
        self.state = STATE_CONF
        self.utxo = utxo_dct
        self.head_utxo = utxo_dct
        self.txn_ids = [utxo_dct["output"].split(":")[0]]
        self.prvs = prvkeys
        self.pubs = [pydoge.privtopub(p, compressed=False) for p in prvkeys]
        self.script = pydoge.mk_multisig_script(self.pubs, len(self.pubs))
        self.addr = pydoge.mk_multisig_address(
            self.pubs, len(self.pubs), version=self.DOGE_P2SH_MAGIC
        )
        self.tip = tip
        self.index = 0
        self.signed_inscribed_tx = None

    def make_tx(self):
        tx = pydoge.mktx(
            [self.head_utxo],
            [{"value": self.head_utxo["value"] - self.tip, "address": self.addr}],
        )
        serial = mk_opreturn(self.clip[self.index], pydoge.serialize(tx))
        inscribed = pydoge.deserialize(serial)
        sigs = [
            pydoge.multisign(inscribed, 0, self.script, prv) for prv in self.prvs
        ]
        self.signed_inscribed_tx = pydoge.apply_multisignatures(
            inscribed, 0, self.script, sigs
        )
        self.state = STATE_READY

    def broadcast(self):
        raw_hex = pydoge.serialize(self.signed_inscribed_tx)
        cast_txid = node.rpc_request("sendrawtransaction", [raw_hex])
        self.txn_ids.append(cast_txid)
        self.head_utxo = {
            "output": f"{cast_txid}:0",
            "value": self.head_utxo["value"] - self.tip,
        }
        self.index += 1
        self.state = STATE_SENT

    def update(self):
        txid = self.head_utxo["output"].split(":")[0]
        info = node.rpc_request("gettransaction", [txid])
        if info.get("confirmations", 0) > 0:
            self.state = STATE_CONF
            if self.index == self.og_len:
                self.state = STATE_DONE


# --- CadenaAtom (single-key, precomputed) -----------------------------------
# precompute() builds and signs all N transactions in memory. After it runs,
# every txid in the strand is knowable — including the final tail txid —
# without anything yet on chain. broadcast() pushes them all in dependency
# order; for chains over the mempool ancestor limit it auto-waves (broadcast,
# wait for confirmation, repeat). Use this when you want the strand to either
# fully exist or fully not exist on chain — the bordado mode of inscription.
class CadenaAtom:
    """Atomic single-key strand. Build all transactions in memory first,
    then broadcast in one operation."""

    def __init__(self, prvkey, data, utxo_dct, tip):
        self.data = data
        self.clip = [data[i:i + 80] for i in range(0, len(data), 80)]
        self.og_len = len(self.clip)
        self.utxo = utxo_dct
        self.prv = prvkey
        self.addr = pydoge.privtoaddr(prvkey, compressed=False)
        self.tip = tip
        self.state = STATE_INIT
        self.txns = []
        self.txn_ids = []

    def precompute(self):
        """Build and sign every transaction in the strand. No network calls."""
        head_utxo = dict(self.utxo)
        for op_data in self.clip:
            tx = pydoge.mktx(
                [head_utxo],
                [{"value": head_utxo["value"] - self.tip, "address": self.addr}],
            )
            serial = mk_opreturn(op_data, pydoge.serialize(tx))
            inscribed = pydoge.deserialize(serial)
            signed = pydoge.signall(inscribed, self.prv)
            signed_hex = pydoge.serialize(signed)
            txid = pydoge.txhash(signed_hex)
            self.txns.append(signed_hex)
            self.txn_ids.append(txid)
            head_utxo = {
                "output": f"{txid}:0",
                "value": head_utxo["value"] - self.tip,
            }
        self.state = STATE_PRECOMPUTED

    def broadcast(self, wave_size=node.MEMPOOL_ANCESTOR_LIMIT,
                  poll_interval_s=30, max_wait_s=600):
        """Push all precomputed txs to the node in dependency order.

        For strands of <= wave_size, pushes everything at once.
        For longer strands, broadcasts in waves of wave_size with waits
        for confirmation between waves.
        """
        if self.state != STATE_PRECOMPUTED:
            raise RuntimeError(
                f"broadcast() requires PRECOMPUTED state; got {self.state}"
            )
        n = len(self.txns)
        i = 0
        while i < n:
            wave_end = min(i + wave_size, n)
            for j in range(i, wave_end):
                node.rpc_request("sendrawtransaction", [self.txns[j]])
            if wave_end < n:
                self._wait_confirmed(self.txn_ids[wave_end - 1],
                                     poll_interval_s, max_wait_s)
            i = wave_end
        self.state = STATE_BROADCAST

    def confirm(self):
        """Check if the final tx in the strand has at least one confirmation."""
        if not self.txn_ids:
            raise RuntimeError("nothing precomputed")
        info = node.rpc_request("gettransaction", [self.txn_ids[-1]])
        if info.get("confirmations", 0) > 0:
            self.state = STATE_CONFIRMED
            return True
        return False

    @staticmethod
    def _wait_confirmed(txid, poll_interval_s, max_wait_s):
        """Block until txid has >= 1 confirmation, or raise on timeout."""
        elapsed = 0
        while elapsed < max_wait_s:
            try:
                info = node.rpc_request("gettransaction", [txid])
                if info.get("confirmations", 0) > 0:
                    return
            except RuntimeError:
                pass
            time.sleep(poll_interval_s)
            elapsed += poll_interval_s
        raise TimeoutError(
            f"tx {txid[:16]}... did not confirm within {max_wait_s}s"
        )


# --- CadenaMultiAtom (multisig, precomputed) --------------------------------
class CadenaMultiAtom:
    """Atomic multisig strand. All listed private keys must sign each tx."""

    DOGE_P2SH_MAGIC = 22

    def __init__(self, prvkeys, data, utxo_dct, tip):
        self.data = data
        self.clip = [data[i:i + 80] for i in range(0, len(data), 80)]
        self.og_len = len(self.clip)
        self.utxo = utxo_dct
        self.prvs = prvkeys
        self.pubs = [pydoge.privtopub(p, compressed=False) for p in prvkeys]
        self.script = pydoge.mk_multisig_script(self.pubs, len(self.pubs))
        self.addr = pydoge.mk_multisig_address(
            self.pubs, len(self.pubs), version=self.DOGE_P2SH_MAGIC
        )
        self.tip = tip
        self.state = STATE_INIT
        self.txns = []
        self.txn_ids = []

    def precompute(self):
        """Build, multisign, and serialize every tx in the strand."""
        head_utxo = dict(self.utxo)
        for op_data in self.clip:
            tx = pydoge.mktx(
                [head_utxo],
                [{"value": head_utxo["value"] - self.tip, "address": self.addr}],
            )
            serial = mk_opreturn(op_data, pydoge.serialize(tx))
            inscribed = pydoge.deserialize(serial)
            sigs = [
                pydoge.multisign(inscribed, 0, self.script, prv)
                for prv in self.prvs
            ]
            signed = pydoge.apply_multisignatures(inscribed, 0, self.script, sigs)
            signed_hex = pydoge.serialize(signed)
            txid = pydoge.txhash(signed_hex)
            self.txns.append(signed_hex)
            self.txn_ids.append(txid)
            head_utxo = {
                "output": f"{txid}:0",
                "value": head_utxo["value"] - self.tip,
            }
        self.state = STATE_PRECOMPUTED

    def broadcast(self, wave_size=node.MEMPOOL_ANCESTOR_LIMIT,
                  poll_interval_s=30, max_wait_s=600):
        return CadenaAtom.broadcast(self, wave_size, poll_interval_s, max_wait_s)

    def confirm(self):
        return CadenaAtom.confirm(self)
