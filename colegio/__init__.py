"""colegio — the Colegio Invisible quipu toolkit, rebuilt on pydoge.

Read, write, and render quipu (multi-strand OP_RETURN inscriptions on Dogecoin).
Built on the owned `pydoge` transaction layer instead of the fragile cryptos
dependency.

Planned modules (ported incrementally from the original Colegio_Invisible):
    node          — RPC client to a (forked) dogecoind
    reading       — scan / read / walk quipu strands
    inscriptions  — Cadena writers, the diamond (builds on pydoge)
    crypto        — ECIES box crypto, AES seal, keydrop (eciespy/pycryptodome)
    keys          — cinv keyfile loading (eth-keys)
    imaging       — the image bit-codec

See README.md. The first thing proven (tests/) is the pydoge↔cryptos byte
equivalence — the foundation swap.
"""

__version__ = "0.0.1"
