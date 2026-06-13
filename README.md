# colegio

The **Colegio Invisible** quipu toolkit, rebuilt as a clean package — read,
write, and render *quipu*: multi-strand inscriptions on the Dogecoin blockchain.

This is the rebuild. The original lives in
[Colegio_Invisible](https://github.com/ProfDoeg/Colegio_Invisible) — a monorepo
that grew a course, a book, dancer tooling, and the production code all in one
place. `colegio` is just the production code, as an installable package, on a
clean foundation:

```
  colegio   (this package — quipu: read · write · render · sale)
     │ imports
     ▼
  pydoge    (owned Dogecoin-tx layer, replaces the fragile cryptos dependency)
     │ imports
     ▼
  coincurve (libsecp256k1 ECDSA)
```

It carries no book and no dancer tooling — those stay in the original repo.

## Status

**Early — the rebuild is in progress.** The first thing proven here is the
foundation swap: that **pydoge reproduces `cryptos` byte-for-byte** for every
operation the toolkit uses (see `tests/test_pydoge_vs_cryptos.py`). `cryptos` is
a test-only dependency for that migration cross-check; the package itself depends
only on `pydoge`.

## License

MIT. See [`LICENSE`](LICENSE).
