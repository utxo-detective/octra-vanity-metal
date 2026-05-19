# octra-vanity-metal

Mac-native Octra vanity address miner. GPU brute-forces ed25519 keypairs on
Apple Silicon (and Intel Macs with a discrete GPU) via Metal, looking for
addresses that match a pattern you pick.

This is a Metal port of the CUDA implementation at
[0x02937/octra_vanity](https://github.com/0x02937/octra_vanity).
Algorithm is identical (precomputed-window scalar mult, truncated SHA-512,
32-bit base58 matchers, etc.) — just retargeted from `__device__` CUDA C
to MSL kernels and a Swift host. **One real bug from the original is fixed
here** (see [Correctness fix](#correctness-fix) below) — wallets produced by
this miner re-derive correctly through standard ed25519, which is not always
true of the CUDA reference.

```
oct Utxo YZt1UhJPUUSv3SDSqo1YtMfjVk7cgDFrrrjPgoQ
    ^^^^
    you pick this part
```

---

## Requirements

- macOS 12+ (Monterey or newer)
- Xcode command-line tools (for `metal`, `metallib`, and `swiftc`)

```
xcode-select --install
```

No CUDA, no NVIDIA card, no nightmares.

---

## Build

```
make
```

That produces `build/octra_vanity_metal` and `build/default.metallib`.

---

## Usage

### Interactive

Just run it with no arguments and answer the prompts:

```
./build/octra_vanity_metal
```

### Command-line

Non-interactive — the friendly path if you're scripting or impatient:

```
# 5-char prefix, octHELLO...
./build/octra_vanity_metal --prefix HELLO

# Match anywhere, case-insensitive
./build/octra_vanity_metal --anywhere DEGEN -i

# Suffix
./build/octra_vanity_metal --suffix END

# Repeating chars
./build/octra_vanity_metal --rep-start 5     # octXXXXX...
./build/octra_vanity_metal --rep-end   5     # ...XXXXX
./build/octra_vanity_metal --rep-any   6     # 6 same chars anywhere

# Throttle so the rest of your Mac stays responsive (50% GPU duty cycle)
./build/octra_vanity_metal --prefix ABC --gpu-budget 50 --show-gpu

# Save to a non-default RPC
./build/octra_vanity_metal --prefix XYZ --rpc https://my.octra.rpc/
```

All flags:

```
--prefix <pat>       --suffix <pat>     --anywhere <pat>
--rep-start <N>      --rep-end <N>      --rep-any <N>
-i / --case-insensitive
--rpc <url>
--bonus <N>:<mode>   (also save bonus repeating finds — mode 1/2/3 = start/end/anywhere)

--threadgroups <N>   --threads <N>   --iters <N>   --no-auto-tune
--gpu-budget <pct>   keep GPU at roughly pct% duty cycle (1..100)
--show-gpu           print live GPU utilisation in the progress line

-h / --help
```

---

## How fast

Loose numbers on my M3 Max running unthrottled:

| Kernel       | Throughput   |
|--------------|--------------|
| prefix/anywhere/rep-start/rep-any | **~1.6 GK/s**  |
| suffix       | ~50–80 MK/s  |
| rep-end      | ~50–80 MK/s  |

Suffix and rep-end need to fully base58-encode each candidate to look at the
end of the address, so they're an order of magnitude slower than the
prefix-style kernels which can give up after just a few digits of the divide.
This matches the original CUDA implementation's behaviour.

Difficulty doubles roughly per extra character. Rough times on M3 Max:

| Pattern              | Expected time |
|----------------------|---------------|
| 4-char prefix        | < 1 second    |
| 5-char prefix        | ~10 seconds   |
| 6-char prefix        | ~5 minutes    |
| 7-char prefix        | ~5 hours      |
| 5 repeating start    | < 1 second    |
| 7 repeating start    | ~5 minutes    |

---

## Forbidden / unusual patterns

Base58 doesn't use `0`, `O`, `I`, `l` — the miner rejects any pattern that
contains those.

There's also a subtle constraint on **leading** characters: a SHA-256 digest
treated as a 256-bit integer is large enough to need 44 base58 digits ~94%
of the time, and in that case its first character is forced into the first
18 chars of the alphabet:

```
1 2 3 4 5 6 7 8 9 A B C D E F G H J
```

The other ~6% of the time the digest happens to be smaller than `58^43` and
encodes shorter — only then can the address begin with `K`–`Z` or `a`–`z`.
So a prefix like `octu...` is still possible, just ~50× rarer per key than a
prefix like `octa...`. The miner doesn't reject these — it just takes longer.

---

## Correctness fix

The original CUDA implementation has a subtle bug in `gf_par`: it carry-
reduces the `x` coordinate but does not fully reduce it modulo `p =
2^255 - 19` before taking the parity bit. Because `p` is odd, when `x` is in
`[p, 2p)` the parity bit is flipped relative to the canonical
representation. That means roughly half a percent of the wallets the CUDA
miner produces have a **wrong sign bit in the last byte**, so the address
printed by the miner doesn't actually correspond to the saved private key
when re-derived through standard ed25519.

This port follows tweetnacl's `par25519`: full `pack25519` reduction first,
then read the low bit of byte 0. Wallets produced here always re-derive to
the address that was printed — you can sanity-check with libsodium / PyNaCl /
the Octra webcli.

---

## Output

When found:

```
  ***********************************************
  *                  FOUND!                         *
  ***********************************************

  Address     octHELLO... (your pattern)
  Private key <base64>
  Tried       N keys in T s  (R MK/s)

  Saved       wallet_vanity_XXXX.json
```

The JSON file is the same format the Octra webcli imports.

**Keep this file private.** Anyone with the priv can move funds.

---

## Layout

```
shaders/octra_vanity.metal   # MSL kernels (SHA-2, GF math, ed25519, base58, matchers)
src/main.swift               # host: CLI, Metal setup, dispatch, auto-tune, JSON
src/test_derive.swift        # tiny harness to dump pk for one seed (debug)
Makefile                     # `make`, `make run`, `make clean`
reference/                   # original CUDA source for diff-friendliness
```

---

## Credits

Algorithm and original implementation by
[@0x02937](https://github.com/0x02937).
Metal port by [@utxo-detective](https://github.com/utxo-detective).

---

## Donate

If this saves you a CUDA-rig rental, tips welcome:

```
Octra:  octUtxoYZt1UhJPUUSv3SDSqo1YtMfjVk7cgDFrrrjPgoQ
```

(Mined with this miner, naturally.)
