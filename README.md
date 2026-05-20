# octra-vanity-metal

Mac-native Octra vanity address miner. GPU brute-forces ed25519 keypairs on
Apple Silicon (and Intel Macs with a discrete GPU) via Metal, looking for
addresses that match a pattern you pick.

This is a Metal port of the CUDA implementation at
[0x02937/octra_vanity](https://github.com/0x02937/octra_vanity).
Algorithm is identical (precomputed-window scalar mult, truncated SHA-512,
32-bit base58 matchers, etc.) — just retargeted from `__device__` CUDA C
to MSL kernels and a Swift host. Wallets produced by this miner re-derive
correctly through standard ed25519 (verified against libsodium / PyNaCl).

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

## Forbidden / impossible patterns

Base58 doesn't use `0`, `O`, `I`, `l` — the miner rejects any pattern that
contains those.

**Leading-character constraint.** Canonical Octra addresses (the form the
webcli displays) are always `oct` + exactly 44 base58 chars, with the base58
left-padded with `1`s. The first base58 char of such an address is
mathematically constrained to:

```
1 2 3 4 5 6 7 8 9 A B C D E F G H J
```

— 18 options out of 58. Any prefix like `octa...` or `octu...` is **impossible**
no matter how long you mine. The miner fails fast on unreachable prefixes:

```
$ ./octra_vanity_metal --prefix utxo
Pattern 'utxo' starts with 'u', but canonical Octra addresses
always begin with one of:  1 2 3 4 5 6 7 8 9 A B C D E F G H J
```

(The math: a SHA-256 digest read as a 256-bit big-endian integer fits in 44
base58 digits because `2^256 < 58^44`, and the most-significant digit is
`floor(D / 58^43)`, which is bounded by `floor(2^256 / 58^43) = 17`. When the
digest happens to be smaller than `58^43` — about 5.8% of the time — webcli
left-pads with `1`s to keep length at 44, so the first char then becomes `1`.)

`--suffix`, `--anywhere`, and the `--rep-*` modes have no such constraint and
accept any valid base58 chars.

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
