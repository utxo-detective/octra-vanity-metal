# octra-vanity-metal

Mac-native Octra vanity address miner. Brute-forces ed25519 keypairs on the
Apple Silicon GPU via Metal until your address looks the way you want.

```
oct 1Utxo YZt1UhJPUUSv3SDSqo1YtMfjVk7cgDFrrrjPgoQ
    ^^^^^
    you pick this part
```

No CUDA, no NVIDIA card, no nightmares.

## Quick start

```
xcode-select --install                          # one-time, if you don't have Xcode CLI tools
git clone https://github.com/utxo-detective/octra-vanity-metal
cd octra-vanity-metal
make
./build/octra_vanity_metal --prefix HELLO -i    # mines oct...Hello... (case-insensitive)
```

The resulting `build/wallet_vanity_*.json` imports straight into the
[Octra webcli](https://github.com/octra-labs/webcli).

## How fast

| Pattern (case-sensitive) | Expected keys | Time on M3 Max |
|---|---|---|
| 4-char prefix    | ~3 M          | < 1 s |
| 5-char prefix    | ~200 M        | < 1 s |
| 6-char prefix    | ~11 B         | ~5 s |
| 7-char prefix    | ~660 B        | ~5 min |
| 8-char prefix    | ~38 T         | ~4 hr |
| 9-char prefix    | ~2 P          | ~10 days |
| 5 repeating-start | < 1 M        | < 1 s |
| 6 repeating-start | ~30 M        | < 1 s |
| 7 repeating-start | ~2 B          | ~1 s |

Case-insensitive roughly halves the difficulty per letter character. Use
`--estimate` to get an exact prediction for your specific pattern at your
current GPU rate.

| Kernel       | Throughput on M3 Max |
|---|---|
| prefix / anywhere / rep-start / rep-any | **~2.4 GK/s** |
| suffix       | ~50–80 MK/s |
| rep-end      | ~50–80 MK/s |

Suffix and rep-end have to fully base58-encode each candidate before they can
check the tail, so they're an order of magnitude slower than the prefix-style
kernels which can short-circuit after the first few base58 digits.

## Usage

### Interactive

Just run it with no arguments and answer the prompts:

```
./build/octra_vanity_metal
```

### Command-line

```
# 5-char prefix, octHELLO...
./build/octra_vanity_metal --prefix HELLO

# Match anywhere in the address, case-insensitive
./build/octra_vanity_metal --anywhere DEGEN -i

# Suffix
./build/octra_vanity_metal --suffix END

# Repeating chars
./build/octra_vanity_metal --rep-start 5     # octXXXXX...
./build/octra_vanity_metal --rep-end   5     # ...XXXXX
./build/octra_vanity_metal --rep-any   6     # 6 identical chars anywhere

# Throttle so the rest of your Mac stays responsive (25% GPU duty cycle)
./build/octra_vanity_metal --prefix ABC --gpu-budget 25 --show-gpu

# Predict difficulty + time without actually mining
./build/octra_vanity_metal --estimate --prefix 5HELL -i

# Pin a non-default RPC into the saved wallet JSON
./build/octra_vanity_metal --prefix XYZ --rpc https://my.octra.rpc/
```

### All flags

```
Pattern:
  --prefix <pat>       --suffix <pat>     --anywhere <pat>
  --rep-start <N>      --rep-end <N>      --rep-any <N>
  -i, --case-insensitive
  --rpc <url>
  --bonus <N>:<mode>   also record a separate "bonus" wallet matching
                       N repeating chars; mode = 1/2/3 (start/end/anywhere)

Performance:
  --threadgroups <N>   --threads <N>   --iters <N>   --no-auto-tune
  --gpu-budget <pct>   keep GPU at roughly pct% duty cycle (1..100)
  --show-gpu           print live GPU utilisation in the progress line
  --estimate           benchmark + predict difficulty, then exit (no mining)

Misc:
  -h, --help
```

## What patterns work

The miner matches against the **canonical** address format that the Octra
webcli displays — `oct` + exactly 44 base58 chars, where the base58 portion is
left-padded with `1`s to keep length at 44.

Practical consequences:

- **Base58 alphabet only.** No `0`, `O`, `I`, `l`. The miner rejects bad chars
  before launching the GPU.
- **Leading-character constraint.** The first base58 char (i.e. the char right
  after `oct`) is **always** one of: `1 2 3 4 5 6 7 8 9 A B C D E F G H J` —
  18 options out of 58. The miner fails fast on unreachable prefixes:

  ```
  $ ./octra_vanity_metal --prefix utxo
  Pattern 'utxo' starts with 'u', but canonical Octra addresses
  always begin with one of:  1 2 3 4 5 6 7 8 9 A B C D E F G H J
  ```

  (The math: SHA-256 of the public key viewed as a 256-bit integer needs 44
  base58 digits because `2^256 < 58^44`, and the leading digit is bounded by
  `floor(2^256 / 58^43) = 17`. The other ~5.8% of the time the digest fits in
  fewer digits and webcli left-pads with `1`s, so leading `1` is possible.)

- **`--suffix`, `--anywhere`, `--rep-*` modes accept any base58 chars.** Only
  prefix matching cares about the leading-char set.

## Importing the wallet

The output `wallet_*.json` is in webcli's legacy format and works two ways:

**Paste-priv flow (recommended)** — in the webcli UI, paste just the base64
priv:

```
jq -r .priv build/wallet_vanity_*.json
```

The webcli will rederive the canonical address from your private key.

**File-import flow** — drop the JSON next to the webcli binary as
`wallet.json` and start; webcli will pick it up and offer to migrate to its
encrypted format.

Either way, the priv is a 32-byte ed25519 seed, base64-encoded.

## What's running

```
shaders/octra_vanity.metal   # MSL kernels: SHA-256/512, GF(2^255-19) field math,
                             # Ed25519 scalarbase with precomputed 8-bit window
                             # table, base58, pattern matchers
src/main.swift               # Swift host: CLI, Metal setup, auto-tune,
                             # GPU duty-cycle throttle, single-instance lock,
                             # JSON wallet writer
Makefile                     # `make`, `make run`, `make clean`
reference/                   # original CUDA source kept for diff-friendliness
```

The five algorithmic optimisations from the CUDA implementation are all
preserved: precomputed-window scalar mult, truncated SHA-512 (first 32 bytes
only), 32-bit base58 routines, dedicated `ed_double`, and 32-bit-limb GF math
to keep register pressure low. Wallets re-derive correctly through standard
ed25519 — verified against libsodium / PyNaCl / the Octra webcli.

## Safety

- **Single instance only.** Trying to start a second miner while one is
  running will print which PID holds the lock and exit. Two miners on the
  same GPU just thrash each other and race on `wallet_*.json` files.
- **Throttle by default in development.** `--gpu-budget 25` keeps the rest of
  the system responsive at the cost of ~4× wall-clock time.
- **Wallet files contain private keys.** Anyone with a `wallet_*.json` can
  move funds from that address. `build/wallet_*.json` is in `.gitignore` so
  you can't accidentally commit them, but copy them somewhere safe.

## Troubleshooting

- **"Permission denied" running the binary** — macOS sometimes drops the
  executable bit on the built file (especially when a sandbox is involved).
  Run `chmod +x build/octra_vanity_metal` or just `make clean && make`.
- **"Another octra_vanity_metal is already running"** — exactly what it
  says. The lockfile lives at `/tmp/octra_vanity_metal.lock`. If you're sure
  the holder is dead, remove the lock by hand.
- **macOS gatekeeper warning** — the binary isn't signed. If you'd rather
  not click through, build from source (the recommended path anyway).

## Credits

Algorithm and original CUDA implementation by
[@0x02937](https://github.com/0x02937).
Metal port by [@utxo-detective](https://github.com/utxo-detective).

## Donate

If this saved you a CUDA-rig rental, tips welcome:

```
Octra:  oct1UtxoYZt1UhJPUUSv3SDSqo1YtMfjVk7cgDFrrrjPgoQ
```

(Mined with this miner, naturally — case-insensitive `Utxo`, displayed by
webcli with the canonical leading `1` pad.)
