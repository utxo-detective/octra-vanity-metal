// octra_vanity_metal — Octra vanity address miner, Metal version.
//
// CLI flags or interactive mode. Reads Apple GPU utilisation via IOKit and
// optionally throttles to a duty-cycle budget so the rest of the system stays
// responsive.

import Foundation
import Metal
import QuartzCore
import IOKit

// MARK: - Shader-matching constants

let MODE_PREFIX     : Int32 = 0
let MODE_SUFFIX     : Int32 = 1
let MODE_ANYWHERE   : Int32 = 2
let MODE_REP_START  : Int32 = 3
let MODE_REP_END    : Int32 = 4
let MODE_REP_ANY    : Int32 = 5

let BASE58_ALPHABET = Array("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

struct VanityCfg {
    var mode: Int32 = 0
    var pat: (CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar,
              CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar) =
              (0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0)
    var plen: Int32 = 0
    var rep: Int32 = 0
    var caseInsensitive: Int32 = 0
    var repBonus: Int32 = 0
    var repBonusMode: Int32 = 0

    mutating func setPattern(_ s: String) {
        let bytes = Array(s.utf8)
        let n = min(63, bytes.count)
        withUnsafeMutableBytes(of: &pat) { buf in
            for i in 0..<n { buf[i] = bytes[i] }
            for i in n..<64 { buf[i] = 0 }
        }
        plen = Int32(n)
    }
}

let PRECOMP_TABLE_ENTRIES = 32 * 256
let PRECOMP_POINT_SIZE = 96

// MARK: - Helpers

func sep() { print("  ***********************************************") }

func readLineTrimmed() -> String {
    return (readLine() ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
}

func readIntDefault(_ defaultVal: Int) -> Int {
    let s = readLineTrimmed()
    if s.isEmpty { return defaultVal }
    if let v = Int(s), v > 0 { return v }
    return defaultVal
}

func validB58(_ s: String) -> Bool {
    for c in s { if BASE58_ALPHABET.firstIndex(of: c) == nil { return false } }
    return true
}

func b64encode32(_ bytes: [UInt8]) -> String {
    return Data(bytes).base64EncodedString()
}

func writeWallet(filename: String, priv: String, addr: String, rpc: String) {
    let s = """
    {
      "priv": "\(priv)",
      "addr": "oct\(addr)",
      "rpc": "\(rpc)"
    }

    """
    do { try s.write(toFile: filename, atomically: true, encoding: .utf8) }
    catch { FileHandle.standardError.write("  Warning: could not save wallet file: \(error)\n".data(using: .utf8)!) }
}

// MARK: - GPU utilisation (IOKit)

func readGpuUtilizationPercent() -> Int? {
    let matching = IOServiceMatching("IOAccelerator")
    var iter: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) == KERN_SUCCESS else {
        return nil
    }
    defer { IOObjectRelease(iter) }
    var best: Int? = nil
    while case let service = IOIteratorNext(iter), service != 0 {
        defer { IOObjectRelease(service) }
        var unmanaged: Unmanaged<CFMutableDictionary>?
        if IORegistryEntryCreateCFProperties(service, &unmanaged, kCFAllocatorDefault, 0) == KERN_SUCCESS,
           let dict = unmanaged?.takeRetainedValue() as? [String: Any],
           let perf = dict["PerformanceStatistics"] as? [String: Any],
           let util = perf["Device Utilization %"] as? Int {
            // If multiple accelerators (rare), prefer max.
            if best == nil || util > best! { best = util }
        }
    }
    return best
}

// MARK: - CLI flag parsing

func usage() {
    print("""

      octra_vanity_metal — Octra vanity address miner (Metal)

      Interactive (no flags):
        octra_vanity_metal

      Non-interactive:
        --prefix <pat>        match  oct<pat>...
        --suffix <pat>        match  ...<pat>
        --anywhere <pat>      match  ...<pat>...
        --rep-start <N>       N identical chars right after "oct"
        --rep-end <N>         N identical chars at the end
        --rep-any <N>         N identical chars anywhere

        -i, --case-insensitive   match prefix/suffix/anywhere ignoring case
        --rpc <url>              RPC URL written into the wallet json (default: https://devnet.octra.com)
        --bonus <N>:<mode>       also record bonus pattern of N repeating chars; mode = 1/2/3 (start/end/anywhere)

      Performance:
        --threadgroups <N>       override threadgroup count (default: auto-tune)
        --threads <N>            override threads per threadgroup
        --iters <N>              keys per thread per launch
        --no-auto-tune           skip auto-tune even if other knobs unset
        --gpu-budget <pct>       throttle to roughly pct% GPU duty cycle (1..100, default 100 = unthrottled)
        --show-gpu               print live GPU utilisation alongside progress

      Misc:
        -h, --help               this text
    """)
}

struct CLI {
    var mode: Int32? = nil
    var pattern: String? = nil
    var rep: Int = 0
    var caseInsensitive: Bool = false
    var rpc: String = "https://devnet.octra.com"
    var bonusN: Int = 0
    var bonusMode: Int = 0

    var threadgroups: Int? = nil
    var threads: Int? = nil
    var iters: Int? = nil
    var autoTune: Bool = true
    var gpuBudget: Int = 100   // 1..100
    var showGpu: Bool = false

    var nonInteractive: Bool = false
}

func parseArgs() -> CLI {
    var c = CLI()
    var args = Array(CommandLine.arguments.dropFirst())
    func need(_ flag: String) -> String {
        guard !args.isEmpty else { print("Missing value for \(flag)"); exit(2) }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "-h", "--help": usage(); exit(0)
        case "--prefix":     c.mode = MODE_PREFIX;    c.pattern = need(a); c.nonInteractive = true
        case "--suffix":     c.mode = MODE_SUFFIX;    c.pattern = need(a); c.nonInteractive = true
        case "--anywhere":   c.mode = MODE_ANYWHERE;  c.pattern = need(a); c.nonInteractive = true
        case "--rep-start":  c.mode = MODE_REP_START; c.rep = Int(need(a)) ?? 0; c.nonInteractive = true
        case "--rep-end":    c.mode = MODE_REP_END;   c.rep = Int(need(a)) ?? 0; c.nonInteractive = true
        case "--rep-any":    c.mode = MODE_REP_ANY;   c.rep = Int(need(a)) ?? 0; c.nonInteractive = true
        case "-i", "--case-insensitive": c.caseInsensitive = true
        case "--rpc":        c.rpc = need(a)
        case "--bonus":
            let v = need(a).split(separator: ":")
            if v.count == 2, let n = Int(v[0]), let m = Int(v[1]) { c.bonusN = n; c.bonusMode = m }
        case "--threadgroups": c.threadgroups = Int(need(a)); c.autoTune = false
        case "--threads":      c.threads      = Int(need(a)); c.autoTune = false
        case "--iters":        c.iters        = Int(need(a)); c.autoTune = false
        case "--no-auto-tune": c.autoTune = false
        case "--gpu-budget":
            if let v = Int(need(a)) { c.gpuBudget = max(1, min(100, v)) }
        case "--show-gpu":     c.showGpu = true
        default:
            print("Unknown argument: \(a)"); usage(); exit(2)
        }
    }
    return c
}

let cli = parseArgs()

// MARK: - Metal setup

guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal device available."); exit(1)
}

func locateMetallib() -> URL {
    let fm = FileManager.default
    let exeURL = URL(fileURLWithPath: CommandLine.arguments[0])
    let exeDir = exeURL.deletingLastPathComponent()
    let beside = exeDir.appendingPathComponent("default.metallib")
    if fm.fileExists(atPath: beside.path) { return beside }
    let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
    let underBuild = cwd.appendingPathComponent("build/default.metallib")
    if fm.fileExists(atPath: underBuild.path) { return underBuild }
    let plain = cwd.appendingPathComponent("default.metallib")
    if fm.fileExists(atPath: plain.path) { return plain }
    return beside
}

let metallibURL = locateMetallib()
let library: MTLLibrary
do { library = try device.makeLibrary(URL: metallibURL) }
catch {
    print("Could not load Metal library at \(metallibURL.path): \(error)")
    exit(1)
}

guard
    let initFn   = library.makeFunction(name: "init_table_kernel"),
    let kFn      = library.makeFunction(name: "vanity_kernel"),
    let kSuffix  = library.makeFunction(name: "vanity_kernel_suffix"),
    let kRepEnd  = library.makeFunction(name: "vanity_kernel_rep_end")
else { print("Missing kernel function in metallib."); exit(1) }

let initPSO    = try! device.makeComputePipelineState(function: initFn)
let psoGeneric = try! device.makeComputePipelineState(function: kFn)
let psoSuffix  = try! device.makeComputePipelineState(function: kSuffix)
let psoRepEnd  = try! device.makeComputePipelineState(function: kRepEnd)

guard let queue = device.makeCommandQueue() else {
    print("Cannot create command queue."); exit(1)
}

print()
sep()
print("  *  Octra Vanity Address Miner — Metal             *")
sep()
print("  Device : \(device.name)")
let family: String =
    device.supportsFamily(.apple9) ? "Apple9" :
    device.supportsFamily(.apple8) ? "Apple8" :
    device.supportsFamily(.apple7) ? "Apple7" : "?"
print("  Family : \(family)")
let maxThreads = psoGeneric.maxTotalThreadsPerThreadgroup
print("  Max threads/tg : \(maxThreads)")
if let u = readGpuUtilizationPercent() { print("  GPU now : \(u)% utilisation") }
print()

// MARK: - Resolve cfg from CLI or interactive prompts

var cfg = VanityCfg()
var rpc = cli.rpc

if cli.nonInteractive {
    cfg.mode = cli.mode!
    switch cfg.mode {
    case MODE_PREFIX, MODE_SUFFIX, MODE_ANYWHERE:
        guard let p = cli.pattern, !p.isEmpty else { print("Missing pattern."); exit(2) }
        if !validB58(p) { print("Pattern '\(p)' contains non-base58 characters."); exit(2) }
        cfg.setPattern(p)
    case MODE_REP_START, MODE_REP_END, MODE_REP_ANY:
        if cli.rep < 2 { print("--rep-* requires N >= 2"); exit(2) }
        cfg.rep = Int32(cli.rep)
    default: print("Pick a mode."); usage(); exit(2)
    }
    cfg.caseInsensitive = cli.caseInsensitive ? 1 : 0
    if cli.bonusN > 0 && (1...3).contains(cli.bonusMode) {
        cfg.repBonus = Int32(cli.bonusN); cfg.repBonusMode = Int32(cli.bonusMode)
    }
} else {
    // Interactive prompts (kept identical to the CUDA reference UI).
    sep()
    print("  Where should the pattern appear?\n")
    print("    1.  Prefix      oct[PATTERN]...")
    print("    2.  Suffix      ...[PATTERN]")
    print("    3.  Anywhere    ...[PATTERN]...")
    print("    4.  Repeating   consecutive identical characters\n")
    var type = 0
    while type < 1 || type > 4 {
        print("  > ", terminator: ""); fflush(stdout)
        type = Int(readLineTrimmed()) ?? 0
        if type < 1 || type > 4 { print("  Please enter 1, 2, 3 or 4.") }
    }
    print()
    sep()
    if type <= 3 {
        let label = ["", "prefix", "suffix", "anywhere"][type]
        print("  Enter the pattern (\(label)).")
        print("  Valid chars: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        print("  Note: 0 O I l are NOT in base58.\n")
        var pat = ""
        while pat.isEmpty {
            print("  Pattern > ", terminator: ""); fflush(stdout)
            let s = readLineTrimmed()
            if s.isEmpty { print("  Pattern cannot be empty."); continue }
            if !validB58(s) { print("  Contains invalid base58 characters. Try again."); continue }
            pat = s
        }
        cfg.mode = (type == 1) ? MODE_PREFIX : (type == 2) ? MODE_SUFFIX : MODE_ANYWHERE
        cfg.setPattern(pat)
        print()
        print("  Case sensitive? (Y/n) > ", terminator: ""); fflush(stdout)
        let ans = readLineTrimmed()
        cfg.caseInsensitive = (ans.first.map { $0 == "n" || $0 == "N" } ?? false) ? 1 : 0
        print()
        print("  Also record long repeating patterns? (y/N) > ", terminator: ""); fflush(stdout)
        let bonus = readLineTrimmed()
        if bonus.first.map({ $0 == "y" || $0 == "Y" }) ?? false {
            print()
            var repB = 0
            while repB < 2 {
                print("  What's the length of these patterns? (min 2) > ", terminator: ""); fflush(stdout)
                repB = Int(readLineTrimmed()) ?? 0
                if repB < 2 { print("  Must be at least 2.") }
            }
            cfg.repBonus = Int32(repB)
            var rmode = 0
            while rmode < 1 || rmode > 3 {
                print("  Where these patterns must occur? (1 - prefix, 2 - suffix, 3 - anywhere) > ", terminator: ""); fflush(stdout)
                rmode = Int(readLineTrimmed()) ?? 0
                if rmode < 1 || rmode > 3 { print("  Please enter 1, 2, or 3.") }
            }
            cfg.repBonusMode = Int32(rmode)
        }
    } else {
        print("  Where should the repeating characters appear?\n")
        print("    1.  At start   oct[XXXXX]...")
        print("    2.  At end     ...[XXXXX]")
        print("    3.  Anywhere   ...[XXXXX]...\n")
        var sub = 0
        while sub < 1 || sub > 3 {
            print("  > ", terminator: ""); fflush(stdout)
            sub = Int(readLineTrimmed()) ?? 0
            if sub < 1 || sub > 3 { print("  Please enter 1, 2 or 3.") }
        }
        cfg.mode = (sub == 1) ? MODE_REP_START : (sub == 2) ? MODE_REP_END : MODE_REP_ANY
        print()
        var rep = 0
        while rep < 2 {
            print("  How many identical chars? (min 2) > ", terminator: ""); fflush(stdout)
            rep = Int(readLineTrimmed()) ?? 0
            if rep < 2 { print("  Must be at least 2.") }
        }
        cfg.rep = Int32(rep)
    }
    print()
    print("  RPC URL for saved wallet [\(rpc)] > ", terminator: ""); fflush(stdout)
    let rpcBuf = readLineTrimmed()
    if !rpcBuf.isEmpty { rpc = rpcBuf }
}

// MARK: - Print target summary

func patternString() -> String {
    var out = [CChar]()
    withUnsafeBytes(of: cfg.pat) { buf in
        for b in buf {
            let c = CChar(bitPattern: b)
            if c == 0 { break }
            out.append(c)
        }
    }
    out.append(0)
    return out.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
}
let ci = cfg.caseInsensitive != 0 ? "(case-insensitive)" : "(case-sensitive)"
sep()
switch cfg.mode {
case MODE_PREFIX:    print("  Target:  oct\(patternString())... \(ci)")
case MODE_SUFFIX:    print("  Target:  ...\(patternString()) \(ci)")
case MODE_ANYWHERE:  print("  Target:  ...\(patternString())... \(ci)")
case MODE_REP_START: print("  Target:  oct[\(cfg.rep)x same]...")
case MODE_REP_END:   print("  Target:  ...[\(cfg.rep)x same]")
case MODE_REP_ANY:   print("  Target:  ...[\(cfg.rep)x same]...")
default: break
}
let nForEst: Int = (cfg.mode <= MODE_ANYWHERE) ? Int(cfg.plen) : Int(cfg.rep)
var difficulty: Double
switch cfg.mode {
case MODE_PREFIX, MODE_SUFFIX:       difficulty = pow(58.0, Double(nForEst))
case MODE_ANYWHERE:                  difficulty = pow(58.0, Double(nForEst)) / 38.0
case MODE_REP_START, MODE_REP_END:   difficulty = pow(58.0, Double(nForEst - 1))
case MODE_REP_ANY:                   difficulty = pow(58.0, Double(nForEst - 1)) / 38.0
default:                             difficulty = 1.0
}
if cfg.caseInsensitive != 0 { difficulty /= pow(2.0, Double(min(nForEst, 22))) }
let speedEst = 1.5e9
let seconds = difficulty / speedEst
let estStr: String
if seconds < 2 { estStr = "< 1 second" }
else if seconds < 120 { estStr = String(format: "~%.0f seconds", seconds) }
else if seconds < 7200 { estStr = String(format: "~%.0f minutes", seconds/60.0) }
else if seconds < 172800 { estStr = String(format: "~%.1f hours", seconds/3600.0) }
else { estStr = String(format: "~%.1f days", seconds/86400.0) }
print(String(format: "  Difficulty: ~%.3g keys on average", difficulty))
print("  Estimated:  \(estStr)  (assuming \(Int(speedEst/1e6)) MK/s)")
if cli.gpuBudget < 100 {
    print("  Throttle:   GPU duty-cycle budget \(cli.gpuBudget)%")
}
print()

// MARK: - Buffers

let cfgBuf = device.makeBuffer(length: MemoryLayout<VanityCfg>.size, options: .storageModeShared)!
let mainOutBuf  = device.makeBuffer(length: 104, options: .storageModeShared)!
let bonusOutBuf = device.makeBuffer(length: 104, options: .storageModeShared)!
let tableBuf = device.makeBuffer(length: PRECOMP_TABLE_ENTRIES * PRECOMP_POINT_SIZE,
                                 options: .storageModePrivate)!

func resetFound(_ buf: MTLBuffer) { memset(buf.contents(), 0, 104) }

withUnsafePointer(to: &cfg) { _ = memcpy(cfgBuf.contents(), $0, MemoryLayout<VanityCfg>.size) }
resetFound(mainOutBuf); resetFound(bonusOutBuf)

// MARK: - Precompute table

sep()
print("  Precomputing scalar-base table (32 x 256 entries)...")
let initStart = Date()
do {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(initPSO)
    enc.setBuffer(tableBuf, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}
print(String(format: "  Table ready in %.2fs\n", -initStart.timeIntervalSinceNow))

// MARK: - Dispatch helper

@inline(__always)
func pickPSO(_ mode: Int32) -> MTLComputePipelineState {
    switch mode {
    case MODE_SUFFIX:  return psoSuffix
    case MODE_REP_END: return psoRepEnd
    default:           return psoGeneric
    }
}

func launch(baseSeed: UInt64, iters: Int, threadgroups: Int, tgSize: Int, pso: MTLComputePipelineState) -> Double {
    var bs = baseSeed
    var it = Int32(iters)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBytes(&bs, length: MemoryLayout<UInt64>.size, index: 0)
    enc.setBytes(&it, length: MemoryLayout<Int32>.size,  index: 1)
    enc.setBuffer(cfgBuf, offset: 0, index: 2)
    enc.setBuffer(mainOutBuf, offset: 0, index: 3)
    enc.setBuffer(bonusOutBuf, offset: 0, index: 4)
    enc.setBuffer(tableBuf, offset: 0, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    enc.endEncoding()
    let start = CACurrentMediaTime()
    cb.commit(); cb.waitUntilCompleted()
    return CACurrentMediaTime() - start
}

// MARK: - Grid params (defaults / auto-tune)

var threadgroupSize = cli.threads      ?? min(128, maxThreads)
var threadgroups   = cli.threadgroups  ?? 4096
var iters          = cli.iters         ?? 16

if cli.autoTune {
    print("  Auto-tuning performance parameters...")
    let pso = pickPSO(cfg.mode)
    var tgs: [Int] = []
    var v = 32
    while v <= maxThreads { tgs.append(v); v *= 2 }
    let groupsList: [Int] = [256, 512, 1024, 2048, 4096, 8192]
    let iterList: [Int]   = [4, 8, 16, 32, 64]

    var bestKps = 0.0
    var bestTG = threadgroupSize, bestG = threadgroups, bestI = iters

    _ = launch(baseSeed: 0, iters: 4, threadgroups: 1024, tgSize: min(128, maxThreads), pso: pso)
    resetFound(mainOutBuf); resetFound(bonusOutBuf)

    for tg in tgs {
        for g in groupsList {
            for it in iterList {
                resetFound(mainOutBuf); resetFound(bonusOutBuf)
                let ms = launch(baseSeed: 0, iters: it, threadgroups: g, tgSize: tg, pso: pso)
                let kps = Double(UInt64(g) * UInt64(tg) * UInt64(it)) / max(ms, 1e-9)
                if kps > bestKps { bestKps = kps; bestTG = tg; bestG = g; bestI = it }
            }
        }
    }
    threadgroupSize = bestTG; threadgroups = bestG; iters = bestI
    print(String(format: "  Auto-tune best: %d groups x %d threads x %d iters  -->  %.2f MK/s\n",
                 bestG, bestTG, bestI, bestKps / 1e6))
    resetFound(mainOutBuf); resetFound(bonusOutBuf)
}

// MARK: - Search loop

let pso = pickPSO(cfg.mode)
let perLaunch = UInt64(threadgroups) * UInt64(threadgroupSize) * UInt64(iters)
print("  Searching with \(threadgroups) groups x \(threadgroupSize) threads x \(iters) iters = \(perLaunch) keys/launch\n")

func currentBaseSeed() -> UInt64 {
    var r: UInt64 = 0
    let res = SecRandomCopyBytes(kSecRandomDefault, MemoryLayout<UInt64>.size, &r)
    if res == errSecSuccess { return r ^ 0xc0ffeedeadbeef }
    return UInt64(Date().timeIntervalSince1970 * 1e9) ^ 0xc0ffeedeadbeef
}

func extractFound(_ buf: MTLBuffer) -> (UInt32, [UInt8], String) {
    let raw = buf.contents()
    var flag: UInt32 = 0
    memcpy(&flag, raw, MemoryLayout<UInt32>.size)
    let seedOff = 8
    let addrOff = 40
    var seedBytes = [UInt8](repeating: 0, count: 32)
    memcpy(&seedBytes, raw + seedOff, 32)
    let addr: String = {
        let ptr = (raw + addrOff).assumingMemoryBound(to: CChar.self)
        return String(cString: ptr)
    }()
    return (flag, seedBytes, addr)
}

var baseSeed = currentBaseSeed()
let tStart = Date()
var totalKeys: UInt64 = 0
var lastProgress = Date(timeIntervalSince1970: 0)
var done = false

while !done {
    let elapsedSec = launch(baseSeed: baseSeed, iters: iters,
                            threadgroups: threadgroups, tgSize: threadgroupSize, pso: pso)
    let keysThisLaunch = perLaunch
    baseSeed &+= keysThisLaunch
    totalKeys &+= keysThisLaunch

    if cli.gpuBudget < 100 {
        // Duty-cycle throttle: sleep so launch_time / (launch_time + sleep) ≈ budget/100.
        // sleep = launch * (100 - budget) / budget.
        let sleepSec = elapsedSec * Double(100 - cli.gpuBudget) / Double(cli.gpuBudget)
        if sleepSec > 0 {
            // usleep takes useconds_t; clamp to avoid overflow on absurd values.
            let us = min(UInt32(2_000_000), UInt32(sleepSec * 1e6))
            usleep(us)
        }
    }

    let (bonusFlag, bSeed, bAddr) = extractFound(bonusOutBuf)
    if bonusFlag != 0 && !bAddr.isEmpty {
        print("\r\u{001B}[2K  Found bonus pattern: oct\(bAddr)")
        let priv = b64encode32(bSeed)
        let short = String(bAddr.prefix(8))
        writeWallet(filename: "wallet_bonus_\(short).json", priv: priv, addr: bAddr, rpc: rpc)
        resetFound(bonusOutBuf)
        lastProgress = Date(timeIntervalSince1970: 0)
    }

    let (mainFlag, _, _) = extractFound(mainOutBuf)
    if mainFlag != 0 { done = true; break }

    let now = Date()
    if !done && now.timeIntervalSince(lastProgress) >= 2.0 {
        let elapsed = now.timeIntervalSince(tStart)
        let kps = elapsed > 0 ? Double(totalKeys) / elapsed : 0
        let kpsInst = elapsedSec > 0 ? Double(keysThisLaunch) / elapsedSec : 0
        let gpuPart: String = {
            guard cli.showGpu, let u = readGpuUtilizationPercent() else { return "" }
            return String(format: " gpu %2d%%", u)
        }()
        print(String(format: "\r  Tried: %-14llu  %5.0fs  avg %9.0f k/s  inst %6.0f MK/s%@",
                     totalKeys, elapsed, kps, kpsInst / 1e6, gpuPart as CVarArg),
              terminator: "")
        fflush(stdout)
        lastProgress = now
    }
}

let (_, sBytes, sAddr) = extractFound(mainOutBuf)
let elapsed = Date().timeIntervalSince(tStart)
let kpsFinal = elapsed > 0 ? Double(totalKeys) / elapsed : 0
let priv = b64encode32(sBytes)

print("\n")
sep()
print("  *                  FOUND!                         *")
sep()
print()
print("  Address     oct\(sAddr)")
print("  Private key \(priv)")
print(String(format: "  Tried       %llu keys in %.0f s  (%.0f MK/s)\n",
             totalKeys, elapsed, kpsFinal / 1e6))

let short = String(sAddr.prefix(8))
let outName = "wallet_vanity_\(short).json"
writeWallet(filename: outName, priv: priv, addr: sAddr, rpc: rpc)
print("  Saved       \(outName)")
print()
