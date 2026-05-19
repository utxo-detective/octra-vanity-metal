// Debug entry: takes one seed (hex on argv[1]) and prints derived pk hex.
import Foundation
import Metal

let arg = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "ea8411766126434963ddc7bab116b000196adb4778ca37d98d7ad7d99c6a3446"

func hexToBytes(_ s: String) -> [UInt8] {
    var out = [UInt8]()
    var idx = s.startIndex
    while idx < s.endIndex {
        let nxt = s.index(idx, offsetBy: 2)
        out.append(UInt8(s[idx..<nxt], radix: 16)!)
        idx = nxt
    }
    return out
}

let device = MTLCreateSystemDefaultDevice()!
let libURL = URL(fileURLWithPath: "build/default.metallib")
let lib = try device.makeLibrary(URL: libURL)
let initFn = lib.makeFunction(name: "init_table_kernel")!
let debugFn = lib.makeFunction(name: "debug_derive")!
let initPSO = try device.makeComputePipelineState(function: initFn)
let dbgPSO  = try device.makeComputePipelineState(function: debugFn)
let queue = device.makeCommandQueue()!

let table = device.makeBuffer(length: 32*256*96, options: .storageModePrivate)!
do {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(initPSO)
    enc.setBuffer(table, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}

let seedBytes = hexToBytes(arg)
let seedBuf = device.makeBuffer(length: 32, options: .storageModeShared)!
memcpy(seedBuf.contents(), seedBytes, 32)
let pkBuf = device.makeBuffer(length: 32, options: .storageModeShared)!
memset(pkBuf.contents(), 0, 32)

let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(dbgPSO)
enc.setBuffer(seedBuf, offset: 0, index: 0)
enc.setBuffer(pkBuf,   offset: 0, index: 1)
enc.setBuffer(table,   offset: 0, index: 2)
enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
enc.endEncoding()
cb.commit(); cb.waitUntilCompleted()

let raw = pkBuf.contents().assumingMemoryBound(to: UInt8.self)
var hex = ""
for i in 0..<32 { hex += String(format: "%02x", raw[i]) }
print("seed: \(arg)")
print("pk:   \(hex)")
