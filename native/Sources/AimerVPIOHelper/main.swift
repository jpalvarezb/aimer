import AVFoundation
import Foundation

// Unbuffered stdout so capture frames reach the bridge immediately (FileHandle
// writes bypass C stdio buffering, but set this too for any stray stdio use).
setvbuf(stdout, nil, _IONBF, 0)

let selftest = CommandLine.arguments.contains("--selftest")
let helper = VPIOHelper()

helper.ensureMicAccess()

do {
    try helper.start()
} catch {
    log("[helper] failed to start VPIO engine: \(error)")
    exit(1)
}

if selftest {
    // Run the engine briefly and report whether capture produced frames. Used as an
    // on-device sanity check (CI cannot validate VPIO efficacy).
    Thread.sleep(forTimeInterval: 2.0)
    log(
        "[helper] selftest: frames=\(helper.framesEmitted) peak=\(helper.peakAmplitude) "
            + "rawInputPeak=\(helper.rawInputPeak)")
    helper.shutdown()
    exit(helper.framesEmitted > 0 ? 0 : 2)
}

// Normal operation: read model audio from stdin and run until stdin closes.
helper.startStdinReader()
RunLoop.main.run()
