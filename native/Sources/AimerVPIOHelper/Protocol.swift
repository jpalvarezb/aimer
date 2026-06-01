import Foundation

/// Shared wire-protocol constants. Mirror of the Python side
/// (`duplex_bridge/audio_backends/native_vpio_backend.py`).
///
/// Length-prefixed binary PCM over stdio; stderr is logs only.
/// - bridge → helper stdin (model audio): [4-byte LE uint32 N][N bytes 24 kHz mono int16]
///   N = 0 is the barge-in flush sentinel: drop all audio scheduled on the player node.
/// - helper → bridge stdout (mic):        [4-byte LE uint32 N][N bytes 16 kHz mono int16], N = 3200
enum Wire {
    static let lenPrefix = 4
    static let captureRate = 16_000.0
    static let frameSamples = 1_600  // 100 ms at 16 kHz; matches the bridge's VAD cadence
    static let frameBytes = 1_600 * 2  // 3200 bytes int16 mono
    static let modelRate = 24_000.0  // rate of model audio arriving on stdin
}

/// Write a line to stderr (the helper's log channel).
func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}
