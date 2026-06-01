import AVFoundation
import Foundation

/// Owns the VPIO `AVAudioEngine` (capture **and** playback through an
/// `AVAudioPlayerNode`, which is the echo reference) and bridges raw PCM with the
/// Python side over stdio.
///
/// Realtime safety: the input tap converts to 16 kHz int16 and hands the bytes to a
/// serial `ioQueue`; all stdout writes and frame slicing happen there, never on the
/// audio thread. The stdin reader runs on its own thread and only schedules buffers.
/// Swift ARC retains scheduled buffers until their completion handler fires, so
/// playback does not go silent (the PyObjC failure mode this helper exists to fix).
final class VPIOHelper {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()

    private var captureFormat: AVAudioFormat!  // 16 kHz mono int16 (to the bridge)
    private var modelFormat: AVAudioFormat!  // 24 kHz mono int16 (from the bridge)
    private var renderFormat: AVAudioFormat!  // output node format VPIO forces
    private var captureConverter: AVAudioConverter?  // mono hw input -> captureFormat
    private var captureMonoFormat: AVAudioFormat?  // 1 ch float32 at hw rate (channel 0)
    private var playbackConverter: AVAudioConverter?  // modelFormat -> renderFormat

    private let ioQueue = DispatchQueue(label: "com.aimer.vpio.io")
    private let stdout = FileHandle.standardOutput
    private var captureAccum = Data()

    // Counters for --selftest (read after a fixed run; minor cross-thread race is fine).
    private(set) var framesEmitted = 0
    private(set) var peakAmplitude: Int16 = 0
    private(set) var rawInputPeak: Float = 0  // peak of channel 0 pre-conversion (diagnostic)

    private var stdinThread: Thread?

    // MARK: - Lifecycle

    /// Best-effort microphone authorization. When the parent process (terminal / uv)
    /// already holds TCC mic access the completion fires `granted` immediately.
    func ensureMicAccess() {
        let semaphore = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if !granted { log("[helper] microphone access not granted") }
            semaphore.signal()
        }
        semaphore.wait()
    }

    func start() throws {
        let input = engine.inputNode
        try input.setVoiceProcessingEnabled(true)

        // VPIO input is multichannel float32 at the hardware rate. Read CHANNEL 0
        // (VPIO's processed/echo-cancelled output) directly and resample mono->16k
        // int16. Feeding the layout-less multichannel VPIO format to AVAudioConverter
        // to downmix produces SILENCE; channel-0 read mirrors the working PyObjC path.
        let hwFormat = input.outputFormat(forBus: 0)
        captureFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: Wire.captureRate,
            channels: 1, interleaved: true)
        captureMonoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: hwFormat.sampleRate,
            channels: 1, interleaved: false)
        captureConverter = AVAudioConverter(from: captureMonoFormat!, to: captureFormat)

        // Player is the echo reference. Connect to the OUTPUT node (not mainMixer):
        // VPIO forces the output to the input's rate; the mixer's default rate makes
        // start() fail.
        engine.attach(player)
        renderFormat = engine.outputNode.inputFormat(forBus: 0)
        engine.connect(player, to: engine.outputNode, format: renderFormat)

        modelFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: Wire.modelRate,
            channels: 1, interleaved: true)
        playbackConverter = AVAudioConverter(from: modelFormat, to: renderFormat)

        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            self?.handleCapture(buffer)
        }

        engine.prepare()
        try engine.start()
        player.play()

        // Enabling voice processing fires an AVAudioEngineConfigurationChange that
        // STOPS the engine (the capture-killer diagnosed in the PyObjC backend); device
        // changes do the same. Restart + re-arm the player whenever it fires.
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            self?.restart()
        }

        log("[helper] VPIO engine started: input \(hwFormat), render \(renderFormat!)")
    }

    private func restart() {
        do {
            if !engine.isRunning { try engine.start() }
            player.play()
            log("[helper] engine restarted after configuration change")
        } catch {
            log("[helper] engine restart failed: \(error)")
        }
    }

    func shutdown() {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        // Disable VP so coreaudiod's aggregate device isn't left wedged.
        try? engine.inputNode.setVoiceProcessingEnabled(false)
    }

    // MARK: - Capture (mic -> bridge stdout)

    private func handleCapture(_ buffer: AVAudioPCMBuffer) {
        guard let converter = captureConverter, let monoFormat = captureMonoFormat,
            let src = buffer.floatChannelData
        else { return }
        let n = Int(buffer.frameLength)
        if n == 0 { return }

        // Copy channel 0 into a mono buffer — the multichannel VPIO format has no layout
        // for the converter to downmix; channel 0 is the processed (echo-cancelled) output.
        guard let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: AVAudioFrameCount(n))
        else { return }
        mono.frameLength = AVAudioFrameCount(n)
        memcpy(mono.floatChannelData![0], src[0], n * MemoryLayout<Float>.size)

        // Diagnostic: peak of the raw channel-0 input — distinguishes silent input
        // (permission/device) from a downstream conversion bug.
        var raw: Float = 0
        for i in 0..<n {
            let m = Swift.abs(src[0][i])
            if m > raw { raw = m }
        }
        if raw > rawInputPeak { rawInputPeak = raw }

        let ratio = Wire.captureRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(n) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: captureFormat, frameCapacity: capacity)
        else { return }

        var error: NSError?
        var fed = false
        let status = converter.convert(to: out, error: &error) { _, inStatus in
            if fed {
                inStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inStatus.pointee = .haveData
            return mono
        }
        if status == .error || out.frameLength == 0 { return }
        guard let samples = out.int16ChannelData else { return }

        let count = Int(out.frameLength)
        let data = Data(bytes: samples[0], count: count * 2)
        ioQueue.async { [weak self] in self?.appendCapture(data) }
    }

    /// Serial on `ioQueue`: accumulate int16 bytes and emit fixed 3200-byte frames.
    private func appendCapture(_ data: Data) {
        captureAccum.append(data)
        while captureAccum.count >= Wire.frameBytes {
            let frame = captureAccum.prefix(Wire.frameBytes)
            captureAccum.removeFirst(Wire.frameBytes)
            writeFrame(Data(frame))
            framesEmitted += 1
            updatePeak(Data(frame))
        }
    }

    private func updatePeak(_ frame: Data) {
        frame.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Int16.self)
            for sample in ptr {
                let magnitude = sample == Int16.min ? Int16.max : abs(sample)
                if magnitude > peakAmplitude { peakAmplitude = magnitude }
            }
        }
    }

    private func writeFrame(_ payload: Data) {
        var n = UInt32(payload.count).littleEndian
        var out = Data(bytes: &n, count: Wire.lenPrefix)
        out.append(payload)
        stdout.write(out)
    }

    // MARK: - Playback (bridge stdin -> player)

    func startStdinReader() {
        let thread = Thread { [weak self] in
            guard let self else { return }
            let fd = FileHandle.standardInput.fileDescriptor
            while true {
                guard let header = Self.readExact(fd, Wire.lenPrefix) else { break }
                let raw = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
                let n = Int(UInt32(littleEndian: raw))
                if n == 0 {
                    // Barge-in flush sentinel: drop everything already scheduled.
                    self.flushPlayback()
                    continue
                }
                guard let payload = Self.readExact(fd, n) else { break }
                self.schedulePlayback(payload)
            }
            log("[helper] stdin closed; shutting down")
            self.shutdown()
            exit(0)
        }
        thread.stackSize = 1 << 20
        thread.start()
        stdinThread = thread
    }

    private static func readExact(_ fd: Int32, _ count: Int) -> Data? {
        var buffer = Data(count: count)
        var got = 0
        let ok = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            while got < count {
                let r = read(fd, base.advanced(by: got), count - got)
                if r <= 0 { return false }  // EOF or error
                got += r
            }
            return true
        }
        return ok ? buffer : nil
    }

    /// Drop all audio scheduled on the player node and re-arm it for the next turn.
    /// `stop()` discards buffers the bridge already sent ahead of real time; without
    /// this the assistant keeps talking through that backlog after a barge-in.
    private func flushPlayback() {
        player.stop()
        player.play()
        log("[helper] playback flushed (barge-in)")
    }

    private func schedulePlayback(_ payload: Data) {
        let sampleCount = payload.count / 2
        guard sampleCount > 0,
            let input = AVAudioPCMBuffer(
                pcmFormat: modelFormat, frameCapacity: AVAudioFrameCount(sampleCount))
        else { return }
        input.frameLength = AVAudioFrameCount(sampleCount)
        if let dst = input.int16ChannelData {
            payload.withUnsafeBytes { raw in
                if let base = raw.baseAddress { memcpy(dst[0], base, payload.count) }
            }
        }

        guard let converter = playbackConverter else { return }
        let ratio = renderFormat.sampleRate / modelFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(sampleCount) * ratio) + 16
        guard let output = AVAudioPCMBuffer(pcmFormat: renderFormat, frameCapacity: capacity)
        else { return }

        var error: NSError?
        var fed = false
        let status = converter.convert(to: output, error: &error) { _, inStatus in
            if fed {
                inStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inStatus.pointee = .haveData
            return input
        }
        if status == .error || output.frameLength == 0 { return }

        // ARC keeps `output` alive until the completion handler runs.
        player.scheduleBuffer(output, completionHandler: nil)
        if !player.isPlaying { player.play() }
    }
}
