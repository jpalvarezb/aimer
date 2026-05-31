// swift-tools-version: 6.0
import PackageDescription

// The binary (product) name is `aimer-vpio-helper`, which the Python
// NativeVpioBackend resolves at `.build/release/aimer-vpio-helper`.
//
// Info.plist is embedded into the executable via the linker so the running tool
// carries a CFBundleIdentifier + NSMicrophoneUsageDescription. Without a bundle id
// VPIO (`setVoiceProcessingEnabled`) fails on a bare command-line binary.
//
// Language mode .v5: this target bridges the realtime audio render thread, an
// AVAudioEngine tap, a stdin reader thread, and a serial I/O queue around shared
// mutable state. Swift 6's strict concurrency would demand actor isolation /
// Sendable annotations throughout; .v5 keeps the audio-callback idioms that
// AVFoundation itself is written against. Revisit under full Swift 6 if desired.
let package = Package(
    name: "aimer-vpio-helper",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "aimer-vpio-helper", targets: ["AimerVPIOHelper"])
    ],
    targets: [
        .executableTarget(
            name: "AimerVPIOHelper",
            path: "Sources/AimerVPIOHelper",
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Info.plist",
                ])
            ]
        )
    ]
)
