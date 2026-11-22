# XCFramework

The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```

The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

**Note for this fork:** the URL above points at an upstream `ggml-org/llama.cpp` prebuilt release, which
does not include this fork's changes (MoE expert-cache, Brain/Atlas view, expert-prefetch, etc). This fork
does not currently publish its own prebuilt XCFrameworks - to get an XCFramework with this fork's changes,
build it from source instead (see the CMake build docs and target the relevant Apple platform).
