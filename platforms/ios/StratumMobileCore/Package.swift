// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "StratumMobileCore",
  platforms: [.iOS(.v18)],
  products: [.library(name: "StratumMobileCore", targets: ["StratumMobileCore"])],
  targets: [
    .binaryTarget(name: "stratum_mobile_core",
                  path: "../../../target/aarch64-apple-ios/release/libstratum_mobile_core.dylib"),
    .target(name: "StratumMobileCore",
            dependencies: ["stratum_mobile_core"],
            publicHeadersPath: "include")
  ]
)
