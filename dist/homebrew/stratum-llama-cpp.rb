# typed: false
# frozen_string_literal: true

# Stratum with the `provider-llama-cpp` feature enabled.
#
# Builds from source rather than from the prebuilt release binary, because
# the llama.cpp dependency adds ~350 MB of native C++ build artifacts that
# the default `stratum` formula deliberately omits.
#
# Cold build is ~10-20 minutes on M1 Pro; subsequent reinstalls via the
# Homebrew cache are seconds.
#
# Installation:
#
#   brew tap krishnendu/stratum
#   brew install stratum-llama-cpp
#
# Coexistence with the prebuilt `stratum` formula: pick one or the other.
# Both install a `stratum` binary; the LLM-enabled build replaces the
# prebuilt's `stratum` symlink.

class StratumLlamaCpp < Formula
  desc "Stratum with provider-llama-cpp feature (real local LLM inference)"
  homepage "https://github.com/krishnendu/stratum"
  license any_of: ["Apache-2.0", "MIT"]
  version "0.1.2"  # MAINTAINER: bump on each release; sourced from the same tag as `stratum`

  url "https://github.com/krishnendu/stratum/archive/refs/tags/v0.1.2.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"  # PLACEHOLDER

  depends_on "rust" => :build
  depends_on "cmake" => :build
  depends_on "pkg-config" => :build

  def install
    system "cargo", "build", "--release", "--features", "provider-llama-cpp", "--bin", "stratum"
    bin.install "target/release/stratum"
    generate_completions_from_executable(bin/"stratum", "completions")
  end

  test do
    # Sanity: same surface as the prebuilt formula — the feature
    # only affects the provider path, not the doctor/echo surfaces.
    assert_match "echo", shell_output("#{bin}/stratum --json doctor")
    assert_match(/stratum \d+\.\d+\.\d+/, shell_output("#{bin}/stratum --version"))
    # Smoke: --features should advertise via the binary's --version long form
    # once the build-info wire-up lands. Until then this is a noop check.
    assert_match "stratum", shell_output("#{bin}/stratum --version")
  end
end
