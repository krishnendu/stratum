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
  desc "Stratum — your local model crew (with real local LLM inference via llama.cpp)"
  homepage "https://github.com/krishnendu/stratum"
  license any_of: ["Apache-2.0", "MIT"]
  version "0.2.9"  # MAINTAINER: bump on each release; sourced from the same tag as `stratum`

  url "https://github.com/krishnendu/stratum/archive/refs/tags/v0.2.9.tar.gz"
  sha256 "0fdeee794f9bcfb9997843f7a4aff5a5634aa0dbd5bdb2a4ff1c8a0aba01f8ef"

  conflicts_with "stratum",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  depends_on "rust" => :build
  depends_on "cmake" => :build
  depends_on "pkg-config" => :build

  def install
    # `--locked` pins to the committed Cargo.lock so the formula
    # builds the exact dependency graph CI tested.
    system "cargo", "build", "--release", "--locked",
                   "--features", "provider-llama-cpp", "--bin", "stratum"
    bin.install "target/release/stratum"
    generate_completions_from_executable(bin/"stratum", "completions")
  end

  test do
    # Mirror the prebuilt formula's stable smoke surface.
    output = shell_output("#{bin}/stratum --json doctor")
    assert_match "schema_version", output
    assert_match "stratum_version", output
    assert_match(/stratum \d+\.\d+\.\d+/, shell_output("#{bin}/stratum --version"))
  end
end
