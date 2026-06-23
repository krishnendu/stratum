# typed: false
# frozen_string_literal: true

# Homebrew formula for Stratum.
#
# Workflow:
#   1. Cut v0.1.0 via `git tag v0.1.0 && git push --tags`.
#   2. Release workflow uploads `stratum-v0.1.0-aarch64-apple-darwin.tar.gz` (+ sha256).
#   3. Maintainer updates this formula's `version`, `url`, `sha256` to match.
#   4. Maintainer pushes the formula to `krishnendu/homebrew-stratum`.
#   5. Users install via `brew tap krishnendu/stratum && brew install stratum`.
#
# This file lives in-repo as a scaffold. The published tap is at
# https://github.com/krishnendu/homebrew-stratum.

class Stratum < Formula
  desc "Stratum — your local model crew. Local-first chat with an LLM agent"
  homepage "https://github.com/krishnendu/stratum"
  license any_of: ["Apache-2.0", "MIT"]
  version "1.0.0"

  conflicts_with "stratum-llama-cpp",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.0/stratum-v1.0.0-aarch64-apple-darwin.tar.gz"
      sha256 "a9125e82a30ebac8b19a4512e70ecce016c2728168a05def8486a7a1ccca9db5"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.0/stratum-v1.0.0-x86_64-apple-darwin.tar.gz"
      sha256 "926cd617f636b9d267d0edf950e0d3ac2bae00eec8664e3cf27f195918bdade0"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.0/stratum-v1.0.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "7d72e0fd4065372ca22795587da79c3f176e38afe9e59b311786b146ee191e80"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.0/stratum-v1.0.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "6a4976887a27082e8892a22323dee3b96a597d950e5dd4a06ff259e9becb9263"
    end
  end

  def install
    bin.install "stratum"
    # Tab completions
    generate_completions_from_executable(bin/"stratum", "completions")
  end

  test do
    # `doctor --json` is the most stable smoke surface: a single
    # subcommand, no provider or model state, prints a JSON document
    # with documented top-level keys from `DoctorReport`.
    output = shell_output("#{bin}/stratum --json doctor")
    assert_match "schema_version", output
    assert_match "stratum_version", output
    # Non-zero return on missing subcommand should be exit 64.
    output = shell_output("#{bin}/stratum bogus 2>&1", 64)
    assert_match "Usage", output
  end
end
