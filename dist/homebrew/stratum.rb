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
  version "0.2.9"

  conflicts_with "stratum-llama-cpp",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.9/stratum-v0.2.9-aarch64-apple-darwin.tar.gz"
      sha256 "4e1d685f4378c0c862395c7ae8393ec7e56846d5d7767adf913d25142949fa60"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.9/stratum-v0.2.9-x86_64-apple-darwin.tar.gz"
      sha256 "50c669b67a5fe6ad842375d31642e2f91b4dc57edc1bb7737cb52ef33249641d"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.9/stratum-v0.2.9-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "b73eb17122c8a8b0215bb6c519e2d4da8d8c1557998dfe0c573a89c2c78365f8"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.9/stratum-v0.2.9-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "c0b6d14102fc50e278e27f2b2f60947429f0fef9ff17eb8a0c5fcaf5a16c9f44"
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
