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
  version "1.0.1"

  conflicts_with "stratum-llama-cpp",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.1/stratum-v1.0.1-aarch64-apple-darwin.tar.gz"
      sha256 "2591227aea3972786523994c8eca53577de124d3abcaf98d34ee730387884988"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.1/stratum-v1.0.1-x86_64-apple-darwin.tar.gz"
      sha256 "f81b2538560220b809455db2f159c9a6e5daea32034af4645113abc0e3341a15"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.1/stratum-v1.0.1-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "10eb2953c92efdbea582facbc235568eeb62b2740f21628d38163475993ead04"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v1.0.1/stratum-v1.0.1-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "19d011d66d5d043a3826251ec591462ccb581967f79849c050524f728225e694"
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
