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
  desc "Local-LLM agentic TUI agent for laptop-class hardware"
  homepage "https://github.com/krishnendu/stratum"
  license any_of: ["Apache-2.0", "MIT"]
  version "0.1.1"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.1.1/stratum-v0.1.1-aarch64-apple-darwin.tar.gz"
      sha256 "1739c0a8f415205648a097fdb63d0f33ae642518361ff55ce9aa059c5a3c4bbb"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.1.1/stratum-v0.1.1-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "38309a71c4264a69d5a1c38c19b65d8a15421c1860341b3ad64d079df5cf5a85"
    end
  end

  def install
    bin.install "stratum"
    # Tab completions
    generate_completions_from_executable(bin/"stratum", "completions")
  end

  test do
    # Sanity: echo provider runs without a real LLM dep.
    assert_match "echo", shell_output("#{bin}/stratum --json doctor")
    # Non-zero return on missing subcommand should be exit 64.
    output = shell_output("#{bin}/stratum bogus 2>&1", 64)
    assert_match "Usage", output
  end
end
