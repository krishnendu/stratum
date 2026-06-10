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
  version "0.1.2"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.1.2/stratum-v0.1.2-aarch64-apple-darwin.tar.gz"
      sha256 "6bea085105f72b2d9fa1e97549890bdb672c4edd6ff2e727a7ccc1a9520ab9bd"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.1.2/stratum-v0.1.2-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "cb36636783987d48fc1460d2aaf1e6c7f1202c8d6bb963af8a6a649e33d160a7"
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
