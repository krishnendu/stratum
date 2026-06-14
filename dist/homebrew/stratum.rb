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
  version "0.2.1"

  conflicts_with "stratum-llama-cpp",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.1/stratum-v0.2.1-aarch64-apple-darwin.tar.gz"
      sha256 "42de59e279f7231c45139ff4949e20fba58950abb82f85eed638ccfb21f8946f"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.1/stratum-v0.2.1-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "496696bcbef6671bb9ccae9ef44363467b8cb517ecf0cb12d3140f58d6094ce2"
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
