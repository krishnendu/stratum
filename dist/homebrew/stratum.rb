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
  version "0.2.10"

  conflicts_with "stratum-llama-cpp",
    because: "both install a `stratum` binary; pick the prebuilt or the LLM-enabled build"

  on_macos do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.10/stratum-v0.2.10-aarch64-apple-darwin.tar.gz"
      sha256 "c2b9859eb9692c0aa68c87e92f3e3f89e75db5c39cf2939cc3569b1f50c6c151"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.10/stratum-v0.2.10-x86_64-apple-darwin.tar.gz"
      sha256 "8431389e7150fdb7609ebff3fd217b8864aeaf1fcd81f36c76d7028d6864e016"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.10/stratum-v0.2.10-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "5641ad304373fde0ae0368624bc3f7d133983dc030f5a3443f950af3c5351185"
    end
    on_intel do
      url "https://github.com/krishnendu/stratum/releases/download/v0.2.10/stratum-v0.2.10-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "fa00eb0f424d2a3b344786fdd20dd8e771ddaa7a464d338b2cb44b88d376f340"
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
