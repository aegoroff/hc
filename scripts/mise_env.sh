# Sourced by the Linux build scripts: installs the tools pinned in the repo's
# mise.toml (zig) and puts them first on PATH. Exporting PATH, rather than
# wrapping calls in `mise exec`, also covers `zig cc` / `zig ar` that OpenSSL's
# Configure and make invoke from build_external_libs.sh.
#
# Usage: source "<repo>/scripts/mise_env.sh"

if ! command -v mise >/dev/null 2>&1; then
  echo "error: mise not found on PATH (https://mise.jdx.dev/getting-started.html)" >&2
  exit 1
fi

_hc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mise trust --quiet "${_hc_repo_root}/mise.toml"
mise install --cd "${_hc_repo_root}" --yes
eval "$(mise env --cd "${_hc_repo_root}" -s bash)"
unset _hc_repo_root
echo "==> zig $(zig version) ($(command -v zig))"
