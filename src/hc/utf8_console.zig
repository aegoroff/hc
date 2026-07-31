const builtin = @import("builtin");

/// Set Windows console input/output code page to UTF-8 so digests and paths
/// print correctly. No-op on non-Windows.
pub fn setupConsole() void {
    if (comptime builtin.os.tag != .windows) return;
    const kernel32 = struct {
        extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
        extern "kernel32" fn SetConsoleCP(wCodePageID: u32) callconv(.winapi) i32;
    };
    _ = kernel32.SetConsoleOutputCP(65001);
    _ = kernel32.SetConsoleCP(65001);
}
