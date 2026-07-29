extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
extern "kernel32" fn SetConsoleCP(wCodePageID: u32) callconv(.winapi) i32;

pub fn setupConsole() void {
    _ = SetConsoleOutputCP(65001);
    _ = SetConsoleCP(65001);
}
