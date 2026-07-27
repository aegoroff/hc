//! Entry point for the `hc` executable.
//!
//! HC application entry point. Owns process setup: stdout buffering, the interrupt
//! handler (prints brute-force timings when interrupting a hash restore) and
//! dispatch to the CLI (cli.zig) which mirrors the former configuration.c CLI.

const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const bf = @import("bf");
const cli = @import("cli.zig");

/// Pointer to the process' stdout writer so the interrupt handler can flush a
/// best-effort timing line. Safe to read from a signal/console handler
/// (pointer is stable for the whole process lifetime once main sets it).
var g_out: ?*std.Io.Writer = null;

fn printHashInterruptTimings() void {
    if (cli.active_mode != .hash) return;
    const out = g_out orelse return;
    lib.stopTimer();
    bf.outputTimings(out, bf.getAttempts(), lib.readElapsedTime()) catch {};
    out.flush() catch {};
}

const interrupt_install = switch (builtin.os.tag) {
    .windows => struct {
        const windows = std.os.windows;
        const CTRL_C_EVENT: windows.DWORD = 0;

        extern "kernel32" fn SetConsoleCtrlHandler(
            HandlerRoutine: ?*const fn (windows.DWORD) callconv(.winapi) windows.BOOL,
            Add: windows.BOOL,
        ) callconv(.winapi) windows.BOOL;

        fn onConsoleCtrl(ctrl_type: windows.DWORD) callconv(.winapi) windows.BOOL {
            if (ctrl_type != CTRL_C_EVENT) return .FALSE;
            printHashInterruptTimings();
            // Match classic hc.c: return FALSE so the default handler terminates
            // the process. Do not ExitProcess from this thread — that can
            // deadlock if the main thread holds locks (loader/heap/stdio).
            return .FALSE;
        }

        fn install() void {
            _ = SetConsoleCtrlHandler(onConsoleCtrl, .TRUE);
        }
    },
    .linux, .macos => struct {
        fn onInterrupt(sig: std.posix.SIG) callconv(.c) void {
            _ = sig;
            printHashInterruptTimings();
            std.process.exit(0);
        }

        fn install() void {
            const empty = std.posix.sigemptyset();
            const sa = std.posix.Sigaction{
                .handler = .{ .handler = onInterrupt },
                .mask = empty,
                .flags = 0,
            };
            std.posix.sigaction(std.posix.SIG.INT, &sa, null);
        }
    },
    else => struct {
        fn install() void {}
    },
};

/// Installs SIGINT on POSIX and SetConsoleCtrlHandler on Windows so Ctrl+C
/// during hash restore prints the same timing summary as the C release.
fn installSignalHandler() void {
    interrupt_install.install();
}

pub fn main(init: std.process.Init) !void {
    installSignalHandler();

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    var out = &stdout_writer.interface;
    g_out = out;
    defer {
        out.flush() catch {};
    }

    const allocator = init.arena.allocator();

    const args = try init.minimal.args.toSlice(allocator);

    const outcome = cli.run(allocator, init.io, out, args[1..]) catch |err| {
        out.flush() catch {};
        switch (err) {
            // Modes abort with these on invalid input (e.g. unknown hash);
            // they have already printed a user-facing message.
            error.UnknownHash,
            error.InvalidArgument,
            => std.process.exit(1),
            else => {
                std.debug.print("hc: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            },
        }
    };

    out.flush() catch {};

    switch (outcome) {
        .ok => {},
        .invalid_command, .invalid_options => std.process.exit(1),
    }
}
