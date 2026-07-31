//! Entry point for the `hc` executable.
//!
//! HC application entry point. Owns process setup: stdout buffering, the interrupt
//! handler (stops a brute-force crack on Ctrl+C so the main loop prints the same
//! timing summary as the C release) and dispatch to the CLI (cli.zig) which
//! mirrors the former configuration.c CLI.

const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const bf = @import("bf");
const hashes = @import("hashes");
const cli = @import("cli.zig");

/// Set only from the SIGINT / console handler. The handler does nothing else
/// that is not async-signal-safe: no I/O, no allocation, no process exit.
/// The main thread observes it after `cli.run` returns (a crack prints its own
/// timing summary via the buffered writer before returning) and performs the
/// flush + exit on the main thread.
var g_interrupted: std.atomic.Value(bool) = .init(false);

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
            // Signal the main loop, then let the default handler run. The brute
            // force workers poll the shared "found" flag, so this makes them
            // stop quickly; the main thread prints timings + exits.
            g_interrupted.store(true, .release);
            bf.signalStopCrack();
            return .FALSE;
        }

        fn install() void {
            _ = SetConsoleCtrlHandler(onConsoleCtrl, .TRUE);
        }
    },
    .linux, .macos => struct {
        fn onInterrupt(sig: std.posix.SIG) callconv(.c) void {
            _ = sig;
            // Async-signal-safe: one relaxed atomic store + set the shared
            // brute-force stop flag. No I/O / exit here — that would deadlock
            // if the main thread held the stdio or arena lock when interrupted.
            g_interrupted.store(true, .release);
            bf.signalStopCrack();
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
    lib.setupConsoleUtf8();
    installSignalHandler();
    // Before any OpenSSL digest: activate SHA-NI / ASM via OPENSSL_ia32cap_P.
    hashes.ensureOpenSslReady();

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    var out = &stdout_writer.interface;
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
                out.print("hc: {s}\n", .{@errorName(err)}) catch {};
                std.process.exit(1);
            },
        }
    };

    out.flush() catch {};

    // A Ctrl+C during a brute-force crack sets the flag (and the workers stop
    // promptly via the shared found flag). The crack's timing summary was
    // already printed through the buffered writer above; exit cleanly.
    if (g_interrupted.load(.acquire)) {
        std.process.exit(0);
    }

    switch (outcome) {
        .ok => {},
        .invalid_command, .invalid_options => std.process.exit(1),
    }
}

test {
    _ = @import("cli.zig");
}
