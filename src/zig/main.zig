//! Entry point for the `hc` executable.
//!
//! Replaces src/hc/hc.c. Owns process setup: stdout buffering, the SIGINT
//! handler (prints brute-force timings when interrupting a hash restore) and
//! dispatch to the CLI (cli.zig) which mirrors src/hc/configuration.c.

const std = @import("std");
const builtin = @import("builtin");
const lib = @import("lib");
const bf = @import("bf");
const cli = @import("cli.zig");

/// Pointer to the process' stdout writer so the SIGINT handler can flush a
/// best-effort timing line. Safe to read from a signal handler (pointer is
/// stable for the whole process lifetime once main sets it).
var g_out: ?*std.Io.Writer = null;

fn onInterrupt(sig: std.posix.SIG) callconv(.c) void {
    _ = sig;
    if (cli.active_mode == .hash) {
        const out = g_out orelse {
            std.process.exit(0);
        };
        lib.stopTimer();
        bf.outputTimings(out, bf.getAttempts(), lib.readElapsedTime()) catch {};
        out.flush() catch {};
    }
    std.process.exit(0);
}

/// Installs the SIGINT handler on POSIX. Windows would use SetConsoleCtrlHandler
/// (TODO: not wired here since the build targets Linux for now).
fn installSignalHandler() void {
    if (builtin.os.tag != .linux and builtin.os.tag != .macos) return;

    const empty = std.posix.sigemptyset();
    const sa = std.posix.Sigaction{
        .handler = .{ .handler = onInterrupt },
        .mask = empty,
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sa, null);
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
