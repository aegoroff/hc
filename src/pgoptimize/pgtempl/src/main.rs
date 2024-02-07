use clap::{crate_name, crate_version, App, Arg};
use handlebars::Handlebars;
use serde::Serialize;
use std::process::Command;

fn main() {
    let app = build_cli();
    let matches = app.get_matches();

    let executable = matches.value_of("exe").unwrap_or("hc");

    let hashes = [
        "crc32",
        "crc32c",
        "md4",
        "md5",
        "sha1",
        "sha256",
        "sha384",
        "sha512",
        "whirlpool",
        "md2",
        "sha224",
        "tiger",
        "tiger2",
        "ripemd128",
        "ripemd160",
        "ripemd256",
        "ripemd320",
        "gost",
        "snefru256",
        "snefru128",
        "tth",
        "haval-128-3",
        "haval-128-4",
        "haval-128-5",
        "haval-160-3",
        "haval-160-4",
        "haval-160-5",
        "haval-192-3",
        "haval-192-4",
        "haval-192-5",
        "haval-224-3",
        "haval-224-4",
        "haval-224-5",
        "haval-256-3",
        "haval-256-4",
        "haval-256-5",
        "edonr256",
        "edonr512",
        "ntlm",
        "sha-3-224",
        "sha-3-256",
        "sha-3-384",
        "sha-3-512",
        "sha-3k-224",
        "sha-3k-256",
        "sha-3k-384",
        "sha-3k-512",
        "blake2b",
        "blake2s",
        "blake3",
    ];

    let hashes: Vec<Hash> = hashes
        .into_iter()
        .map(|algorithm| Hash {
            class: title(algorithm).replace("-", "_"),
            algo: algorithm.to_string(),
            hash123: calculate(executable, algorithm, "123"),
            hash_empty: calculate(executable, algorithm, ""),
            hash_start: calculate(executable, algorithm, "12"),
            hash_middle: calculate(executable, algorithm, "2"),
            hash_trail: calculate(executable, algorithm, "23"),
        })
        .collect();

    let data = Pgo { hashes };

    let reg = Handlebars::new();
    let res = reg
        .render_template(TEMPLATE, &data)
        .unwrap()
        .trim_start()
        .to_string();

    println!("{res}");
}

fn calculate(executable: &str, algorithm: &str, string_to_hash: &str) -> String {
    let mut process = Command::new(executable);
    let child = process
        .arg(algorithm)
        .arg("string")
        .arg("-s")
        .arg(string_to_hash);
    let out = child.output().unwrap();
    std::str::from_utf8(&out.stdout).unwrap().trim().to_string()
}

fn title(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}

#[derive(Serialize, Default, Debug)]
pub struct Hash {
    pub class: String,
    pub algo: String,
    pub hash123: String,
    pub hash_empty: String,
    pub hash_start: String,
    pub hash_middle: String,
    pub hash_trail: String,
}

#[derive(Serialize, Default, Debug)]
pub struct Pgo {
    pub hashes: Vec<Hash>,
}

fn build_cli() -> App<'static> {
    return App::new(crate_name!())
        .version(crate_version!())
        .author("egoroff <egoroff@gmail.com>")
        .about("PGO template tool")
        .arg(
            Arg::new("exe")
                .long("exe")
                .short('e')
                .takes_value(true)
                .help("Executable path")
                .required(false),
        );
}

const TEMPLATE: &str = r###"
/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2009-2024 Alexander Egorov
 */

namespace _tst.net
{
    public abstract class Hash
    {
        /// <summary>
        /// Gets the hash of "123" string
        /// </summary>
        public abstract string HashString { get; }

        public abstract string EmptyStringHash { get; }

        /// <summary>
        /// Gets the hash of "12" string
        /// </summary>
        public abstract string StartPartStringHash { get; }

        /// <summary>
        /// Gets the hash of "2" string
        /// </summary>
        public abstract string MiddlePartStringHash { get; }

        /// <summary>
        /// Gets the hash of "23" string
        /// </summary>
        public abstract string TrailPartStringHash { get; }

        public abstract string Algorithm { get; }

        public string InitialString => "123";
    }
    {{#each hashes}}

    public class {{ class }} : Hash
    {
        public override string HashString => "{{ hash123 }}";

        public override string EmptyStringHash => "{{ hash_empty }}";

        public override string StartPartStringHash => "{{ hash_start }}";

        public override string MiddlePartStringHash => "{{ hash_middle }}";

        public override string TrailPartStringHash => "{{ hash_trail }}";

        public override string Algorithm => "{{ algo }}";
    }
    {{/each}}
}
"###;
