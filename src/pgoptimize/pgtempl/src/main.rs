use clap::builder::PossibleValue;
use clap::Arg;
use clap::ValueEnum;
use handlebars::Handlebars;
use serde::Serialize;
use std::process::Command;

#[macro_use]
extern crate clap;

const EXE_ARG: &str = "exe";
const LANG_ARG: &str = "lang";

fn main() {
    let app = build_cli();
    let matches = app.get_matches();

    let default = "hc".to_string();
    let executable = matches.get_one::<String>(EXE_ARG).unwrap_or(&default);
    let language = matches.get_one::<Lang>(LANG_ARG).unwrap_or(&Lang::Csharp);
    let template = match language {
        Lang::Csharp => TEMPLATE_CSHARP,
        Lang::Rust => TEMPLATE_RUST,
    };

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
        .render_template(template, &data)
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

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    #[default]
    Csharp,
    Rust,
}

impl ValueEnum for Lang {
    fn value_variants<'a>() -> &'a [Self] {
        &[Lang::Csharp, Lang::Rust]
    }

    fn to_possible_value<'a>(&self) -> Option<PossibleValue> {
        Some(match self {
            Lang::Csharp => PossibleValue::new("csharp"),
            Lang::Rust => PossibleValue::new("rust"),
        })
    }
}

fn build_cli() -> clap::Command {
    #![allow(non_upper_case_globals)]
    command!(crate_name!())
        .version(crate_version!())
        .author("egoroff <egoroff@gmail.com>")
        .about("PGO template tool")
        .arg(
            Arg::new(LANG_ARG)
                .long(LANG_ARG)
                .short('l')
                .help("Language to generate")
                .value_parser(value_parser!(Lang))
                .required(false),
        )
        .arg(
            Arg::new(EXE_ARG)
                .long(EXE_ARG)
                .short('e')
                .help("Executable path")
                .required(false),
        )
}

const TEMPLATE_CSHARP: &str = r###"
/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2009-2026 Alexander Egorov
 */

namespace _tst.net;

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
"###;

const TEMPLATE_RUST: &str = r###"
pub trait Hash {
    /// Gets the hash of "123" string.
    const HASH_STRING: &'static str;

    /// Gets empty string hash.
    const EMPTY_STRING_HASH: &'static str;

    /// Gets the hash of "12" string.
    const START_PART_STRING_HASH: &'static str;

    /// Gets the hash of "2" string.
    const MIDDLE_PART_STRING_HASH: &'static str;

    /// Gets the hash of "23" string.
    const TRAIL_PART_STRING_HASH: &'static str;

    /// Algorithm name.
    const ALGORITHM: &'static str;

    /// Initial string (default is "123").
    fn initial_string() -> &'static str {
        "123"
    }
}
{{#each hashes}}

pub struct {{ class }};

impl Hash for {{ class }} {
    const HASH_STRING: &'static str = "{{ hash123 }}";
    const EMPTY_STRING_HASH: &'static str = "{{ hash_empty }}";
    const START_PART_STRING_HASH: &'static str = "{{ hash_start }}";
    const MIDDLE_PART_STRING_HASH: &'static str = "{{ hash_middle }}";
    const TRAIL_PART_STRING_HASH: &'static str = "{{ hash_trail }}";
    const ALGORITHM: &'static str = "{{ algo }}";
}
{{/each}}
"###;
