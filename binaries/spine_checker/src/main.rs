use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use spine_checker::SpineChecker;

#[derive(Debug, Parser)]
#[command(name = "spine_checker")]
#[command(about = "Checks strict canonical spine correctness for Humdrum **kern snippets")]
struct Args {
    /// Read the snippet from stdin instead of a file path.
    #[arg(long)]
    stdin: bool,

    /// Optional path to a snippet file.
    input: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let snippet = match read_snippet(&args) {
        Ok(snippet) => snippet,
        Err(error) => {
            eprintln!("failed to read snippet: {error}");
            return ExitCode::FAILURE;
        }
    };

    match SpineChecker::new().check_snippet(&snippet) {
        Ok(()) => {
            println!("OK");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn read_snippet(args: &Args) -> io::Result<String> {
    if args.stdin || args.input.is_none() {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        return Ok(buffer);
    }

    fs::read_to_string(args.input.as_ref().expect("input path should be present"))
}
