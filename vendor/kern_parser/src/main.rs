use std::process::ExitCode;

use kern_parser::cli;

fn main() -> ExitCode {
    match cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::from(cli::exit_code(error.as_ref()))
        }
    }
}
