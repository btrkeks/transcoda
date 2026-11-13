use clap::{Parser, Subcommand};
use commands::serialize::{serialize_dir_command, SerializeDirArgs};
mod io;
pub use io::exit_code;
mod commands;
mod preset;
mod progress;
use commands::concatenate_dir::{concatenate_dir_command, ConcatenateDirArgs};
use commands::generate_layout_targets::{
    generate_layout_targets_command, GenerateLayoutTargetsArgs,
};
use commands::normalize_dir::{normalize_command, NormalizeArgs};
use commands::quality_filter::{quality_filter_command, QualityFilterArgs};
use commands::render_settings_search::{render_settings_search_command, RenderSettingsSearchArgs};
use commands::render_settings_search::{
    render_settings_search_worker_command, RenderSettingsSearchWorkerArgs,
};
use commands::serialize::{serialize_jsonl_command, SerializeJsonlArgs};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    #[command(name = "serialize-dir")]
    SerializeDir(SerializeDirArgs),
    #[command(name = "normalize")]
    Normalize(NormalizeArgs),
    #[command(name = "quality-filter-dir")]
    QualityFilter(QualityFilterArgs),
    #[command(name = "concatenate-dir")]
    ConcatenateDir(ConcatenateDirArgs),
    #[command(name = "render-settings-search")]
    RenderSettingsSearch(RenderSettingsSearchArgs),
    #[command(name = "__render-settings-search-worker", hide = true)]
    RenderSettingsSearchWorker(RenderSettingsSearchWorkerArgs),
    #[command(name = "generate-layout-targets")]
    GenerateLayoutTargets(GenerateLayoutTargetsArgs),
    #[command(name = "serialize-jsonl")]
    SerializeJsonl(SerializeJsonlArgs),
}
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::SerializeDir(args) => serialize_dir_command(args)?,
        Command::Normalize(args) => normalize_command(args)?,
        Command::QualityFilter(args) => quality_filter_command(args)?,
        Command::ConcatenateDir(args) => concatenate_dir_command(args)?,
        Command::RenderSettingsSearch(args) => render_settings_search_command(args)?,
        Command::RenderSettingsSearchWorker(args) => render_settings_search_worker_command(args)?,
        Command::GenerateLayoutTargets(args) => generate_layout_targets_command(args)?,
        Command::SerializeJsonl(args) => serialize_jsonl_command(args)?,
    }
    Ok(())
}
