use std::fs;
use std::path::{Path, PathBuf};

use spine_checker::{SpineCheckError, SpineChecker};

#[test]
fn accepts_all_fixtures_in_accepted_directory() {
    let checker = SpineChecker::new();

    for path in collect_fixture_paths("accepted") {
        let snippet = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));

        if let Err(error) = checker.check_snippet(&snippet) {
            panic!(
                "expected fixture {} to be accepted, but got: {error}",
                path.display()
            );
        }
    }
}

#[test]
fn rejects_all_fixtures_in_rejected_directory() {
    let checker = SpineChecker::new();

    for path in collect_fixture_paths("rejected") {
        let snippet = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));

        match checker.check_snippet(&snippet) {
            Ok(()) => panic!(
                "expected fixture {} to be rejected, but it was accepted",
                path.display()
            ),
            Err(SpineCheckError::CanonicalViolation(_)) => {}
            Err(SpineCheckError::NotYetImplemented) => panic!(
                "fixture {} reached the placeholder implementation; replace it with real validation",
                path.display()
            ),
        }
    }
}

fn collect_fixture_paths(group: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(group);

    let mut paths = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", root.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| path.extension().is_some_and(|extension| extension == "krn"))
        .collect::<Vec<_>>();

    paths.sort();
    paths
}
