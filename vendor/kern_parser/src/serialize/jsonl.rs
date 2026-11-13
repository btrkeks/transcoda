use std::io::{BufRead, Write};
use std::panic::{self, AssertUnwindSafe};

use rayon::prelude::*;
use serde_json::{Map, Value};

use crate::{KernDocument, ParseOptions, SerializeOptions, SerializePreset};

#[derive(Debug, Clone)]
pub struct SerializeJsonlOptions {
    pub preset: SerializePreset,
    pub workers: usize,
    pub text_fields: Vec<String>,
    pub id_field: String,
    pub fail_fast: bool,
    pub parse_options: ParseOptions,
    pub serialize_options: SerializeOptions,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProcessJsonlReport {
    pub processed: usize,
    pub rejected: usize,
    pub execution_failures: usize,
}

pub fn process_jsonl<R, W>(
    reader: R,
    mut writer: W,
    options: &SerializeJsonlOptions,
) -> Result<ProcessJsonlReport, Box<dyn std::error::Error>>
where
    R: BufRead,
    W: Write,
{
    let mut report = ProcessJsonlReport::default();
    if options.fail_fast || options.workers <= 1 {
        for (line_index, line) in reader.lines().enumerate() {
            let outcome = serialize_jsonl_line(line_index + 1, &line?, options);
            write_jsonl_row(&mut writer, &outcome.row)?;
            report.processed += 1;
            report.rejected += usize::from(!outcome.success);
            report.execution_failures += usize::from(outcome.execution_failure);
            if options.fail_fast && !outcome.success {
                break;
            }
        }
        writer.flush()?;
        return Ok(report);
    }

    let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let outcomes = pool.install(|| {
        lines
            .par_iter()
            .enumerate()
            .map(|(line_index, line)| serialize_jsonl_line(line_index + 1, line, options))
            .collect::<Vec<_>>()
    });

    for outcome in outcomes {
        write_jsonl_row(&mut writer, &outcome.row)?;
        report.processed += 1;
        report.rejected += usize::from(!outcome.success);
        report.execution_failures += usize::from(outcome.execution_failure);
    }
    writer.flush()?;

    Ok(report)
}

#[derive(Debug)]
struct JsonlRowOutcome {
    row: Map<String, Value>,
    success: bool,
    execution_failure: bool,
}

fn serialize_jsonl_line(
    line_number: usize,
    line: &str,
    options: &SerializeJsonlOptions,
) -> JsonlRowOutcome {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        serialize_jsonl_line_inner(line_number, line, options)
    }));

    match result {
        Ok(outcome) => outcome,
        Err(payload) => {
            let row = serde_json::from_str::<Value>(line)
                .ok()
                .and_then(|value| match value {
                    Value::Object(row) => Some(row),
                    _ => None,
                })
                .unwrap_or_default();
            let mut outcome = jsonl_failure_row(
                row,
                &options.text_fields,
                format!("panic: {}", panic_message(payload)),
            );
            outcome.execution_failure = true;
            outcome
        }
    }
}

fn serialize_jsonl_line_inner(
    line_number: usize,
    line: &str,
    options: &SerializeJsonlOptions,
) -> JsonlRowOutcome {
    let value = match serde_json::from_str::<Value>(line) {
        Ok(value) => value,
        Err(error) => {
            return jsonl_failure_row(
                Map::new(),
                &options.text_fields,
                format!("invalid JSON on line {line_number}: {error}"),
            );
        }
    };
    let mut row = match value {
        Value::Object(row) => row,
        _ => {
            return jsonl_failure_row(
                Map::new(),
                &options.text_fields,
                format!("line {line_number}: JSONL row must be an object"),
            );
        }
    };

    if !row.contains_key(&options.id_field) {
        return jsonl_failure_row(
            row,
            &options.text_fields,
            format!("missing id field `{}`", options.id_field),
        );
    }

    let text_fields = normalized_text_fields(&options.text_fields);
    let multi_field = text_fields.len() > 1;
    let mut errors = Vec::new();
    let mut execution_failure = false;

    for text_field in &text_fields {
        match serialize_text_field(&mut row, text_field, options) {
            Ok(()) => {
                if multi_field {
                    row.insert(format!("{text_field}_ok"), Value::Bool(true));
                    row.insert(format!("{text_field}_error"), Value::Null);
                }
            }
            Err(error) => {
                execution_failure |= error.execution_failure;
                let error = error.message;
                row.insert(text_field.clone(), Value::Null);
                if multi_field {
                    row.insert(format!("{text_field}_ok"), Value::Bool(false));
                    row.insert(format!("{text_field}_error"), Value::String(error.clone()));
                }
                if multi_field {
                    errors.push(format!("{text_field}: {error}"));
                } else {
                    errors.push(error);
                }
            }
        }
    }

    if errors.is_empty() {
        row.insert("ok".to_owned(), Value::Bool(true));
        row.insert("error".to_owned(), Value::Null);
    } else {
        row.insert("ok".to_owned(), Value::Bool(false));
        row.insert("error".to_owned(), Value::String(errors.join("; ")));
    }

    let success = errors.is_empty();
    JsonlRowOutcome {
        row,
        success,
        execution_failure,
    }
}

fn jsonl_failure_row(
    mut row: Map<String, Value>,
    text_fields: &[String],
    error: String,
) -> JsonlRowOutcome {
    for text_field in normalized_text_fields(text_fields) {
        row.insert(text_field, Value::Null);
    }
    row.insert("ok".to_owned(), Value::Bool(false));
    row.insert("error".to_owned(), Value::String(error));

    JsonlRowOutcome {
        row,
        success: false,
        execution_failure: false,
    }
}

fn normalized_text_fields(text_fields: &[String]) -> Vec<String> {
    let mut fields = text_fields
        .iter()
        .filter(|field| !field.is_empty())
        .cloned()
        .collect::<Vec<_>>();
    if fields.is_empty() {
        fields.push("transcription".to_owned());
    }
    fields
}

struct FieldError {
    message: String,
    execution_failure: bool,
}

impl FieldError {
    fn rejected(message: String) -> Self {
        Self {
            message,
            execution_failure: false,
        }
    }
}

fn serialize_text_field(
    row: &mut Map<String, Value>,
    text_field: &str,
    options: &SerializeJsonlOptions,
) -> Result<(), FieldError> {
    let input = match row.get(text_field) {
        Some(Value::String(input)) => input.clone(),
        Some(_) => {
            return Err(FieldError::rejected(format!(
                "text field `{text_field}` must be a string"
            )));
        }
        None => {
            return Err(FieldError::rejected(format!(
                "missing text field `{text_field}`"
            )));
        }
    };

    let document = KernDocument::parse_with_options(&input, options.parse_options)
        .map_err(|error| FieldError::rejected(format!("parse error: {error}")))?;
    let serialized = options
        .preset
        .try_serialize_document_with_options(&document, options.serialize_options)
        .map_err(|error| FieldError {
            execution_failure: error.is_execution_failure(),
            message: format!("serialization error: {error}"),
        })?;

    row.insert(text_field.to_owned(), Value::String(serialized));
    Ok(())
}

fn write_jsonl_row<W: Write>(
    writer: &mut W,
    row: &Map<String, Value>,
) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "serialization panicked".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_music_is_rejected_without_losing_row_metadata() {
        let input = serde_json::json!({
            "row_index": 17,
            "metadata": { "piece": "fixture" },
            "transcription": "**kern\t**kern\n*M4/4\t*M4/4\n2c\t4e\n*^\t*\n.\t4d\t4f\n*-\t*-\t*-\n",
        });
        let options = SerializeJsonlOptions {
            preset: SerializePreset::TranscodaV2,
            workers: 1,
            text_fields: vec!["transcription".into()],
            id_field: "row_index".into(),
            fail_fast: false,
            parse_options: ParseOptions::default(),
            serialize_options: SerializeOptions::default(),
        };
        let text = format!("{input}\n");
        let mut output = Vec::new();
        let report = process_jsonl(std::io::Cursor::new(text), &mut output, &options).unwrap();
        assert_eq!(
            report,
            ProcessJsonlReport {
                processed: 1,
                rejected: 1,
                execution_failures: 0
            }
        );
        let row: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(row["row_index"], 17);
        assert_eq!(row["metadata"], serde_json::json!({ "piece": "fixture" }));
        assert_eq!(row["ok"], false);
        assert!(row["transcription"].is_null());
    }
}
