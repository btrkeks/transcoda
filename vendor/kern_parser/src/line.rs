use crate::ParseError;

pub(crate) fn split_spine_tokens(line_number: usize, raw: &str) -> Result<Vec<&str>, ParseError> {
    raw.split('\t')
        .enumerate()
        .map(|(column, token)| {
            if token.is_empty() {
                Err(ParseError::EmptySpineToken {
                    line: line_number,
                    column: column + 1,
                })
            } else {
                Ok(token)
            }
        })
        .collect()
}

pub(crate) fn split_spine_tokens_preserving_empty(raw: &str) -> Vec<&str> {
    raw.split('\t').collect()
}
