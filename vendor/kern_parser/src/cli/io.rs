use std::{error::Error, io};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct Rejected(pub String);

pub(crate) fn accepted(success: bool, message: &str) -> Result<(), Box<dyn Error>> {
    if success {
        Ok(())
    } else {
        Err(Box::new(Rejected(message.to_owned())))
    }
}

/// Process status for a rejected request versus an execution failure.
pub fn exit_code(error: &(dyn Error + 'static)) -> u8 {
    if error
        .downcast_ref::<io::Error>()
        .is_some_and(|error| error.kind() == io::ErrorKind::InvalidData)
    {
        return 1;
    }
    if let Some(error) = error.downcast_ref::<crate::SerializeError>() {
        return if error.is_execution_failure() { 2 } else { 1 };
    }
    if let Some(error) = error.downcast_ref::<crate::document::DocumentError>() {
        return if error.is_execution_failure() { 2 } else { 1 };
    }
    if error.is::<Rejected>() || error.is::<crate::ParseError>() {
        1
    } else {
        2
    }
}
