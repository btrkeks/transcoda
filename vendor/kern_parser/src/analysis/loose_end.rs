use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LooseEnd {
    pub spanner: Spanner,
    pub side: LooseEndSide,
    pub staff: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Spanner {
    Slur,
    Tie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LooseEndSide {
    Open,
    Close,
}
