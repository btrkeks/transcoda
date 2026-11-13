use crate::{
    DynamContent, FingContent, KernContent, MusicalDuration, MxmlHarmonyContent, Rest,
    SpineContent, TextContent,
};

pub(crate) fn plain_invisible_rest_duration(content: &SpineContent) -> Option<MusicalDuration> {
    let SpineContent::Kern(KernContent::Rest(rest)) = content else {
        return None;
    };
    is_plain_invisible_rest(rest).then(|| MusicalDuration::from_duration(&rest.duration))
}

pub(crate) fn plain_invisible_rest_mut(content: &mut SpineContent) -> Option<&mut Rest> {
    let SpineContent::Kern(KernContent::Rest(rest)) = content else {
        return None;
    };
    is_plain_invisible_rest(rest).then_some(rest)
}

pub(crate) fn is_plain_invisible_rest(rest: &Rest) -> bool {
    rest.invisible
        && rest.cross_staff.is_none()
        && rest.placement.is_none()
        && rest.breath.is_none()
        && rest.fermata.is_none()
        && rest.slurs.is_empty()
        && rest.phrases.is_empty()
}

pub(crate) fn is_kern_null(content: &SpineContent) -> bool {
    matches!(content, SpineContent::Kern(KernContent::Null))
}

pub(crate) fn is_non_kern_data_null(content: &SpineContent) -> bool {
    matches!(
        content,
        SpineContent::Dynam(DynamContent::Null)
            | SpineContent::Fing(FingContent::Null)
            | SpineContent::Text(TextContent::Null)
            | SpineContent::MxmlHarmony(MxmlHarmonyContent::Null)
    )
}
