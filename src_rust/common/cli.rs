use std::path::Path;

pub fn next_arg(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

pub fn split_value(value: &str, option: &str) -> Result<String, String> {
    let raw = value
        .split_once('=')
        .map(|(_, right)| right)
        .unwrap_or_default();
    if raw.is_empty() {
        Err(format!("missing value for {option}"))
    } else {
        Ok(raw.to_owned())
    }
}

pub fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub fn display_command(command: &[String]) -> String {
    command
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '='))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}
