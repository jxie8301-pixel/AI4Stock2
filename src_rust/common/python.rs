use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::env;
use std::path::{Path, PathBuf};

pub fn prepare_python_path(python: Python<'_>) -> PyResult<()> {
    let repo_root = repo_root()?;
    let site_paths = current_python_site_paths(python)?;
    for path in site_paths.iter().rev() {
        prepend_sys_path(python, path)?;
    }
    for path in env_python_paths().iter().rev() {
        prepend_sys_path(python, path)?;
    }
    prepend_sys_path(python, &repo_root)?;
    Ok(())
}

fn repo_root() -> PyResult<PathBuf> {
    if let Some(root) = env::var_os("AI4STOCK_REPO_ROOT") {
        return Ok(PathBuf::from(root));
    }
    env::current_dir()
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))
}

fn current_python_site_paths(python: Python<'_>) -> PyResult<Vec<PathBuf>> {
    let sysconfig = python.import("sysconfig")?;
    let paths = sysconfig.call_method0("get_paths")?.cast_into::<PyDict>()?;
    let mut out = Vec::new();
    for key in ["purelib", "platlib"] {
        if let Some(value) = paths.get_item(key)? {
            let path = PathBuf::from(value.extract::<String>()?);
            if path.exists() && !out.iter().any(|existing| existing == &path) {
                out.push(path);
            }
        }
    }
    Ok(out)
}

fn env_python_paths() -> Vec<PathBuf> {
    env::var_os("AI4STOCK_PYTHONPATH")
        .map(|raw| env::split_paths(&raw).collect())
        .unwrap_or_default()
}

fn prepend_sys_path(python: Python<'_>, path: &Path) -> PyResult<()> {
    let path_text = path.to_string_lossy();
    let sys = python.import("sys")?;
    let sys_path = sys.getattr("path")?;
    if !sys_path
        .call_method1("__contains__", (path_text.as_ref(),))?
        .extract::<bool>()?
    {
        sys_path.call_method1("insert", (0, path_text.as_ref()))?;
    }
    Ok(())
}
