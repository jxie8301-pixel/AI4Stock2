use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    let Some(conda_prefix) = env::var_os("CONDA_PREFIX") else {
        return;
    };
    let lib_dir = PathBuf::from(conda_prefix).join("lib");
    println!("cargo:rustc-link-arg-bin=ai4stock-train=-Wl,--disable-new-dtags");
    println!(
        "cargo:rustc-link-arg-bin=ai4stock-train=-Wl,-rpath,{}",
        lib_dir.display()
    );
}
