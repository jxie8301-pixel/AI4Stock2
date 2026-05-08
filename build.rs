fn main() {
    println!("cargo:rustc-link-arg-bin=ai4stock-train=-Wl,--disable-new-dtags");
    println!(
        "cargo:rustc-link-arg-bin=ai4stock-train=-Wl,-rpath,$ORIGIN/../../.pixi/envs/default/lib"
    );
}
