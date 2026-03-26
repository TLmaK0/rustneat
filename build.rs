fn main() {
    // Only run for the openai feature (which uses PyO3)
    #[cfg(feature = "openai")]
    {
        use std::process::Command;

        // Get Python library path from python3-config
        let output = Command::new("python3")
            .args(&[
                "-c",
                "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
            ])
            .output()
            .expect("Failed to get Python library path");

        let libdir = String::from_utf8(output.stdout)
            .expect("Invalid UTF-8 from Python")
            .trim()
            .to_string();

        // Tell the linker to add this path to rpath so the binary can find libpython at runtime
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir);

        // Also add standard Python lib search path
        println!("cargo:rustc-link-search=native={}", libdir);
    }
}
