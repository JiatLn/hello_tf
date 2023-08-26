fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/infer.proto");
    tonic_build::configure()
        .out_dir("src/")
        .compile(&["proto/infer.proto"], &["proto"])?;
    Ok(())
}
