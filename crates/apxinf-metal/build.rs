fn main() {
    println!("cargo:rerun-if-changed=src/metal_w8_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_mlp_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_gdn_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_linear_layer_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_linear_layer_stack3_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_mlp_stack3_boundary_v1_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8_tail_mlp_head_v1_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_gdn_recurrent_count18_profile_v1_bridge.mm");
    println!("cargo:rerun-if-changed=src/metal_w8.metal");
    println!("cargo:rerun-if-changed=src/metal_w8_matvec.metal");
    println!("cargo:rerun-if-changed=src/metal_w8_mlp.metal");
    println!("cargo:rerun-if-changed=src/metal_w8_gdn.metal");
    println!("cargo:rerun-if-changed=src/metal_w8_gdn_out_g32.metal");
    println!("cargo:rerun-if-changed=src/metal_w8_linear_layer.metal");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let shader =
        std::fs::read_to_string("src/metal_w8.metal").expect("read the Metal W8 shader source");
    const DELIMITER: &str = "APX_METAL";
    assert!(
        !shader.contains(&format!("){}\"", DELIMITER)),
        "Metal shader contains the generated C++ raw-string delimiter"
    );
    let output_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
    std::fs::write(
        output_dir.join("metal_w8_source.inc"),
        format!("constexpr const char *kMetalSource = R\"{DELIMITER}({shader}){DELIMITER}\";\n"),
    )
    .expect("write the generated Metal shader include");

    let matvec_shader = std::fs::read_to_string("src/metal_w8_matvec.metal")
        .expect("read the Metal W8 matvec shader source");
    assert!(
        !matvec_shader.contains(&format!("){}\"", DELIMITER)),
        "Metal matvec shader contains the generated C++ raw-string delimiter"
    );
    std::fs::write(
        output_dir.join("metal_w8_matvec_source.inc"),
        format!(
            "constexpr const char *kMetalMatVecSource = R\"{DELIMITER}({matvec_shader}){DELIMITER}\";\n"
        ),
    )
    .expect("write the generated Metal matvec shader include");

    let mlp_shader = std::fs::read_to_string("src/metal_w8_mlp.metal")
        .expect("read the Metal W8 MLP shader source");
    assert!(
        !mlp_shader.contains(&format!("){}\"", DELIMITER)),
        "Metal MLP shader contains the generated C++ raw-string delimiter"
    );
    std::fs::write(
        output_dir.join("metal_w8_mlp_source.inc"),
        format!(
            "constexpr const char *kMetalMlpSource = R\"{DELIMITER}({mlp_shader}){DELIMITER}\";\n"
        ),
    )
    .expect("write the generated Metal MLP shader include");

    let gdn_shader = std::fs::read_to_string("src/metal_w8_gdn.metal")
        .expect("read the Metal W8 GDN shader source");
    assert!(
        !gdn_shader.contains(&format!("){}\"", DELIMITER)),
        "Metal GDN shader contains the generated C++ raw-string delimiter"
    );
    std::fs::write(
        output_dir.join("metal_w8_gdn_source.inc"),
        format!(
            "constexpr const char *kMetalGdnSource = R\"{DELIMITER}({gdn_shader}){DELIMITER}\";\n"
        ),
    )
    .expect("write the generated Metal GDN shader include");

    let linear_layer_shader = std::fs::read_to_string("src/metal_w8_linear_layer.metal")
        .expect("read the Metal W8 linear-layer shader source");
    let gdn_out_g32_shader = std::fs::read_to_string("src/metal_w8_gdn_out_g32.metal")
        .expect("read the Metal W8 GDN-output-G32 shader source");
    assert!(
        !linear_layer_shader.contains(&format!("){}\"", DELIMITER)),
        "Metal linear-layer shader contains the generated C++ raw-string delimiter"
    );
    let combined_linear_layer_shader =
        format!("{gdn_shader}\n{mlp_shader}\n{linear_layer_shader}\n{gdn_out_g32_shader}");
    std::fs::write(
        output_dir.join("metal_w8_linear_layer_source.inc"),
        format!(
            "constexpr const char *kMetalLinearLayerSource = R\"{DELIMITER}({combined_linear_layer_shader}){DELIMITER}\";\n"
        ),
    )
    .expect("write the generated Metal linear-layer shader include");

    // Tail v1 composes the existing RMS/residual, MLP, and top-4 kernels in
    // one library. The kernel files above remain the only shader source.
    let combined_tail_mlp_head_shader = format!("{mlp_shader}\n{linear_layer_shader}\n{shader}");
    std::fs::write(
        output_dir.join("metal_w8_tail_mlp_head_v1_source.inc"),
        format!(
            "constexpr const char *kMetalTailMlpHeadSourceV1 = R\"{DELIMITER}({combined_tail_mlp_head_shader}){DELIMITER}\";\n"
        ),
    )
    .expect("write the generated Metal tail MLP+head v1 shader include");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_mlp_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_mlp_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_gdn_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_gdn_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_linear_layer_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_linear_layer_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_linear_layer_stack3_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_linear_layer_stack3_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_mlp_stack3_boundary_v1_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_mlp_stack3_boundary_v1_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_w8_tail_mlp_head_v1_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_w8_tail_mlp_head_v1_bridge");

    cc::Build::new()
        .cpp(true)
        .file("src/metal_gdn_recurrent_count18_profile_v1_bridge.mm")
        .include(&output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .compile("apxinf_metal_gdn_recurrent_count18_profile_v1_bridge");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
