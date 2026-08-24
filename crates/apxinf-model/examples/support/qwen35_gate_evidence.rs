use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Read};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const PROFILE_FORMAT: &str = "apxinf-hf-macos-deployment-profile-v1";
const PROFILE_ID: &str = "qwen35-0.8b-macos-cpu";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const PROFILE_RELATIVE_PATH: &str = "../../configs/hf-onboarding/qwen35-0.8b-macos-cpu.json";
const PROFILE_BYTES: &[u8] =
    include_bytes!("../../../../configs/hf-onboarding/qwen35-0.8b-macos-cpu.json");
const GENERAL_SOURCE_BYTES: &[u8] = include_bytes!("../../src/qwen35/general.rs");
const LLM_TRAIT_SOURCE_BYTES: &[u8] = include_bytes!("../../src/llm_trait.rs");
const GATE_EVIDENCE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_gate_evidence.rs");
const APXINF_METAL_LIB_SOURCE_BYTES: &[u8] = include_bytes!("../../../apxinf-metal/src/lib.rs");
const GDN_RUST_SOURCE_BYTES: &[u8] = include_bytes!("../../../apxinf-metal/src/gdn.rs");
const LINEAR_LAYER_RUST_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/linear_layer.rs");
const STACK3_RUST_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/linear_layer/stack3.rs");
const STACK3_BRIDGE_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_linear_layer_stack3_bridge.mm");
const METAL_W8_MLP_BRIDGE_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_mlp_bridge.mm");
const METAL_W8_HEAD_BRIDGE_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_bridge.mm");
const APXINF_METAL_BUILD_SOURCE_BYTES: &[u8] = include_bytes!("../../../apxinf-metal/build.rs");
const METAL_W8_HEAD_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8.metal");
const METAL_W8_MATVEC_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_matvec.metal");
const METAL_W8_GDN_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_gdn.metal");
const METAL_W8_MLP_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_mlp.metal");
const METAL_W8_LINEAR_LAYER_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_linear_layer.metal");
const METAL_W8_GDN_OUT_G32_SOURCE_BYTES: &[u8] =
    include_bytes!("../../../apxinf-metal/src/metal_w8_gdn_out_g32.metal");
const MAX_CACHE_ENTRIES: usize = 4096;

#[derive(Clone, Copy)]
struct BuildSourceSpec {
    receipt_name: &'static str,
    label: &'static str,
    manifest_relative_path: &'static str,
    embedded_bytes: &'static [u8],
    metal_shader: bool,
}

fn stack3_source_specs() -> [BuildSourceSpec; 9] {
    [
        BuildSourceSpec {
            receipt_name: "gate_evidence",
            label: "gate evidence source",
            manifest_relative_path: "examples/support/qwen35_gate_evidence.rs",
            embedded_bytes: GATE_EVIDENCE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "general",
            label: "general source",
            manifest_relative_path: "src/qwen35/general.rs",
            embedded_bytes: GENERAL_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "stack3_rust",
            label: "Metal stack3 Rust source",
            manifest_relative_path: "../apxinf-metal/src/linear_layer/stack3.rs",
            embedded_bytes: STACK3_RUST_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "stack3_bridge",
            label: "Metal stack3 bridge source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_linear_layer_stack3_bridge.mm",
            embedded_bytes: STACK3_BRIDGE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "apxinf_metal_build",
            label: "apxinf-metal build source",
            manifest_relative_path: "../apxinf-metal/build.rs",
            embedded_bytes: APXINF_METAL_BUILD_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_gdn",
            label: "Metal W8 GDN shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_gdn.metal",
            embedded_bytes: METAL_W8_GDN_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_mlp",
            label: "Metal W8 MLP shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_mlp.metal",
            embedded_bytes: METAL_W8_MLP_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_linear_layer",
            label: "Metal W8 linear-layer shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_linear_layer.metal",
            embedded_bytes: METAL_W8_LINEAR_LAYER_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_gdn_out_g32",
            label: "Metal W8 GDN-output-G32 shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_gdn_out_g32.metal",
            embedded_bytes: METAL_W8_GDN_OUT_G32_SOURCE_BYTES,
            metal_shader: true,
        },
    ]
}

fn stack3_lm_head_v2_source_specs() -> [BuildSourceSpec; 17] {
    [
        BuildSourceSpec {
            receipt_name: "gate_evidence",
            label: "gate evidence source",
            manifest_relative_path: "examples/support/qwen35_gate_evidence.rs",
            embedded_bytes: GATE_EVIDENCE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "general",
            label: "general source",
            manifest_relative_path: "src/qwen35/general.rs",
            embedded_bytes: GENERAL_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "llm_trait",
            label: "shared generation source",
            manifest_relative_path: "src/llm_trait.rs",
            embedded_bytes: LLM_TRAIT_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "apxinf_metal_lib",
            label: "apxinf-metal public API source",
            manifest_relative_path: "../apxinf-metal/src/lib.rs",
            embedded_bytes: APXINF_METAL_LIB_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "apxinf_metal_build",
            label: "apxinf-metal build source",
            manifest_relative_path: "../apxinf-metal/build.rs",
            embedded_bytes: APXINF_METAL_BUILD_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "gdn_rust",
            label: "Metal GDN Rust source",
            manifest_relative_path: "../apxinf-metal/src/gdn.rs",
            embedded_bytes: GDN_RUST_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "linear_layer_rust",
            label: "Metal linear-layer module source",
            manifest_relative_path: "../apxinf-metal/src/linear_layer.rs",
            embedded_bytes: LINEAR_LAYER_RUST_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "stack3_rust",
            label: "Metal Stack3 Rust source",
            manifest_relative_path: "../apxinf-metal/src/linear_layer/stack3.rs",
            embedded_bytes: STACK3_RUST_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "stack3_bridge",
            label: "Metal Stack3 bridge source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_linear_layer_stack3_bridge.mm",
            embedded_bytes: STACK3_BRIDGE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_mlp_bridge",
            label: "Metal W8 MLP bridge source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_mlp_bridge.mm",
            embedded_bytes: METAL_W8_MLP_BRIDGE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_head_bridge",
            label: "Metal W8 tied-head bridge source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_bridge.mm",
            embedded_bytes: METAL_W8_HEAD_BRIDGE_SOURCE_BYTES,
            metal_shader: false,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_gdn",
            label: "Metal W8 GDN shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_gdn.metal",
            embedded_bytes: METAL_W8_GDN_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_mlp",
            label: "Metal W8 MLP shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_mlp.metal",
            embedded_bytes: METAL_W8_MLP_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_linear_layer",
            label: "Metal W8 linear-layer shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_linear_layer.metal",
            embedded_bytes: METAL_W8_LINEAR_LAYER_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_gdn_out_g32",
            label: "Metal W8 GDN-output-G32 shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_gdn_out_g32.metal",
            embedded_bytes: METAL_W8_GDN_OUT_G32_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_head",
            label: "Metal W8 tied-head shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8.metal",
            embedded_bytes: METAL_W8_HEAD_SOURCE_BYTES,
            metal_shader: true,
        },
        BuildSourceSpec {
            receipt_name: "metal_w8_matvec",
            label: "Metal W8 matvec shader source",
            manifest_relative_path: "../apxinf-metal/src/metal_w8_matvec.metal",
            embedded_bytes: METAL_W8_MATVEC_SOURCE_BYTES,
            metal_shader: true,
        },
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileAttestation {
    pub path: PathBuf,
    pub size: u64,
    pub sha256: String,
    pub nlink: u64,
}

pub struct AttestedBytes {
    pub bytes: Vec<u8>,
    pub attestation: FileAttestation,
}

pub struct GateCustody {
    model_dir: PathBuf,
    cache_present: bool,
    profile_id: String,
    source_lock_content_sha256: String,
    source_lock_value: Value,
    profile: FileAttestation,
    source_lock: FileAttestation,
    binary: FileAttestation,
    gate_source: FileAttestation,
    source_closure: &'static str,
    metal_shader_receipt_key: &'static str,
    rust_sources: BTreeMap<String, FileAttestation>,
    metal_shader_sources: BTreeMap<String, FileAttestation>,
    model_artifacts: BTreeMap<String, FileAttestation>,
}

impl GateCustody {
    #[allow(dead_code)]
    pub fn capture(
        model_dir: &Path,
        source_lock_path: &Path,
        gate_source_name: &str,
        gate_source_build_bytes: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        Self::capture_inner(
            model_dir,
            source_lock_path,
            gate_source_name,
            gate_source_build_bytes,
            "gate-general-compile-inputs-v1",
        )
    }

    pub fn capture_stack3(
        model_dir: &Path,
        source_lock_path: &Path,
        gate_source_name: &str,
        gate_source_build_bytes: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        Self::capture_inner(
            model_dir,
            source_lock_path,
            gate_source_name,
            gate_source_build_bytes,
            "stack3-direct-compile-inputs-v1",
        )
    }

    pub fn capture_stack3_lm_head_v2(
        model_dir: &Path,
        source_lock_path: &Path,
        gate_source_name: &str,
        gate_source_build_bytes: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        Self::capture_inner(
            model_dir,
            source_lock_path,
            gate_source_name,
            gate_source_build_bytes,
            "stack3-lm-head-v2-direct-compile-inputs-v1",
        )
    }

    fn capture_inner(
        model_dir: &Path,
        source_lock_path: &Path,
        gate_source_name: &str,
        gate_source_build_bytes: &[u8],
        source_closure: &'static str,
    ) -> Result<Self, Box<dyn Error>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let profile_path = manifest_dir.join(PROFILE_RELATIVE_PATH);
        let profile_bytes = read_attested_bytes(&profile_path, "deployment profile")?;
        require_build_source_match(&profile_bytes.bytes, PROFILE_BYTES, "deployment profile")?;
        let profile_value: Value = serde_json::from_slice(&profile_bytes.bytes)?;
        let (profile_id, source_lock_content_sha256, expected_artifacts) =
            validate_profile(&profile_value)?;

        let source_lock_bytes = read_attested_bytes(source_lock_path, "source lock")?;
        let source_lock_value: Value = serde_json::from_slice(&source_lock_bytes.bytes)?;
        validate_source_lock(
            &source_lock_value,
            &source_lock_content_sha256,
            &expected_artifacts,
        )?;

        let (canonical_model_dir, cache_present, model_artifacts) =
            attest_model_dir(model_dir, &expected_artifacts)?;

        let gate_source_path = manifest_dir.join("examples").join(gate_source_name);
        let gate_source_bytes = read_attested_bytes(&gate_source_path, "gate source")?;
        require_build_source_match(
            &gate_source_bytes.bytes,
            gate_source_build_bytes,
            "gate source",
        )?;
        let mut rust_sources = BTreeMap::new();
        let mut metal_shader_sources = BTreeMap::new();
        if source_closure != "gate-general-compile-inputs-v1" {
            let specs = match source_closure {
                "stack3-direct-compile-inputs-v1" => stack3_source_specs().to_vec(),
                "stack3-lm-head-v2-direct-compile-inputs-v1" => {
                    stack3_lm_head_v2_source_specs().to_vec()
                }
                _ => return Err(invalid("unknown gate source closure")),
            };
            for spec in specs {
                let source = read_attested_bytes(
                    &manifest_dir.join(spec.manifest_relative_path),
                    spec.label,
                )?;
                require_build_source_match(&source.bytes, spec.embedded_bytes, spec.label)?;
                if spec.metal_shader {
                    metal_shader_sources.insert(spec.receipt_name.to_string(), source.attestation);
                } else {
                    rust_sources.insert(spec.receipt_name.to_string(), source.attestation);
                }
            }
        } else {
            let general_source_path = manifest_dir.join("src/qwen35/general.rs");
            let general_source_bytes = read_attested_bytes(&general_source_path, "general source")?;
            require_build_source_match(
                &general_source_bytes.bytes,
                GENERAL_SOURCE_BYTES,
                "general source",
            )?;
            rust_sources.insert("general".into(), general_source_bytes.attestation);
        }
        let binary_path = fs::canonicalize(std::env::current_exe()?)?;
        let binary = attest_file(&binary_path, "running gate binary", None)?;

        Ok(Self {
            model_dir: canonical_model_dir,
            cache_present,
            profile_id,
            source_lock_content_sha256,
            source_lock_value,
            profile: profile_bytes.attestation,
            source_lock: source_lock_bytes.attestation,
            binary,
            gate_source: gate_source_bytes.attestation,
            source_closure,
            metal_shader_receipt_key: if source_closure
                == "stack3-lm-head-v2-direct-compile-inputs-v1"
            {
                "compiled_metal_shader_sources"
            } else {
                "stack3_compiled_shader_sources"
            },
            rust_sources,
            metal_shader_sources,
            model_artifacts,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn source_lock_value(&self) -> &Value {
        &self.source_lock_value
    }

    pub fn receipt_json(&self) -> Value {
        let artifacts = self
            .model_artifacts
            .iter()
            .map(|(name, attestation)| (name.clone(), attestation_json(attestation)))
            .collect::<Map<_, _>>();
        let rust_sources = self
            .rust_sources
            .iter()
            .map(|(name, attestation)| (name.clone(), attestation_json(attestation)))
            .collect::<Map<_, _>>();
        let metal_shader_sources = self
            .metal_shader_sources
            .iter()
            .map(|(name, attestation)| (name.clone(), attestation_json(attestation)))
            .collect::<Map<_, _>>();
        let mut sources = json!({
            "closure": self.source_closure,
            "captured_at_start": true,
            "gate": attestation_json(&self.gate_source),
            "rust_and_bridge_sources": rust_sources,
        });
        sources
            .as_object_mut()
            .expect("sources is an object")
            .insert(
                self.metal_shader_receipt_key.to_string(),
                Value::Object(metal_shader_sources),
            );
        json!({
            "profile": {
                "profile_id": self.profile_id,
                "path": self.profile.path,
                "file_size": self.profile.size,
                "file_sha256": self.profile.sha256,
                "direct_regular_file": true,
                "single_link": self.profile.nlink == 1,
            },
            "model_dir": {
                "path": self.model_dir,
                "closure": "exact-profile-artifacts-plus-safe-cache-v1",
                "cache_present": self.cache_present,
                "artifacts": artifacts,
            },
            "source_lock": {
                "path": self.source_lock.path,
                "file_size": self.source_lock.size,
                "file_sha256": self.source_lock.sha256,
                "content_sha256": self.source_lock_content_sha256,
                "direct_regular_file": true,
                "single_link": self.source_lock.nlink == 1,
            },
            "binary": attestation_json(&self.binary),
            "sources": sources,
        })
    }

    pub fn verify_unchanged(&self) -> Result<(), Box<dyn Error>> {
        verify_file_unchanged(&self.profile.path, &self.profile, "deployment profile")?;
        verify_file_unchanged(&self.source_lock.path, &self.source_lock, "source lock")?;
        verify_file_unchanged(&self.binary.path, &self.binary, "running gate binary")?;
        verify_file_unchanged(&self.gate_source.path, &self.gate_source, "gate source")?;
        for (name, source) in &self.rust_sources {
            verify_file_unchanged(&source.path, source, &format!("custody source {name}"))?;
        }
        for (name, source) in &self.metal_shader_sources {
            verify_file_unchanged(
                &source.path,
                source,
                &format!("custody Metal shader source {name}"),
            )?;
        }
        let expected = self
            .model_artifacts
            .iter()
            .map(|(name, attestation)| {
                (name.clone(), (attestation.size, attestation.sha256.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let (model_dir, cache_present, artifacts) = attest_model_dir(&self.model_dir, &expected)?;
        if model_dir != self.model_dir
            || cache_present != self.cache_present
            || artifacts != self.model_artifacts
        {
            return Err(invalid("model directory changed during gate execution"));
        }
        Ok(())
    }

    pub fn verify_unchanged_receipt(&self) -> Result<Value, Box<dyn Error>> {
        self.verify_unchanged()?;
        let rust_sources = self
            .rust_sources
            .iter()
            .map(|(name, attestation)| (name.clone(), attestation_json(attestation)))
            .collect::<Map<_, _>>();
        let metal_shader_sources = self
            .metal_shader_sources
            .iter()
            .map(|(name, attestation)| (name.clone(), attestation_json(attestation)))
            .collect::<Map<_, _>>();
        let mut receipt = json!({
            "verified_at_end": true,
            "source_closure": self.source_closure,
            "binary": attestation_json(&self.binary),
            "gate": attestation_json(&self.gate_source),
            "rust_and_bridge_sources": rust_sources,
        });
        receipt
            .as_object_mut()
            .expect("receipt is an object")
            .insert(
                self.metal_shader_receipt_key.to_string(),
                Value::Object(metal_shader_sources),
            );
        Ok(receipt)
    }
}

pub fn read_attested_bytes(path: &Path, label: &str) -> Result<AttestedBytes, Box<dyn Error>> {
    let (bytes, attestation) = read_and_attest(path, label, None, true)?;
    Ok(AttestedBytes {
        bytes: bytes.expect("collect=true must retain bytes"),
        attestation,
    })
}

pub fn read_attested_json(
    path: &Path,
    label: &str,
) -> Result<(Value, FileAttestation), Box<dyn Error>> {
    let attested = read_attested_bytes(path, label)?;
    let value = serde_json::from_slice(&attested.bytes)?;
    Ok((value, attested.attestation))
}

pub fn attest_file(
    path: &Path,
    label: &str,
    expected_size: Option<u64>,
) -> Result<FileAttestation, Box<dyn Error>> {
    read_and_attest(path, label, expected_size, false).map(|(_, attestation)| attestation)
}

pub fn verify_file_unchanged(
    path: &Path,
    expected: &FileAttestation,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = attest_file(path, label, Some(expected.size))?;
    if &actual != expected {
        return Err(invalid(format!(
            "{label} changed during gate execution: expected SHA-256 {}, got {}",
            expected.sha256, actual.sha256
        )));
    }
    Ok(())
}

pub fn attestation_json(attestation: &FileAttestation) -> Value {
    json!({
        "path": attestation.path,
        "size": attestation.size,
        "sha256": attestation.sha256,
        "direct_regular_file": true,
        "single_link": attestation.nlink == 1,
    })
}

fn read_and_attest(
    path: &Path,
    label: &str,
    expected_size: Option<u64>,
    collect: bool,
) -> Result<(Option<Vec<u8>>, FileAttestation), Box<dyn Error>> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("cannot inspect {label}: {error}")))?;
    require_direct_single_link_file(&before, label)?;
    if expected_size.is_some_and(|expected| before.len() != expected) {
        return Err(invalid(format!(
            "{label} size mismatch: expected {}, got {}",
            expected_size.expect("checked Some"),
            before.len()
        )));
    }
    let mut file =
        File::open(path).map_err(|error| invalid(format!("cannot open {label}: {error}")))?;
    let opened = file
        .metadata()
        .map_err(|error| invalid(format!("cannot inspect opened {label}: {error}")))?;
    require_same_file(&before, &opened, label)?;

    let mut hasher = Sha256::new();
    let mut retained = collect.then(|| Vec::with_capacity(opened.len() as usize));
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        if let Some(bytes) = retained.as_mut() {
            bytes.extend_from_slice(&buffer[..count]);
        }
    }
    let after = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("cannot re-inspect {label}: {error}")))?;
    require_direct_single_link_file(&after, label)?;
    require_same_file(&opened, &after, label)?;
    let canonical = fs::canonicalize(path)
        .map_err(|error| invalid(format!("cannot canonicalize {label}: {error}")))?;
    Ok((
        retained,
        FileAttestation {
            path: canonical,
            size: opened.len(),
            sha256: format!("{:x}", hasher.finalize()),
            nlink: opened.nlink(),
        },
    ))
}

fn require_direct_single_link_file(
    metadata: &fs::Metadata,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!(
            "{label} must be a direct regular non-symlink file"
        )));
    }
    if metadata.nlink() != 1 {
        return Err(invalid(format!(
            "{label} must have exactly one hard link, got {}",
            metadata.nlink()
        )));
    }
    Ok(())
}

fn require_same_file(
    expected: &fs::Metadata,
    actual: &fs::Metadata,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    require_direct_single_link_file(actual, label)?;
    if expected.dev() != actual.dev()
        || expected.ino() != actual.ino()
        || expected.len() != actual.len()
    {
        return Err(invalid(format!("{label} changed while it was being read")));
    }
    Ok(())
}

fn require_build_source_match(
    disk: &[u8],
    embedded: &[u8],
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if disk != embedded {
        return Err(invalid(format!(
            "{label} differs from the bytes embedded in the running binary; rebuild required"
        )));
    }
    Ok(())
}

fn validate_profile(
    profile: &Value,
) -> Result<(String, String, BTreeMap<String, (u64, String)>), Box<dyn Error>> {
    if profile.get("format").and_then(Value::as_str) != Some(PROFILE_FORMAT)
        || profile.get("profile_id").and_then(Value::as_str) != Some(PROFILE_ID)
        || profile.pointer("/source/repo_id").and_then(Value::as_str) != Some(REPO_ID)
        || profile
            .pointer("/source/resolved_commit")
            .and_then(Value::as_str)
            != Some(LOCKED_REVISION)
    {
        return Err(invalid(
            "deployment profile identity is not the frozen Qwen3.5-0.8B profile",
        ));
    }
    let content_sha = profile
        .pointer("/source/source_lock_content_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_sha256(value))
        .ok_or_else(|| invalid("deployment profile source-lock content SHA-256 is invalid"))?
        .to_string();
    let artifacts = profile
        .get("artifacts")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("deployment profile artifacts must be an object"))?;
    let expected_names = BTreeSet::from([
        "chat_template.jinja",
        "config.json",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]);
    let observed_names = artifacts
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if observed_names != expected_names {
        return Err(invalid(
            "deployment profile must contain exactly the six frozen model artifacts",
        ));
    }
    let mut expected = BTreeMap::new();
    for (name, record) in artifacts {
        let size = record
            .get("size")
            .and_then(Value::as_u64)
            .filter(|size| *size > 0)
            .ok_or_else(|| invalid(format!("profile artifact {name} size is invalid")))?;
        let sha256 = record
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|value| is_sha256(value))
            .ok_or_else(|| invalid(format!("profile artifact {name} SHA-256 is invalid")))?;
        expected.insert(name.clone(), (size, sha256.to_string()));
    }
    Ok((PROFILE_ID.to_string(), content_sha, expected))
}

fn validate_source_lock(
    lock: &Value,
    expected_content_sha256: &str,
    artifacts: &BTreeMap<String, (u64, String)>,
) -> Result<(), Box<dyn Error>> {
    if lock.get("format").and_then(Value::as_str) != Some(SOURCE_LOCK_FORMAT)
        || lock.get("repo_id").and_then(Value::as_str) != Some(REPO_ID)
        || lock.get("resolved_commit").and_then(Value::as_str) != Some(LOCKED_REVISION)
        || lock.get("requested_revision").and_then(Value::as_str) != Some(LOCKED_REVISION)
    {
        return Err(invalid(
            "source lock identity is not the frozen Qwen3.5-0.8B source",
        ));
    }
    let declared = lock
        .get("content_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("source lock content_sha256 is missing"))?;
    let computed = canonical_json_sha256_without_content(lock)?;
    if declared != expected_content_sha256 || computed != declared {
        return Err(invalid(format!(
            "source lock canonical content SHA-256 mismatch: declared {declared}, computed {computed}"
        )));
    }
    let metadata = indexed_records(lock.pointer("/metadata/files"), "source lock metadata")?;
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ] {
        require_record_matches(&metadata, name, artifacts)?;
    }
    let weights = indexed_records(lock.pointer("/weights/files"), "source lock weights")?;
    let expected_weights = artifacts
        .keys()
        .filter(|name| name.ends_with(".safetensors"))
        .cloned()
        .collect::<BTreeSet<_>>();
    if weights.keys().cloned().collect::<BTreeSet<_>>() != expected_weights {
        return Err(invalid(
            "source lock weights do not exactly match the profile",
        ));
    }
    for name in expected_weights {
        require_record_matches(&weights, &name, artifacts)?;
    }
    Ok(())
}

fn indexed_records<'a>(
    value: Option<&'a Value>,
    label: &str,
) -> Result<BTreeMap<String, &'a Value>, Box<dyn Error>> {
    let records = value
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{label} must be an array")))?;
    let mut indexed = BTreeMap::new();
    for record in records {
        let name = record
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid(format!("{label} record path is invalid")))?;
        if indexed.insert(name.to_string(), record).is_some() {
            return Err(invalid(format!("{label} contains duplicate {name}")));
        }
    }
    Ok(indexed)
}

fn require_record_matches(
    records: &BTreeMap<String, &Value>,
    name: &str,
    artifacts: &BTreeMap<String, (u64, String)>,
) -> Result<(), Box<dyn Error>> {
    let record = records
        .get(name)
        .ok_or_else(|| invalid(format!("source lock does not bind {name}")))?;
    let (size, sha256) = artifacts
        .get(name)
        .ok_or_else(|| invalid(format!("profile does not bind {name}")))?;
    if record.get("size").and_then(Value::as_u64) != Some(*size)
        || record.get("sha256").and_then(Value::as_str) != Some(sha256)
    {
        return Err(invalid(format!("source lock/profile mismatch for {name}")));
    }
    Ok(())
}

fn attest_model_dir(
    model_dir: &Path,
    expected: &BTreeMap<String, (u64, String)>,
) -> Result<(PathBuf, bool, BTreeMap<String, FileAttestation>), Box<dyn Error>> {
    let root = fs::symlink_metadata(model_dir)
        .map_err(|error| invalid(format!("cannot inspect model directory: {error}")))?;
    if root.file_type().is_symlink() || !root.is_dir() {
        return Err(invalid(
            "model directory must be a direct non-symlink directory",
        ));
    }
    let canonical = fs::canonicalize(model_dir)?;
    let observed = fs::read_dir(&canonical)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF8 model entry"))
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let cache_present = observed.contains(".cache");
    let mut allowed = expected.keys().cloned().collect::<BTreeSet<_>>();
    if cache_present {
        allowed.insert(".cache".into());
    }
    if observed != allowed {
        return Err(invalid(format!(
            "model directory allowlist mismatch: expected {allowed:?}, got {observed:?}"
        )));
    }
    if cache_present {
        validate_cache_tree(&canonical.join(".cache"))?;
    }
    let mut artifacts = BTreeMap::new();
    for (name, (size, sha256)) in expected {
        let attestation = attest_file(
            &canonical.join(name),
            &format!("model artifact {name}"),
            Some(*size),
        )?;
        if &attestation.sha256 != sha256 {
            return Err(invalid(format!("model artifact {name} SHA-256 mismatch")));
        }
        artifacts.insert(name.clone(), attestation);
    }
    Ok((canonical, cache_present, artifacts))
}

fn validate_cache_tree(cache: &Path) -> Result<(), Box<dyn Error>> {
    let root = fs::symlink_metadata(cache)?;
    if root.file_type().is_symlink() || !root.is_dir() {
        return Err(invalid(
            "model .cache must be a direct non-symlink directory",
        ));
    }
    let mut pending = vec![cache.to_path_buf()];
    let mut count = 0usize;
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            count += 1;
            if count > MAX_CACHE_ENTRIES {
                return Err(invalid("model .cache exceeds the safety entry limit"));
            }
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                return Err(invalid("model .cache contains a symlink"));
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                if metadata.nlink() != 1
                    || metadata.permissions().mode() & 0o111 != 0
                    || cache_script_extension(&path)
                    || file_starts_with_shebang(&path)?
                {
                    return Err(invalid(
                        "model .cache contains an executable or linked file",
                    ));
                }
            } else {
                return Err(invalid("model .cache contains a non-regular entry"));
            }
        }
    }
    Ok(())
}

fn cache_script_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "py" | "pyc" | "pyo" | "sh" | "bash" | "zsh" | "fish" | "command" | "so" | "dylib"
            )
        })
}

fn file_starts_with_shebang(path: &Path) -> Result<bool, Box<dyn Error>> {
    let before = fs::symlink_metadata(path)?;
    require_direct_single_link_file(&before, "model cache file")?;
    let mut file = File::open(path)?;
    let opened = file.metadata()?;
    require_same_file(&before, &opened, "model cache file")?;
    let mut prefix = [0u8; 2];
    let count = file.read(&mut prefix)?;
    let after = fs::symlink_metadata(path)?;
    require_same_file(&opened, &after, "model cache file")?;
    Ok(count == 2 && prefix == *b"#!")
}

fn canonical_json_sha256_without_content(value: &Value) -> Result<String, Box<dyn Error>> {
    let mut body = value.clone();
    body.as_object_mut()
        .ok_or_else(|| invalid("source lock root must be an object"))?
        .remove("content_sha256");
    let mut bytes = Vec::new();
    write_python_canonical_json(&body, &mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn write_python_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(value) => output.extend_from_slice(python_number(value).as_bytes()),
        Value::String(value) => output.extend_from_slice(serde_json::to_string(value)?.as_bytes()),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_python_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                write_python_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn python_number(value: &serde_json::Number) -> String {
    let raw = value.to_string();
    let Some(index) = raw.find(['e', 'E']) else {
        return raw;
    };
    let mantissa = &raw[..index];
    let exponent = raw[index + 1..].parse::<i32>().unwrap_or(0);
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_file(label: &str, bytes: &[u8]) -> PathBuf {
        let unique = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "apxinf-qwen35-gate-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn attestation_detects_content_tamper_before_receipt_publication() {
        let path = temp_file("tamper", b"trusted");
        let trusted = read_attested_bytes(&path, "test evidence").unwrap();
        fs::write(&path, b"changed").unwrap();

        let error =
            verify_file_unchanged(&path, &trusted.attestation, "test evidence").unwrap_err();

        assert!(error.to_string().contains("changed"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_attestation_detects_same_size_drift() {
        let path = temp_file("binary-drift", b"binary-a");
        let trusted = attest_file(&path, "test binary", None).unwrap();
        fs::write(&path, b"binary-b").unwrap();

        let error = verify_file_unchanged(&path, &trusted, "test binary").unwrap_err();

        assert!(error.to_string().contains("changed"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn attested_input_rejects_symlinks_and_multiple_hard_links() {
        let target = temp_file("target", b"receipt");
        let symlink = target.with_extension("symlink");
        std::os::unix::fs::symlink(&target, &symlink).unwrap();
        assert!(read_attested_bytes(&symlink, "CPU receipt").is_err());
        fs::remove_file(&symlink).unwrap();

        let hardlink = target.with_extension("hardlink");
        fs::hard_link(&target, &hardlink).unwrap();
        assert!(read_attested_bytes(&target, "CPU receipt").is_err());
        fs::remove_file(hardlink).unwrap();
        fs::remove_file(target).unwrap();
    }

    #[test]
    fn canonical_source_lock_hash_uses_python_exponent_spelling() {
        let value = json!({
            "content_sha256": "ignored",
            "nested": {"z": 0.0, "a": true},
            "epsilon": 1.0e-6,
        });
        let mut body = value.clone();
        body.as_object_mut().unwrap().remove("content_sha256");
        let mut bytes = Vec::new();
        write_python_canonical_json(&body, &mut bytes).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"epsilon":1e-06,"nested":{"a":true,"z":0.0}}"#
        );
    }

    #[test]
    fn stack3_source_closure_binds_every_direct_compile_input_to_embedded_bytes() {
        let specs = stack3_source_specs();
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.receipt_name)
                .collect::<Vec<_>>(),
            vec![
                "gate_evidence",
                "general",
                "stack3_rust",
                "stack3_bridge",
                "apxinf_metal_build",
                "metal_w8_gdn",
                "metal_w8_mlp",
                "metal_w8_linear_layer",
                "metal_w8_gdn_out_g32",
            ]
        );
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for spec in specs {
            let source =
                read_attested_bytes(&manifest_dir.join(spec.manifest_relative_path), spec.label)
                    .unwrap();
            require_build_source_match(&source.bytes, spec.embedded_bytes, spec.label).unwrap();
            assert_eq!(source.attestation.nlink, 1);
        }
    }

    #[test]
    fn stack3_lm_head_v2_source_closure_binds_the_exact_composite_route() {
        let specs = stack3_lm_head_v2_source_specs();
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.receipt_name)
                .collect::<Vec<_>>(),
            vec![
                "gate_evidence",
                "general",
                "llm_trait",
                "apxinf_metal_lib",
                "apxinf_metal_build",
                "gdn_rust",
                "linear_layer_rust",
                "stack3_rust",
                "stack3_bridge",
                "metal_w8_mlp_bridge",
                "metal_w8_head_bridge",
                "metal_w8_gdn",
                "metal_w8_mlp",
                "metal_w8_linear_layer",
                "metal_w8_gdn_out_g32",
                "metal_w8_head",
                "metal_w8_matvec",
            ]
        );
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for spec in specs {
            let source =
                read_attested_bytes(&manifest_dir.join(spec.manifest_relative_path), spec.label)
                    .unwrap();
            require_build_source_match(&source.bytes, spec.embedded_bytes, spec.label).unwrap();
            assert_eq!(source.attestation.nlink, 1);
        }
    }

    #[test]
    fn stack3_lm_head_v2_capture_is_a_separate_public_custody_entrypoint() {
        let capture: fn(&Path, &Path, &str, &[u8]) -> Result<GateCustody, Box<dyn Error>> =
            GateCustody::capture_stack3_lm_head_v2;
        let _ = capture;
    }

    #[test]
    #[ignore = "requires the local onboarding source lock; never reads model weights"]
    fn local_source_lock_matches_the_checked_in_profile_and_canonical_hash() {
        let profile: Value = serde_json::from_slice(PROFILE_BYTES).unwrap();
        let (_, content_sha256, artifacts) = validate_profile(&profile).unwrap();
        let source_lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.apxinf/onboarding/qwen35-0.8b/source-lock.json");
        let source_lock = read_attested_bytes(&source_lock_path, "local source lock").unwrap();
        let source_lock: Value = serde_json::from_slice(&source_lock.bytes).unwrap();

        validate_source_lock(&source_lock, &content_sha256, &artifacts).unwrap();
    }
}
