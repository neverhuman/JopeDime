use super::*;

#[cfg(test)]
mod executable_target_tests {
    use super::validate_executable_target;
    use std::path::Path;

    const LINUX_X86_64: &str = "x86_64-unknown-linux-gnu";
    const MACOS_ARM64: &str = "aarch64-apple-darwin";

    #[test]
    fn full_proposal_requires_the_actual_embedded_engine_and_source() {
        let source = "1".repeat(40);
        let good = serde_json::json!({"source_sha40":source,
            "features":{"engine":true,"embedded":true},"trusts_development_key":false});
        let methods =
            serde_json::json!({"methods":[{"id":jope_service::proposal::METHOD,"compiled":true}]});
        super::validate_proposal_build(&good, &methods, &source).unwrap();
        for (path, value) in [
            ("/features/engine", serde_json::json!(false)),
            ("/features/embedded", serde_json::json!(false)),
            ("/trusts_development_key", serde_json::json!(true)),
            ("/source_sha40", serde_json::json!("2".repeat(40))),
        ] {
            let mut bad = good.clone();
            *bad.pointer_mut(path).unwrap() = value;
            assert!(
                super::validate_proposal_build(&bad, &methods, &source).is_err(),
                "{path}"
            );
        }
        for methods in [
            serde_json::json!({}),
            serde_json::json!({"methods":[]}),
            serde_json::json!({"methods":[{"id":jope_service::proposal::METHOD,"compiled":false}]}),
            serde_json::json!({"methods":[{"id":"proposal.step60000.regression","compiled":true}]}),
        ] {
            assert!(super::validate_proposal_build(&good, &methods, &source).is_err());
        }
        assert!(super::validate_proposal_build(&serde_json::json!({}), &methods, &source).is_err());
    }

    fn x86_64_elf() -> Vec<u8> {
        let mut code = vec![0; 64];
        code[0..4].copy_from_slice(b"\x7fELF");
        code[4] = 2; // ELFCLASS64
        code[5] = 1; // ELFDATA2LSB
        code[6] = 1; // EV_CURRENT
        code[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        code[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        code[20..24].copy_from_slice(&1u32.to_le_bytes()); // EV_CURRENT
        code[52..54].copy_from_slice(&64u16.to_le_bytes()); // ELF64 header size
        code
    }

    fn arm64_mach_o() -> Vec<u8> {
        let mut code = vec![0; 32];
        code[0..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]); // MH_MAGIC_64, little-endian
        code[4..8].copy_from_slice(&0x0100000cu32.to_le_bytes()); // CPU_TYPE_ARM64
        code[12..16].copy_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        code
    }

    #[test]
    fn linux_target_accepts_x86_64_elf_and_rejects_other_formats_or_architectures() {
        assert!(validate_executable_target(&x86_64_elf(), LINUX_X86_64, Path::new("app")).is_ok());

        let mut arm64_elf = x86_64_elf();
        arm64_elf[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        let mut elf32 = x86_64_elf();
        elf32[4] = 1; // ELFCLASS32
        elf32[18..20].copy_from_slice(&3u16.to_le_bytes()); // EM_386
        let mut big_endian = x86_64_elf();
        big_endian[5] = 2; // ELFDATA2MSB
        let mut bad_header_version = x86_64_elf();
        bad_header_version[20..24].copy_from_slice(&0u32.to_le_bytes());
        let mut bad_header_size = x86_64_elf();
        bad_header_size[52..54].copy_from_slice(&0u16.to_le_bytes());

        for bytes in [
            arm64_elf,
            elf32,
            big_endian,
            bad_header_version,
            bad_header_size,
            b"\x7fELF truncated".to_vec(),
            arm64_mach_o(),
            b"MZ windows PE".to_vec(),
        ] {
            let error = validate_executable_target(&bytes, LINUX_X86_64, Path::new("app"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("not a 64-bit little-endian x86_64 ELF file"),
                "{error}"
            );
        }
    }

    #[test]
    fn macos_arm64_target_accepts_only_arm64_64_bit_mach_o_executables() {
        let mach_o = arm64_mach_o();
        assert!(validate_executable_target(&mach_o, MACOS_ARM64, Path::new("app")).is_ok());
        assert!(
            validate_executable_target(&x86_64_elf(), MACOS_ARM64, Path::new("app"))
                .unwrap_err()
                .to_string()
                .contains("not a 64-bit arm64 Mach-O executable")
        );

        let mut wrong_cpu = arm64_mach_o();
        wrong_cpu[4..8].copy_from_slice(&0x01000007u32.to_le_bytes()); // CPU_TYPE_X86_64
        assert!(validate_executable_target(&wrong_cpu, MACOS_ARM64, Path::new("app")).is_err());
        let mut dylib = arm64_mach_o();
        dylib[12..16].copy_from_slice(&6u32.to_le_bytes()); // MH_DYLIB, not MH_EXECUTE
        assert!(validate_executable_target(&dylib, MACOS_ARM64, Path::new("app")).is_err());
    }

    #[test]
    fn unsupported_targets_are_rejected_even_when_the_binary_is_elf() {
        let error =
            validate_executable_target(&x86_64_elf(), "x86_64-pc-windows-gnu", Path::new("app"))
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("unsupported appliance executable target"),
            "{error}"
        );
    }
}
