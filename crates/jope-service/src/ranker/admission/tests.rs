use super::*;
use jope_api::Severity;
use rand_chacha::rand_core::SeedableRng;

struct AdmissionSlateFixture {
    keys: Vec<String>,
    features: Vec<[f32; adapter::PROG_FEATURES]>,
    projections: Vec<[f32; adapter::PROJ_FEATURES]>,
    conditioning: TokenProjection,
}

impl AdmissionSlateFixture {
    fn new() -> Self {
        Self {
            keys: (0..22).map(|i| format!("program-{i}")).collect(),
            features: (0..22)
                .map(|i| [i as f32; adapter::PROG_FEATURES])
                .collect(),
            projections: (0..22)
                .map(|i| [100.0 + i as f32; adapter::PROJ_FEATURES])
                .collect(),
            conditioning: TokenProjection {
                tokens: std::array::from_fn(|i| {
                    std::array::from_fn(|j| (i * 32 + j) as f32 / 1000.0)
                }),
                mask: [1., 1., 1., 0., 1., 1., 0., 1., 0., 1.],
            },
        }
    }

    fn measured(&self) -> Vec<MeasuredAlgorithm<'_>> {
        self.keys[..6]
            .iter()
            .enumerate()
            .map(|(i, key)| MeasuredAlgorithm {
                id: key,
                surrogate_fitness: 5.0 - i as f64,
                gate: i % 2 == 0,
            })
            .collect()
    }

    fn programs(&self) -> Vec<AdmissionProgram<'_>> {
        (0..22)
            .map(|i| AdmissionProgram {
                key: &self.keys[i],
                features412: &self.features[i],
                projection16: &self.projections[i],
            })
            .collect()
    }

    fn prepare(
        &self,
        rule: RevealedRule,
        explore: bool,
    ) -> Result<ServingAdmissionSlate<'_>, AdmissionError> {
        let programs = self.programs();
        ServingAdmissionSlate::prepare(
            &self.measured(),
            &programs[..6],
            &programs[6..if explore { 22 } else { 14 }],
            &self.conditioning,
            rule,
            explore,
        )
    }
}

mod cpu_adapter;
mod guidance;
mod selection;
mod slate;
