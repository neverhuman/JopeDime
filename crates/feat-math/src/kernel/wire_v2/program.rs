use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReservedProgram {
    reservations: Vec<ReservationRef>,
}

impl ReservedProgram {
    pub fn new(reservations: Vec<ReservationRef>) -> VersionResult<Self> {
        check_reservations(&reservations)?;
        Ok(Self { reservations })
    }

    pub fn reservations(&self) -> &[ReservationRef] {
        &self.reservations
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        check_reservations(&self.reservations)?;
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(0)?; // Reservation-only variant; later executable variants get new tags.
        writer.byte(READER_VERSION)?;
        writer.word(RESERVATION_FEATURE)?;
        writer.word(self.reservations.len() as u64)?;
        for row in &self.reservations {
            writer.byte(row.domain.tag())?;
            writer.word(u64::from(row.id))?;
        }
        Ok(writer.finish())
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(MAGIC_V2.len())? != MAGIC_V2 {
            return Err(VersionError::UnsupportedVersion(
                "program envelope version".into(),
            ));
        }
        if reader.byte()? != 0 {
            return Err(VersionError::UnsupportedVersion(
                "program v2 variant".into(),
            ));
        }
        if reader.byte()? != READER_VERSION {
            return Err(VersionError::UnsupportedVersion(
                "minimum reader version".into(),
            ));
        }
        if reader.word()? != RESERVATION_FEATURE {
            return Err(VersionError::UnsupportedVersion(
                "required feature set".into(),
            ));
        }
        let count = reader.length()?;
        if count > WIRE.len() + FUNCTIONS.len() + VARIABLES.len() {
            return Err(VersionError::Malformed("reservation count".into()));
        }
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let domain = KernelWireDomain::from_tag(reader.byte()?)?;
            let id = reader.word()?;
            let id = u16::try_from(id)
                .map_err(|_| VersionError::UnsupportedVersion(format!("{domain:?} id {id}")))?;
            rows.push(ReservationRef::new(domain, id)?);
        }
        reader.finish()?;
        Self::new(rows)
    }
}

#[derive(Clone, Debug)]
// B14: explicit wire-60 ordered program routes.
pub enum VersionedProgram {
    LegacyV1(Program),
    ReservedV2(ReservedProgram),
    ScalarV2(ScalarExpressionProgram),
    LinearScaledV2(LinearScaledWinner),
    // B12: FeatureWithinRow execution is a separate v2 subtype.
    RowVectorV2(super::row_vector::Program),
    DifferenceV2(DifferenceProgram),
    IntegrationV2(super::cumulative_integration::IntegrationProgram),
    TimeGradientV2(super::timescale_nabla::TimeGradientProgram),
    PathStatsV2(super::realized_path_stats::PathStatsProgram),
}

impl VersionedProgram {
    /// Explicit numeric-input scalar fitting, with no v1 coercion.
    pub fn fit_scalar(
        &self,
        training: &crate::Rows,
        labels: &[f64],
        carrier: ScalarCarrier,
        lambda: f64,
    ) -> VersionResult<VersionedModel> {
        match self {
            Self::ScalarV2(program) => Ok(VersionedModel::ScalarV2(Box::new(
                program.fit_training(training, labels, carrier, lambda)?,
            ))),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no scalar fit route".into(),
            )),
        }
    }

    /// Ordered recipes carry their explicit input identity and continuation API.
    pub fn emit_ordered_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<DifferenceRecipe> {
        match self {
            Self::DifferenceV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no ordered recipe route".into(),
            )),
        }
    }

    /// B14 integration artifacts carry their own typed codec and continuation API.
    pub fn emit_integral_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::cumulative_integration::IntegrationRecipe> {
        match self {
            Self::IntegrationV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no integral recipe route".into(),
            )),
        }
    }

    pub fn emit_time_gradient_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::timescale_nabla::TimeGradientRecipe> {
        match self {
            Self::TimeGradientV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no time_gradient recipe route".into(),
            )),
        }
    }

    pub fn emit_path_stats_recipe(
        &self,
        language: DifferenceLanguage,
    ) -> VersionResult<super::realized_path_stats::PathStatsRecipe> {
        match self {
            Self::PathStatsV2(program) => program.emit_recipe(language),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no path_stats recipe route".into(),
            )),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        if bytes.len() < MAGIC_V2.len() || &bytes[..MAGIC_PREFIX.len()] != MAGIC_PREFIX {
            return Err(VersionError::Malformed("program magic".into()));
        }
        match bytes[MAGIC_PREFIX.len()] {
            1 => Ok(Self::LegacyV1(Program::from_bytes(bytes)?)),
            2 => match bytes.get(MAGIC_V2.len()) {
                Some(0) => Ok(Self::ReservedV2(ReservedProgram::from_bytes(bytes)?)),
                Some(1) => Ok(Self::ScalarV2(ScalarExpressionProgram::from_bytes(bytes)?)),
                Some(2) => Ok(Self::LinearScaledV2(LinearScaledWinner::from_bytes(bytes)?)),
                Some(3) => Ok(Self::RowVectorV2(super::row_vector::Program::from_bytes(
                    bytes,
                )?)),
                Some(4) => Ok(Self::DifferenceV2(DifferenceProgram::from_bytes(bytes)?)),
                Some(5) => Ok(Self::IntegrationV2(
                    super::cumulative_integration::IntegrationProgram::from_bytes(bytes)?,
                )),
                Some(6) => Ok(Self::TimeGradientV2(
                    super::timescale_nabla::TimeGradientProgram::from_bytes(bytes)?,
                )),
                Some(7) => Ok(Self::PathStatsV2(
                    super::realized_path_stats::PathStatsProgram::from_bytes(bytes)?,
                )),
                Some(_) => Err(VersionError::UnsupportedVersion(
                    "program v2 variant".into(),
                )),
                None => Err(VersionError::Malformed("truncated program variant".into())),
            },
            version => Err(VersionError::UnsupportedVersion(format!(
                "program envelope {version}"
            ))),
        }
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        match self {
            Self::LegacyV1(program) => Ok(program.to_bytes()?),
            Self::ReservedV2(program) => program.to_bytes(),
            Self::ScalarV2(program) => program.to_bytes(),
            Self::LinearScaledV2(program) => program.to_bytes(),
            Self::RowVectorV2(program) => program.to_bytes(),
            Self::DifferenceV2(program) => program.to_bytes(),
            Self::IntegrationV2(program) => program.to_bytes(),
            Self::TimeGradientV2(program) => program.to_bytes(),
            Self::PathStatsV2(program) => program.to_bytes(),
        }
    }

    /// Explicit ordered-input route; legacy and reservation bodies cannot fall through.
    pub fn fit_ordered(
        &self,
        rows: &[OrderedRow],
        labels: &[f64],
        task: TaskMode,
        fold: FoldBoundary,
    ) -> VersionResult<VersionedModel> {
        match self {
            Self::DifferenceV2(program) => Ok(VersionedModel::DifferenceV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::IntegrationV2(program) => Ok(VersionedModel::IntegrationV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::TimeGradientV2(program) => Ok(VersionedModel::TimeGradientV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            Self::PathStatsV2(program) => Ok(VersionedModel::PathStatsV2(Box::new(
                program.fit_training(rows, labels, task, fold)?,
            ))),
            _ => Err(VersionError::UnsupportedVersion(
                "program has no ordered fit route".into(),
            )),
        }
    }

    pub fn legacy_genome(&self) -> VersionResult<&GenomeIr> {
        match self {
            Self::LegacyV1(Program::LegacyV1(genome)) => Ok(genome),
            Self::ReservedV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 reservations are not executable".into(),
            )),
            Self::RowVectorV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 row-vector program cannot decode as a v1 genome".into(),
            )),
            Self::DifferenceV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for difference execution".into(),
            )),
            Self::IntegrationV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for integral execution".into(),
            )),
            Self::TimeGradientV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for time_gradient execution".into(),
            )),
            Self::PathStatsV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for path_stats execution".into(),
            )),
            Self::ScalarV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 scalar execution is not admitted".into(),
            )),
            Self::LinearScaledV2(_) => Err(VersionError::UnsupportedVersion(
                "linear-scaled winner requires its explicit execution profile".into(),
            )),
        }
    }

    pub fn to_lossless_text(&self) -> VersionResult<String> {
        match self {
            Self::LegacyV1(program) => Ok(program.to_lossless_text()?),
            Self::ReservedV2(_)
            | Self::ScalarV2(_)
            | Self::RowVectorV2(_)
            | Self::DifferenceV2(_)
            | Self::IntegrationV2(_)
            | Self::LinearScaledV2(_)
            | Self::TimeGradientV2(_)
            | Self::PathStatsV2(_) => Ok(format!(
                "(program codec=2 hex=\"{}\")",
                self.to_bytes()?
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )),
        }
    }

    pub fn parse_lossless_text(text: &str) -> VersionResult<Self> {
        if text.starts_with("(program codec=1 ") {
            return Ok(Self::LegacyV1(Program::parse_lossless_text(text)?));
        }
        let Some(hex) = text
            .strip_prefix("(program codec=2 hex=\"")
            .and_then(|value| value.strip_suffix("\")"))
        else {
            return if text.starts_with("(program codec=2 ") || text == "(program codec=2)" {
                Err(VersionError::Malformed("program text envelope".into()))
            } else if text.starts_with("(program codec=") {
                Err(VersionError::UnsupportedVersion(
                    "program text codec".into(),
                ))
            } else {
                Err(VersionError::Malformed("program text envelope".into()))
            };
        };
        if hex.len() > 2 * MAX_BYTES || hex.len() % 2 != 0 {
            return Err(VersionError::Malformed("program text length".into()));
        }
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                (pair[0] as char)
                    .to_digit(16)
                    .zip((pair[1] as char).to_digit(16))
                    .map(|(high, low)| ((high << 4) | low) as u8)
                    .ok_or_else(|| VersionError::Malformed("hex digit".into()))
            })
            .collect::<VersionResult<Vec<_>>>()?;
        match Self::from_bytes(&bytes)? {
            value @ (Self::ReservedV2(_)
            | Self::ScalarV2(_)
            | Self::RowVectorV2(_)
            | Self::DifferenceV2(_)
            | Self::IntegrationV2(_)
            | Self::LinearScaledV2(_)
            | Self::TimeGradientV2(_)
            | Self::PathStatsV2(_)) => Ok(value),
            Self::LegacyV1(_) => Err(VersionError::UnsupportedVersion(
                "program text codec/body mismatch".into(),
            )),
        }
    }
}

/// Existing export routes remain available only for an executable v1 genome.
pub fn export_report(program: &VersionedProgram) -> ExportReport {
    match program {
        VersionedProgram::LegacyV1(Program::LegacyV1(genome)) => export::export_report(genome),
        VersionedProgram::ReservedV2(_) => {
            let refusal = Route::Unsupported(vec!["version 2 reservation-only".into()]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
        VersionedProgram::RowVectorV2(program) => {
            let support = match program.plan() {
                Ok(_) => Route::Ok,
                Err(e) => Route::Unsupported(vec![e.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "v2 row-vector JavaScript route unavailable".into()
                ]),
            }
        }
        VersionedProgram::DifferenceV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "difference ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::IntegrationV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "integral ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::TimeGradientV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "time_gradient ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::PathStatsV2(program) => {
            let support = match program.validate_recipe_closure() {
                Ok(()) => Route::Ok,
                Err(error) => Route::Unsupported(vec![error.to_string()]),
            };
            ExportReport {
                rust_recipe: support.clone(),
                python_recipe: support.clone(),
                cpp17_recipe: support.clone(),
                cpp17_fitted: support,
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: Route::Unsupported(vec![
                    "path_stats ordered JavaScript route unavailable".into(),
                ]),
            }
        }
        VersionedProgram::ScalarV2(_) => {
            let refusal = Route::Unsupported(vec![
                "scalar export requires carrier, width and lambda; use scalar_export_report".into(),
            ]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
        VersionedProgram::LinearScaledV2(_) => {
            let refusal = Route::Unsupported(vec![
                "explicit linear-scaling library export required".into(),
            ]);
            ExportReport {
                rust_recipe: refusal.clone(),
                python_recipe: refusal.clone(),
                cpp17_recipe: refusal.clone(),
                cpp17_fitted: refusal.clone(),
                cpp17_fitted_basis: FittedBasis::Declared,
                javascript: refusal,
            }
        }
    }
}
