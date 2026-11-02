use super::*;

/// Counts the expression cells of one stage as they are generated.
pub(crate) struct Tally {
    counts: StageFaultCounts,
    hard: FaultSet,
    first: Option<u64>,
    varied: bool,
}

impl Tally {
    pub(crate) fn new() -> Self {
        Self {
            counts: StageFaultCounts::default(),
            hard: FaultSeverity::Hard.classes(),
            first: None,
            varied: false,
        }
    }

    /// One generated cell, with the faults of its evaluation. Returns whether they include a Hard
    /// class.
    pub(crate) fn cell(&mut self, value: f64, faults: FaultSet) -> bool {
        self.counts.evaluated_cells += 1;
        let mut bits = faults.bits();
        while bits != 0 {
            self.counts.faulted_cells[bits.trailing_zeros() as usize] += 1;
            bits &= bits - 1;
        }
        if faults.intersects(FaultSeverity::Domain.classes()) {
            self.counts.domain_cells += 1;
        }
        if faults.intersects(FaultSeverity::Saturation.classes()) {
            self.counts.saturation_cells += 1;
        }
        match self.first {
            None => self.first = Some(value.to_bits()),
            Some(first) => self.varied |= value.to_bits() != first,
        }
        faults.intersects(self.hard)
    }

    /// The end of one generated row; `hard` when one of its cells had a Hard class.
    pub(crate) fn row(&mut self, hard: bool) {
        if hard {
            self.counts.hard_rows += 1;
        }
    }

    pub(crate) fn finish(self) -> StageFaultCounts {
        let mut counts = self.counts;
        counts.constant_output = self.first.is_some() && !self.varied;
        counts
    }
}

/// `apply_op_generated`'s `MapExpr` arm with the traced evaluators: out[i][j] = expr(x = r[j],
/// row = r). The compiled `map_row_traced` returns `map_row`'s values, and the interpreter's
/// `eval_element_traced` returns `eval_element(..).unwrap_or(0.0)`.
pub(crate) fn generate_map(expr: &MathExpr, x: &Rows, tally: &mut Tally) -> Rows {
    match expr.compile() {
        Ok(compiled) if compiled.uses_row_summaries() => x
            .iter()
            .map(|r| {
                let (values, faults) = compiled.map_row_traced(r);
                let mut hard = false;
                for (value, faults) in values.iter().zip(faults) {
                    hard |= tally.cell(*value, faults);
                }
                tally.row(hard);
                values
            })
            .collect(),
        Ok(_) => x
            .iter()
            .map(|r| {
                let mut hard = false;
                let mut values = Vec::with_capacity(r.len());
                for v in r {
                    let (value, faults) = expr.eval_element_traced(*v, r);
                    hard |= tally.cell(value, faults);
                    values.push(value);
                }
                tally.row(hard);
                values
            })
            .collect(),
        Err(_) => {
            tally.counts.compile_failed = true;
            x.iter()
                .map(|r| {
                    let values = vec![0.0; r.len()];
                    let mut hard = false;
                    for value in &values {
                        hard |= tally.cell(*value, FaultSet::COMPILE_FAILED);
                    }
                    tally.row(hard);
                    values
                })
                .collect()
        }
    }
}

/// `apply_op_generated`'s `DeriveExpr` arm with the traced evaluators: out[i] = [expr(row = r)].
/// Both `eval_row_traced` forms return `eval_row(..).unwrap_or(0.0)`.
pub(crate) fn generate_derive(expr: &MathExpr, x: &Rows, tally: &mut Tally) -> Rows {
    let mut derived = |(value, faults): (f64, FaultSet)| {
        let hard = tally.cell(value, faults);
        tally.row(hard);
        vec![value]
    };
    match expr.compile() {
        Ok(compiled) if compiled.uses_row_summaries() => x
            .iter()
            .map(|r| derived(compiled.eval_row_traced(r)))
            .collect(),
        Ok(_) => x.iter().map(|r| derived(expr.eval_row_traced(r))).collect(),
        Err(_) => {
            let rows: Rows = x
                .iter()
                .map(|_| derived((0.0, FaultSet::COMPILE_FAILED)))
                .collect();
            tally.counts.compile_failed = true;
            rows
        }
    }
}
