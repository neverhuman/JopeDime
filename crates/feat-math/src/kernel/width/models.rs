use super::*;

/// The input width that `head` accepts, as far as its fitted state reveals it.
///
/// Heads read columns defensively (`take`, `min` and bounds-checked routing),
/// so none panics on width; this is the contract, not a guard. Heads that keep
/// statistics of every training column reveal the exact width they were fitted
/// on. The tree heads (`n_feat` is capped by `max_features`) and `ridge_gcv`
/// (which may screen columns) reveal only a lower bound, and a constant
/// fallback that keeps no column reveals nothing. A composite head requires
/// what its members require, and its members, fitted on the same rows, agree.
pub fn required_head_width(head: &FittedHead) -> WidthReq {
    match head {
        FittedHead::Ridge(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::RidgeGcv(m) => past_max(m.cols.iter().copied()),
        FittedHead::ElasticNet(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Gbm(m) => at_least(m.n_feat),
        FittedHead::Ensemble { ridge, gbm, .. } => {
            meet(WidthReq::Exactly(ridge.means.len()), at_least(gbm.n_feat))
        }
        FittedHead::HistGbm2(m) => at_least(m.n_feat),
        FittedHead::Krr(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Gpr(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::GprKernel(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Ksmooth(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Hurdle2(m) => m
            .components
            .first()
            .map_or(WidthReq::Any, |c| WidthReq::Exactly(c.gate.means.len())),
        FittedHead::Stack(m) => stack_width(m),
        FittedHead::Portfolio(m) => portfolio_width(m),
    }
}

/// Checks that `model` accepts input rows `width` columns wide: walks the
/// fitted stages in order with the width each receives, checking its state,
/// its required input width and its output width, then checks the head.
/// `Model::Symbolic` accepts any width: where its expression reads a column
/// beyond the row, it evaluates to 0.0. A `Model::Portfolio` pads or truncates
/// rows to its fitted width, which it requires as a head does; its members are
/// not checked.
pub fn check_input_width(model: &Model, width: usize) -> std::result::Result<(), WidthMismatch> {
    match model {
        Model::Pipeline { ops, head, .. } => {
            let mut width = width;
            for (stage, op) in ops.iter().enumerate() {
                let expected = required_input_width(op);
                let mismatch = move |kind| WidthMismatch {
                    stage,
                    expected,
                    got: width,
                    kind,
                };
                if let Err(reason) = check_fitted_state(op) {
                    return Err(mismatch(MismatchKind::InconsistentState { reason }));
                }
                if !expected.admits(width) {
                    return Err(mismatch(MismatchKind::StageInput));
                }
                width = finished_width(op, width)
                    .map_err(|_| mismatch(MismatchKind::OutputOverflow))?;
            }
            head_admits(required_head_width(head), ops.len(), width)
        }
        Model::Symbolic { .. } => Ok(()),
        Model::Portfolio(m) => head_admits(portfolio_width(m), 0, width),
    }
}

impl Model {
    /// [`Model::predict`] behind the fitted-width contract.
    ///
    /// Returns a [`WidthMismatch`], before any stage runs, when the rows of `x`
    /// differ in width or [`check_input_width`] refuses their width. That covers
    /// every width at which a fitted stage would panic in `predict`, and every
    /// width at which a stage or head holding one fitted entry per input column
    /// would silently misread the rows; a head's own tables are trusted (see the
    /// module notes). Otherwise it returns `predict(x)` itself, so its bits are
    /// `predict`'s. With no rows there is no width to check.
    ///
    /// `Exactly` requirements make this stricter than `predict`, which accepts
    /// some narrower nonempty inputs. Both paths refuse zero-width rows.
    pub fn try_predict(&self, x: &Rows) -> std::result::Result<Vec<f64>, WidthMismatch> {
        if let Some(first) = x.first() {
            let width = first.len();
            if let Some((row, ragged)) = x.iter().enumerate().find(|(_, r)| r.len() != width) {
                return Err(WidthMismatch {
                    stage: 0,
                    expected: WidthReq::Exactly(width),
                    got: ragged.len(),
                    kind: MismatchKind::RaggedInput { row },
                });
            }
            check_input_width(self, width)?;
        }
        self.predict(x)
    }
}

fn head_admits(
    expected: WidthReq,
    stage: usize,
    width: usize,
) -> std::result::Result<(), WidthMismatch> {
    if expected.admits(width) {
        Ok(())
    } else {
        Err(WidthMismatch {
            stage,
            expected,
            got: width,
            kind: MismatchKind::HeadInput,
        })
    }
}

fn stack_width(m: &StackModel) -> WidthReq {
    [
        m.ridge_a.as_ref().map(|r| WidthReq::Exactly(r.means.len())),
        m.ridge_b.as_ref().map(|r| WidthReq::Exactly(r.means.len())),
        m.net.as_ref().map(|n| WidthReq::Exactly(n.means.len())),
        m.trees.as_ref().map(|t| at_least(t.n_feat)),
        m.kernel.as_ref().map(|k| WidthReq::Exactly(k.means.len())),
    ]
    .into_iter()
    .flatten()
    .fold(WidthReq::Any, meet)
}

/// A mean-only portfolio ignores its input; any other pads or truncates rows to
/// the width it was fitted on.
fn portfolio_width(m: &PortfolioModel) -> WidthReq {
    if m.mean_only {
        WidthReq::Any
    } else {
        WidthReq::Exactly(m.width)
    }
}

/// What two parts fitted on the same rows require together. They agree, so an
/// exact width stands for both.
fn meet(a: WidthReq, b: WidthReq) -> WidthReq {
    match (a, b) {
        (WidthReq::Exactly(n), _) | (_, WidthReq::Exactly(n)) => WidthReq::Exactly(n),
        (WidthReq::AtLeast(x), WidthReq::AtLeast(y)) => WidthReq::AtLeast(x.max(y)),
        (WidthReq::AtLeast(n), WidthReq::Any) | (WidthReq::Any, WidthReq::AtLeast(n)) => {
            WidthReq::AtLeast(n)
        }
        (WidthReq::Any, WidthReq::Any) => WidthReq::Any,
    }
}

fn at_least(n: usize) -> WidthReq {
    if n == 0 {
        WidthReq::Any
    } else {
        WidthReq::AtLeast(n)
    }
}
