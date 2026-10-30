use super::*;

impl MathExpr {
    /// Ramped half-and-half using the profile's equally weighted element pool.
    /// The independent stream leaves all legacy draws unchanged.
    pub fn grow_v2(profile: &SearchProfile, scope: ExpressionScope, seed: u64) -> Result<Self> {
        let search = Search::new(profile, scope)?;
        let mut rng = Pcg32::new(seed, 0x67726f777632);
        let depth = 1 + rng.next_u32() as usize % search.scope.max_depth;
        let full = rng.next_u32() & 1 == 0;
        search.finish(search.grow(search.scope.max_nodes, depth, full, true, &mut rng))
    }
}

impl Search<'_> {
    fn minimum_full(&self, depth: usize) -> usize {
        let Some(arity) = self.operators.iter().map(|o| o.arity()).min() else {
            return 1;
        };
        (1..depth).fold(1, |n, _| 1 + arity * n)
    }

    pub(super) fn leaf(
        &self,
        budget: usize,
        depth: usize,
        variable: bool,
        rng: &mut Pcg32,
    ) -> Expr {
        let constants: Vec<_> = self
            .pool
            .birth_constants
            .iter()
            .copied()
            .filter(|v| !v.is_sign_negative() || (budget >= 2 && depth >= 2))
            .collect();
        if variable {
            return self.variable(rng);
        }
        let draw = rng.next_u32() as usize % (self.variables.len() + constants.len());
        if draw < self.variables.len() {
            Expr::Var(self.variables[draw])
        } else {
            literal(constants[draw - self.variables.len()])
        }
    }

    pub(super) fn grow(
        &self,
        budget: usize,
        mut depth: usize,
        full: bool,
        variable: bool,
        rng: &mut Pcg32,
    ) -> Expr {
        while full && depth > 1 && self.minimum_full(depth) > budget {
            depth -= 1;
        }
        let child_min = if full {
            self.minimum_full(depth.saturating_sub(1))
        } else {
            1
        };
        let choices: Vec<_> = self
            .operators
            .iter()
            .copied()
            .filter(|op| depth > 1 && op.arity() * child_min < budget)
            .collect();
        if choices.is_empty()
            || (!full
                && rng.next_u32() as usize
                    % (choices.len() + self.variables.len() + self.pool.birth_constants.len())
                    >= choices.len())
        {
            return self.leaf(budget, depth, variable, rng);
        }
        let op = choices[rng.next_u32() as usize % choices.len()];
        let force = rng.next_u32() as usize % op.arity();
        let mut remaining = budget - 1;
        let children = (0..op.arity())
            .map(|i| {
                let maximum = remaining - (op.arity() - i - 1) * child_min;
                let child_budget = child_min + rng.next_u32() as usize % (maximum - child_min + 1);
                let child = self.grow(child_budget, depth - 1, full, variable && i == force, rng);
                remaining -= nodes(&child);
                child
            })
            .collect();
        op.build(children)
    }
}
