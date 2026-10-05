//! Typed scalar consumers reuse the existing private arm/site/edit implementation.
use super::*;
impl MathExpr {
    /// Reuse the existing revision-two arm/site draw and return its remaining
    /// RNG state. The scalar projection preserves every actual node/leaf.
    pub(crate) fn scalar_mutation_site_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        seed: u64,
    ) -> Result<(ExpressionArm, Option<Vec<usize>>, Pcg32)> {
        let parent = Search::new(profile, scope)?.finish(self.root.clone())?;
        let mut rng = Pcg32::new(seed, 0x6d7574617632);
        let arm = draw_arm(profile, &mut rng)?;
        let sites = if arm == ExpressionArm::ConstantJitter {
            literal_paths(&parent.root)
        } else {
            paths(&parent.root)
        };
        let path = if sites.is_empty() {
            None
        } else {
            Some(sites[rng.next_u32() as usize % sites.len()].clone())
        };
        Ok((arm, path, rng))
    }

    /// Apply one selected arm at a proxy root with the caller's remaining
    /// complete-tree budget and captured old RNG state. A subtree need not
    /// contain a variable: ConstantJitter edits a literal inside a whole tree.
    /// Complete profile/size/variable admission follows the typed graft.
    pub(crate) fn edit_scalar_proxy_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        arm: ExpressionArm,
        rng: &mut Pcg32,
    ) -> Result<Option<Expr>> {
        let search = Search::new(profile, scope)?;
        scope.validate(self)?;
        Ok(search.edit(
            &self.root,
            arm,
            search.scope.max_nodes,
            search.scope.max_depth,
            rng,
        ))
    }
}
