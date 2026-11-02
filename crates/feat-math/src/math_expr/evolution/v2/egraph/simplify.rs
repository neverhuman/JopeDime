/// The result of [`simplify`].
#[derive(Clone, Debug)]
struct Simplified<'a> {
    /// The smaller tree extraction found, or else the input itself, borrowed: a refused input is
    /// never copied.
    expression: Cow<'a, MathExpr>,
    changed: bool,
    stop: Saturation,
    iterations: usize,
    enodes: usize,
    applications: usize,
    /// The most e-nodes the hashcons held at any point, the input's own included.
    peak: usize,
}

impl<'a> Simplified<'a> {
    /// The refusal of an input that fails the structure check, before saturation starts. It
    /// borrows the input, so nothing copies or walks it.
    fn refused(expression: &'a MathExpr) -> Self {
        Self {
            expression: Cow::Borrowed(expression),
            changed: false,
            stop: Saturation::Malformed,
            iterations: 0,
            enodes: 0,
            applications: 0,
            peak: 0,
        }
    }
}

/// A result compares with a tree as the tree it holds.
impl PartialEq<MathExpr> for Cow<'_, MathExpr> {
    fn eq(&self, other: &MathExpr) -> bool {
        **self == *other
    }
}

/// Whether `expression` passes the structure check `from_ast` runs: at most `MAX_NODES` nodes,
/// depth at most `MAX_DEPTH`, and every call's arity. `check_structure` walks the borrowed tree
/// with an explicit stack, without recursion, and stops at the first violation. It checks a
/// call's arity before pushing its arguments, so every node it visits pushes at most three, and
/// it visits at most `MAX_NODES + 1` nodes whatever the input's size. Nothing may count, clone
/// or recurse into an input before it passes.
fn admitted(expression: &MathExpr) -> bool {
    expression.check_structure(MAX_NODES, MAX_DEPTH).is_ok()
}

/// The saturation limits: rounds, and the hard e-node cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Limits {
    iterations: usize,
    enodes: usize,
}

impl Limits {
    /// The production limits for an input of `nodes` nodes: eight e-nodes per input node, at
    /// least 16 (so a tiny input still has room for a few rewrites) and at most 512.
    fn for_input(nodes: usize) -> Self {
        Self {
            iterations: MAX_ITERATIONS,
            enodes: (NODE_CAP_PER_INPUT_NODE * nodes).clamp(16, NODE_CAP_MAX),
        }
    }
}

/// An e-graph after saturation, rebuilt, with how saturation went.
struct Run {
    eg: EGraph,
    root: Id,
    stop: Saturation,
    iterations: usize,
    applications: usize,
    peak: usize,
}

/// `expression` saturated under the exact rules, applied in `order`, within `limits`.
fn saturate(expression: &MathExpr, order: &[Rule], limits: Limits) -> Run {
    let mut eg = EGraph::default();
    let root = add_expr(&mut eg, &expression.root);
    let cap = limits.enodes;
    let mut stop = Saturation::IterationLimit;
    let mut applications = 0usize;
    let mut iterations = 0usize;
    let mut peak = eg.enodes();
    'rounds: for _ in 0..limits.iterations {
        iterations += 1;
        // Read phase: every match, in rule order, then class id, then node order.
        let mut pending: Vec<(Id, Term)> = Vec::new();
        let roots = eg.roots();
        for &rule in order {
            for &id in &roots {
                for node in &eg.classes[id] {
                    for rhs in matches(&eg, rule, node) {
                        pending.push((id, rhs));
                    }
                }
            }
        }
        // Write phase, checking the cap before every application. The round changed the graph
        // if an application created an e-node (every new e-node opens a class) or merged two
        // classes; a round that did neither leaves the graph, and so every later round, as is.
        let mut changed = false;
        for (id, rhs) in pending {
            if eg.enodes() + term_size(&rhs) > cap {
                stop = Saturation::NodeLimit;
                eg.rebuild();
                break 'rounds;
            }
            let classes = eg.parent.len();
            let new = eg.insert(&rhs);
            let merged = eg.union(id, new);
            changed |= merged || eg.parent.len() > classes;
            peak = peak.max(eg.enodes());
            applications += 1;
        }
        eg.rebuild();
        if !changed {
            stop = Saturation::Saturated;
            break;
        }
    }
    Run {
        eg,
        root,
        stop,
        iterations,
        applications,
        peak,
    }
}

/// `expression` simplified under the exact rules, applied in `order`, within `limits`. The
/// structure check comes first: an input that fails it is returned unchanged, before anything
/// counts, clones or recurses into it, and before any rule sees it.
fn simplify_with<'a>(expression: &'a MathExpr, order: &[Rule], limits: Limits) -> Simplified<'a> {
    if !admitted(expression) {
        return Simplified::refused(expression);
    }
    let run = saturate(expression, order, limits);
    let best = extract(&run.eg);
    let original = (count_nodes(&expression.root), depth(&expression.root));
    let candidate = best
        .get(&run.eg.find(run.root))
        .and_then(|tree| MathExpr::from_ast(tree.clone()).ok());
    let (expression, changed) = match candidate {
        Some(tree) => {
            let size = (count_nodes(&tree.root), depth(&tree.root));
            if size < original {
                (Cow::Owned(tree), true)
            } else {
                (Cow::Borrowed(expression), false)
            }
        }
        None => (Cow::Borrowed(expression), false),
    };
    Simplified {
        expression,
        changed,
        stop: run.stop,
        iterations: run.iterations,
        enodes: run.eg.enodes(),
        applications: run.applications,
        peak: run.peak,
    }
}

/// [`simplify_with`] under every rule, within the production limits for the input's size, which
/// is counted only once the input has passed the structure check.
fn simplify(expression: &MathExpr) -> Simplified<'_> {
    if !admitted(expression) {
        return Simplified::refused(expression);
    }
    simplify_with(
        expression,
        &RULES,
        Limits::for_input(count_nodes(&expression.root)),
    )
}

// laneD:B20-rewrite-simplify: the public entry.
/// Why `simplify_v2` refuses a tree outside the structure limits, before any copy of it.
const STRUCTURE_REFUSAL: &str =
    "simplify_v2 refuses a tree outside the structure limits before it is copied";

impl MathExpr {
    /// The exact simplification of an expression the profile admits under `scope`. It reads no
    /// rows: the result depends on the tree alone.
    ///
    /// `self` must pass the same admission as every search edit (`Search::finish`: the pool, the
    /// caps and the negation rule), or this is an error. `simplify` then runs with the
    /// production rules and limits, and its smaller tree is kept only when that admission accepts
    /// it too. A smaller tree the profile refuses, such as a negation of a variable under a
    /// constants-only negation rule, is discarded and its refusal returned.
    ///
    /// `self` may be any tree, one decoded from JSON included, and the copy and the admission's
    /// normalization both recurse. So the borrowed tree's structure check (`from_ast`'s node,
    /// depth and arity limits, walked with an explicit stack) comes first, and a tree outside it is
    /// refused before anything copies or walks it recursively.
    pub fn simplify_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
    ) -> Result<ExpressionSimplification> {
        self.check_structure(MAX_NODES, MAX_DEPTH)
            .map_err(|error| error.context(STRUCTURE_REFUSAL))?;
        let search = Search::new(profile, scope)?;
        let input = search.finish(self.root.clone())?;
        let result = simplify(&input);
        let (simplified, refusal) = if result.changed {
            match search.finish(result.expression.root.clone()) {
                Ok(expression) => (Some(expression), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            }
        } else {
            (None, None)
        };
        Ok(ExpressionSimplification {
            simplified,
            refusal,
            stop: result.stop.public(),
            iterations: result.iterations,
            enodes: result.enodes,
            applications: result.applications,
            peak: result.peak,
        })
    }
}
