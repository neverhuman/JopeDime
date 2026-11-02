/// An upper bound on the e-nodes a right-hand side can add.
fn term_size(term: &Term) -> usize {
    match term {
        Term::Class(_) => 0,
        Term::Literal(_) => 2,
        Term::Neg(a) => 1 + term_size(a),
        Term::Bin(_, a, b) => 1 + term_size(a) + term_size(b),
        Term::Call(_, args) => 1 + args.iter().map(term_size).sum::<usize>(),
    }
}

fn add_expr(eg: &mut EGraph, expr: &Expr) -> Id {
    match expr {
        Expr::Const(v) => eg.add(Node::Const(v.to_bits())),
        Expr::Var(v) => eg.add(Node::Var(*v)),
        Expr::Unary(UnOp::Neg, a) => {
            let a = add_expr(eg, a);
            eg.add(Node::Neg(a))
        }
        Expr::Binary(o, a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(Node::Bin(tag(*o), a, b))
        }
        Expr::Call(f, args) => {
            let args = args.iter().map(|a| add_expr(eg, a)).collect();
            eg.add(Node::Call(*f, args))
        }
    }
}

type Best = BTreeMap<Id, Expr>;

/// Extraction's order on trees: nodes, then depth, then the `Debug` form, then the literals'
/// bits in order (`Debug` prints every NaN alike), so equal ranks mean equal trees.
fn rank(tree: &Expr) -> (usize, usize, String, Vec<u64>) {
    fn literals(tree: &Expr, out: &mut Vec<u64>) {
        match tree {
            Expr::Const(v) => out.push(v.to_bits()),
            Expr::Var(_) => {}
            Expr::Unary(_, a) => literals(a, out),
            Expr::Binary(_, a, b) => {
                literals(a, out);
                literals(b, out);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| literals(a, out)),
        }
    }
    let mut bits = Vec::new();
    literals(tree, &mut bits);
    (count_nodes(tree), depth(tree), format!("{tree:?}"), bits)
}

/// Every class's pick (see the module doc), in two phases. First the fewest nodes of any tree in
/// each class, a monotone fixed point whose result does not depend on the visiting order. Then,
/// in increasing order of that count, so that every child class is picked before its parent, the
/// lowest-ranked tree among the class's nodes that reach its count, each built over its
/// children's picks.
fn extract(eg: &EGraph) -> Best {
    let roots = eg.roots();
    let mut fewest: BTreeMap<Id, usize> = BTreeMap::new();
    loop {
        let mut changed = false;
        for &id in &roots {
            for node in &eg.classes[id] {
                let Some(n) = node_count(eg, &fewest, node) else {
                    continue;
                };
                if fewest.get(&id).is_none_or(|&m| n < m) {
                    fewest.insert(id, n);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut order: Vec<(usize, Id)> = fewest.iter().map(|(&id, &n)| (n, id)).collect();
    order.sort_unstable();
    let mut best: Best = BTreeMap::new();
    for (n, id) in order {
        let pick = eg.classes[id]
            .iter()
            .filter(|node| node_count(eg, &fewest, node) == Some(n))
            .filter_map(|node| build(eg, &best, node))
            .min_by_key(rank);
        if let Some(tree) = pick {
            best.insert(id, tree);
        }
    }
    best
}

/// The nodes of the smallest tree rooted at `node`, given each child class's fewest.
fn node_count(eg: &EGraph, fewest: &BTreeMap<Id, usize>, node: &Node) -> Option<usize> {
    let child = |id: &Id| fewest.get(&eg.find(*id)).copied();
    Some(match node {
        Node::Const(_) | Node::Var(_) => 1,
        Node::Neg(a) => 1 + child(a)?,
        Node::Bin(_, a, b) => 1 + child(a)? + child(b)?,
        Node::Call(_, args) => 1 + args.iter().map(child).sum::<Option<usize>>()?,
    })
}

fn build(eg: &EGraph, best: &Best, node: &Node) -> Option<Expr> {
    let child = |id: &Id| best.get(&eg.find(*id)).cloned();
    Some(match node {
        Node::Const(bits) => Expr::Const(f64::from_bits(*bits)),
        Node::Var(v) => Expr::Var(*v),
        Node::Neg(a) => Expr::Unary(UnOp::Neg, Box::new(child(a)?)),
        Node::Bin(t, a, b) => Expr::Binary(op(*t), Box::new(child(a)?), Box::new(child(b)?)),
        Node::Call(f, args) => Expr::Call(*f, args.iter().map(child).collect::<Option<Vec<_>>>()?),
    })
}
