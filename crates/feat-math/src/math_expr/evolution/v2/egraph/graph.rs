type Id = usize;

/// The most saturation rounds.
const MAX_ITERATIONS: usize = 8;
/// The hard e-node cap: eight times the input's nodes, at least 16 and at most 512.
const NODE_CAP_PER_INPUT_NODE: usize = 8;
const NODE_CAP_MAX: usize = 512;

const ADD: u8 = 0;
const SUB: u8 = 1;
const MUL: u8 = 2;
const DIV: u8 = 3;
const POW: u8 = 4;

fn tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add => ADD,
        BinOp::Sub => SUB,
        BinOp::Mul => MUL,
        BinOp::Div => DIV,
        BinOp::Pow => POW,
    }
}

fn op(tag: u8) -> BinOp {
    [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow][tag as usize]
}

/// An e-node: an operator over e-class ids. Literals are keyed by their bits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    Const(u64),
    Var(Var),
    Neg(Id),
    Bin(u8, Id, Id),
    Call(Func, Vec<Id>),
}

/// A right-hand side over existing e-classes.
#[derive(Clone, Debug)]
enum Term {
    Class(Id),
    Literal(f64),
    Neg(Box<Term>),
    Bin(u8, Box<Term>, Box<Term>),
    Call(Func, Vec<Term>),
}

/// Why saturation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Saturation {
    /// A round changed nothing: it created no e-node and merged no classes.
    Saturated,
    IterationLimit,
    /// The next application might pass the e-node cap: `term_size` is an upper bound, and the
    /// hashcons keeps stale keys until the rebuild.
    NodeLimit,
    /// The input fails the structure check, so saturation never started.
    Malformed,
}

impl Saturation {
    /// The public name of this stop, for records.
    fn public(self) -> SaturationStop {
        match self {
            Self::Saturated => SaturationStop::Saturated,
            Self::IterationLimit => SaturationStop::IterationLimit,
            Self::NodeLimit => SaturationStop::NodeLimit,
            Self::Malformed => SaturationStop::Malformed,
        }
    }
}

/// The rules, in their fixed application order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rule {
    MulOne,
    DivOne,
    AddNegZero,
    SubZero,
    NegNeg,
    AbsNeg,
    AbsAbs,
    SubToAddNeg,
    AddNegToSub,
    SubNegToAdd,
    MulNegNeg,
    MulNegLeft,
    MulNegRight,
    MulTwo,
    MulMinusOne,
    DivMinusOne,
    DivPowerOfTwo,
    Fold,
}

const RULES: [Rule; 18] = [
    Rule::MulOne,
    Rule::DivOne,
    Rule::AddNegZero,
    Rule::SubZero,
    Rule::NegNeg,
    Rule::AbsNeg,
    Rule::AbsAbs,
    Rule::SubToAddNeg,
    Rule::AddNegToSub,
    Rule::SubNegToAdd,
    Rule::MulNegNeg,
    Rule::MulNegLeft,
    Rule::MulNegRight,
    Rule::MulTwo,
    Rule::MulMinusOne,
    Rule::DivMinusOne,
    Rule::DivPowerOfTwo,
    Rule::Fold,
];

#[derive(Default)]
struct EGraph {
    parent: Vec<Id>,
    classes: Vec<BTreeSet<Node>>,
    memo: BTreeMap<Node, Id>,
}

impl EGraph {
    fn find(&self, mut id: Id) -> Id {
        while self.parent[id] != id {
            id = self.parent[id];
        }
        id
    }

    fn canonical(&self, node: &Node) -> Node {
        match node {
            Node::Neg(a) => Node::Neg(self.find(*a)),
            Node::Bin(t, a, b) => Node::Bin(*t, self.find(*a), self.find(*b)),
            Node::Call(f, args) => Node::Call(*f, args.iter().map(|a| self.find(*a)).collect()),
            other => other.clone(),
        }
    }

    /// The number of distinct e-nodes.
    fn enodes(&self) -> usize {
        self.memo.len()
    }

    fn add(&mut self, node: Node) -> Id {
        let node = self.canonical(&node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        let id = self.parent.len();
        self.parent.push(id);
        self.classes.push(BTreeSet::from([node.clone()]));
        self.memo.insert(node, id);
        id
    }

    /// Merges two classes; the lower id stays the root.
    fn union(&mut self, a: Id, b: Id) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        let (root, child) = (a.min(b), a.max(b));
        self.parent[child] = root;
        let moved = std::mem::take(&mut self.classes[child]);
        self.classes[root].extend(moved);
        true
    }

    /// The live classes: every id that is its own root.
    fn roots(&self) -> Vec<Id> {
        (0..self.classes.len())
            .filter(|&id| self.find(id) == id)
            .collect()
    }

    /// Restores congruence: re-canonicalizes every node and merges classes that now hold the
    /// same node, until nothing changes.
    fn rebuild(&mut self) {
        loop {
            let mut memo: BTreeMap<Node, Id> = BTreeMap::new();
            let mut merges = Vec::new();
            for id in self.roots() {
                let nodes: BTreeSet<Node> =
                    self.classes[id].iter().map(|n| self.canonical(n)).collect();
                for node in &nodes {
                    match memo.get(node) {
                        Some(&other) if self.find(other) != id => merges.push((other, id)),
                        Some(_) => {}
                        None => {
                            memo.insert(node.clone(), id);
                        }
                    }
                }
                self.classes[id] = nodes;
            }
            let mut changed = false;
            for (a, b) in merges {
                changed |= self.union(a, b);
            }
            if !changed {
                self.memo = memo.into_iter().map(|(n, id)| (n, self.find(id))).collect();
                return;
            }
        }
    }

    fn insert(&mut self, term: &Term) -> Id {
        match term {
            Term::Class(id) => self.find(*id),
            Term::Literal(v) => {
                if v.is_sign_negative() {
                    let inner = self.add(Node::Const((-v).to_bits()));
                    self.add(Node::Neg(inner))
                } else {
                    self.add(Node::Const(v.to_bits()))
                }
            }
            Term::Neg(a) => {
                let a = self.insert(a);
                self.add(Node::Neg(a))
            }
            Term::Bin(t, a, b) => {
                let (a, b) = (self.insert(a), self.insert(b));
                self.add(Node::Bin(*t, a, b))
            }
            Term::Call(f, args) => {
                let args = args.iter().map(|a| self.insert(a)).collect();
                self.add(Node::Call(*f, args))
            }
        }
    }

    /// The literal value a class holds: a constant, or the negation of one.
    fn literal(&self, id: Id) -> Option<f64> {
        let id = self.find(id);
        let mut found = None;
        for node in &self.classes[id] {
            match node {
                Node::Const(bits) => return Some(f64::from_bits(*bits)),
                Node::Neg(inner) if found.is_none() => {
                    found = self.classes[self.find(*inner)]
                        .iter()
                        .find_map(|n| match n {
                            Node::Const(bits) => Some(-f64::from_bits(*bits)),
                            _ => None,
                        });
                }
                _ => {}
            }
        }
        found
    }

    fn negations(&self, id: Id) -> Vec<Id> {
        self.classes[self.find(id)]
            .iter()
            .filter_map(|n| match n {
                Node::Neg(a) => Some(*a),
                _ => None,
            })
            .collect()
    }

    fn holds_abs(&self, id: Id) -> bool {
        self.classes[self.find(id)]
            .iter()
            .any(|n| matches!(n, Node::Call(Func::Abs, args) if args.len() == 1))
    }
}
