use serde_json::json;
use slotmap::{DefaultKey as NodeKey, Key, SlotMap};
use smallvec::SmallVec;
use std::{cell::OnceCell, fmt::Display};
use ustr::Ustr;

use crate::{TokenType, Type};

// NOTE: This is not the canonical Rust way of building an AST!
#[derive(Debug, Default)]
pub struct AST<'src> {
    pub nodes: SlotMap<NodeKey, ASTNode<'src>>,
    pub root: NodeKey,
}

type Children = SmallVec<[NodeKey; 3]>;
type NodeMap = serde_json::Map<String, serde_json::Value>;

impl<'src> AST<'src> {
    // Actually, your honor, the AST's slotmap, and by extension the AST itself,
    // never actually *owns* a node's arguments while updating them, so it's
    // impossible for us to fail our end of the bargain while we rebuild him.
    //
    // (This is the price I pay for doing the above)
    pub fn with_args(&mut self, node_key: NodeKey, f: impl FnOnce(&mut Self, &mut Children)) {
        let mut args = std::mem::take(&mut self.nodes[node_key].args);
        f(self, &mut args);
        self.nodes[node_key].args = args;
    }

    pub fn child(&self, node_key: NodeKey, n: usize) -> &ASTNode<'src> {
        &self.nodes[self.nodes[node_key].args[n]]
    }

    pub(crate) fn get_ident(&self, node_key: NodeKey) -> Ustr {
        match self.nodes[node_key].ty {
            ASTNodeType::Ident(n) => n,
            ASTNodeType::Expr(ExprOp::Decl)
            | ASTNodeType::Stmt(StmtKind::Define | StmtKind::Assign) => {
                self.get_ident(self.nodes[node_key].args[0])
            }
            _ => unreachable!("Only the above nodes contain idents."),
        }
    }

    // This code makes building regression tests for the parser easier.
    // Ustr doesn't support serde even though it easily could... bruh.
    pub(crate) fn to_json(&self) -> serde_json::Value {
        fn children_to_map(ast: &AST, node: NodeKey) -> serde_json::Value {
            let mut node_map = serde_json::Map::default();
            for c in &ast.nodes[node].args {
                node_map.insert(
                    format!("{:?} ({:?})", ast.nodes[*c].ty, c.data()),
                    children_to_map(ast, *c),
                );
            }
            serde_json::Value::Object(node_map)
        }

        let module_map = children_to_map(self, self.root);
        json!({
            format!("Module ({:?}):", self.root.data()): module_map
        })
    }

    // Inverse of the above, so for running tests.
    pub(crate) fn from_json(ast_as_json: serde_json::Value) -> Self {
        fn lift_node(ast: &mut AST, node: &NodeMap) {
            // Empty map -> nothing.
            // We want to pack all of the things together
            for (k, v) in node {
                // Keys encode node types (up to first space).
                let Some((node_ty_as_str, _)) = k.split_once(' ') else {
                    panic!("Nonsense JSON -- either you or `to_json` is trippin!");
                };

                // Man, these node types also don't make deserialization convenient.
                // If I did things the traditional way I wouldn't have to write this by
                // hand lol
                //
                // I think I'm going to make limited changes to the source language,
                // though remembering to add nodes to this thing isn't great.

                // Praise be to vim regex
                let node_ty = match node_ty_as_str.split_once('(') {
                    Some((first, rest)) => match rest.split_once('(') {
                        Some((second, rest)) => {}
                        None => match rest {
                            "Func" => ASTNodeType::Expr(ExprOp::Func),
                            "If" => ASTNodeType::Expr(ExprOp::If),
                            "Else" => ASTNodeType::Expr(ExprOp::Else),
                            "Block" => ASTNodeType::Expr(ExprOp::Block),
                            "Group" => ASTNodeType::Expr(ExprOp::Group),
                            "Call" => ASTNodeType::Expr(ExprOp::Call),
                            "SeqSep" => ASTNodeType::Expr(ExprOp::SeqSep),
                            "Decl" => ASTNodeType::Expr(ExprOp::Decl),
                            "Eq" => ASTNodeType::Expr(ExprOp::Eq),
                            "Neq" => ASTNodeType::Expr(ExprOp::Neq),
                            "Gt" => ASTNodeType::Expr(ExprOp::Gt),
                            "Geq" => ASTNodeType::Expr(ExprOp::Geq),
                            "Lt" => ASTNodeType::Expr(ExprOp::Lt),
                            "Leq" => ASTNodeType::Expr(ExprOp::Leq),
                            "Subtract" => ASTNodeType::Expr(ExprOp::Subtract),
                            "Add" => ASTNodeType::Expr(ExprOp::Add),
                            "Divide" => ASTNodeType::Expr(ExprOp::Divide),
                            "Multiply" => ASTNodeType::Expr(ExprOp::Multiply),
                            "Mod" => ASTNodeType::Expr(ExprOp::Mod),
                            "Negate" => ASTNodeType::Expr(ExprOp::Negate),
                            "Or" => ASTNodeType::Expr(ExprOp::Or),
                            "And" => ASTNodeType::Expr(ExprOp::And),
                            "Not" => ASTNodeType::Expr(ExprOp::Not),
                            "Pure" => ASTNodeType::Stmt(StmtKind::Pure),
                            "Assign" => ASTNodeType::Stmt(StmtKind::Assign),
                            "Define" => ASTNodeType::Stmt(StmtKind::Define),
                        },
                    },
                    None => ASTNodeType::Module,
                };
            }
        }

        let mut result = Self::default();

        // Walk the map, the annoying part is having to implement "from str" for node type.
        let json_root = ast_as_json
            .as_object()
            .expect("Come on man, you know better!");

        lift_node(&mut result, json_root);

        result
    }
}

impl Display for AST<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For printing, we want a preorder traversal.
        let mut traversal_stack = Vec::new();
        if !self.root.is_null() {
            traversal_stack.push((0, self.root));
            while let Some((depth, curr)) = traversal_stack.pop() {
                for _ in 0..depth {
                    write!(f, "| ")?;
                }
                let node = &self.nodes[curr];
                writeln!(f, "{:?}                {:?}", node.ty, curr.data())?;
                traversal_stack.extend(node.args.iter().rev().map(|v| (depth + 1, *v)));
            }

            Ok(())
        } else {
            write!(f, "{{empty tree}}")
        }
    }
}

#[derive(Debug)]
pub struct ASTNode<'src> {
    pub ty: ASTNodeType<'src>,
    // Tuples (including those in function defns and calls) as
    // well as blocks will likely end up on the heap.
    pub args: Children,
}

impl<'src> ASTNode<'src> {
    pub fn leaf(ty: ASTNodeType<'src>) -> Self {
        Self {
            ty,
            args: SmallVec::new(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.args.is_empty()
    }
}

// NOTE: Maybe Decl should be its own category of node.
#[derive(Debug, PartialEq)]
pub enum ASTNodeType<'src> {
    Ident(Ustr),
    Literal(Lit<'src>),
    Type(Type),
    Expr(ExprOp),
    Stmt(StmtKind),
    Module,
}

#[derive(Debug, Default, PartialEq)]
pub enum Lit<'src> {
    #[default]
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(&'src str),
}

impl<'s> Lit<'s> {
    pub fn ty(&self) -> Type {
        match self {
            Self::Unit => Type::Unit,
            Self::Bool(_) => Type::Bool,
            Self::Int(_) => Type::ComptimeInt,
            Self::Float(_) => Type::ComptimeFloat,
            Self::String(_) => Type::String,
        }
    }
}

impl<'src> From<TokenType<'src>> for Lit<'src> {
    fn from(value: TokenType<'src>) -> Self {
        match value {
            TokenType::IntLit(i) => Lit::Int(i),
            TokenType::FloatLit(f) => Lit::Float(f),
            TokenType::BoolLit(b) => Lit::Bool(b),
            TokenType::StrLit(s) => Lit::String(s),
            _ => unreachable!(
                "We should never attempt to construct a literal out of any other tokens."
            ),
        }
    }
}

// The order of the enum fields defines operator precedence
// for parsing.
// Appears later -> binds tighter (closer to leaves).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExprOp {
    // Admits a `Group` of `Decl`s (params), a return type or another `Group` of `Decl`s and a `Block`.
    Func,

    // Control flow constructs
    // We parse these like sequences -- `else if` (two tokens) gets merged
    // into `elif`.
    If,
    Else,

    // Regions of code delimited by two curly braces.
    Block,
    // For `()` in arithmetic and procedure literals.
    Group,
    Call,
    SeqSep,

    // Decl is the first colon in `name: tau`, `name := v`, `name :: v`, `name: tau = v`, etc.
    // We need Decl > SeqSep > Group to build procedure parameter lists.
    Decl,

    Eq,
    Neq,
    Gt,
    Geq,
    Lt,
    Leq,

    // Arithmetic
    Subtract,
    Add,
    Divide,
    Multiply,
    Mod,
    Negate,

    // Booleans
    Or,
    And,
    Not,
}

impl ExprOp {
    pub fn arg_count(&self) -> usize {
        match self {
            Self::SeqSep => 0,
            Self::Group | Self::Block | Self::Negate | Self::Not | Self::Else => 1,
            Self::Func => 3, // Params, Returns, Block
            _ => 2,          // Binops, `[El]If : BoolExpr, Block -> ASTNode`
        }
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide | Self::Mod | Self::Negate
        )
    }

    pub fn is_logical(&self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Not)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StmtKind {
    // Just `expr;` or `;`.
    // These may be "pure" but that doesn't mean the
    // code they contain has no side-effects! They are, however,
    // pure w.r.t. to the naming environment.
    Pure,
    // `=`
    Assign,
    // The second colon in `::`
    Define,
}

// I'm tired of writing parsers and checkers, let's transform some code!
// TODO: Delete this and replace it with an SCCP pass once
// we've defined our CFG
//
// If you add a `ComptimeInt` and a `ComptimeFloat`, the result is a `ComptimeFloat`.
// If you add an `f64` and a `ComptimeInt`, the result is an `f64`, although no rewriting will
// happen, that's on the typechecker.
// However, if you try to add an `f64` and an `int`, you'll get a type error.
pub fn fold_constants(ast: &mut AST) {
    ast.with_args(ast.root, |ast, args| {
        for stmt in args {
            fold_stmt(ast, *stmt);
        }
    });
}

fn fold_stmt(ast: &mut AST, stmt: NodeKey) {
    let ASTNode {
        ty: ASTNodeType::Stmt(stmt_op),
        ..
    } = &ast.nodes[stmt]
    else {
        panic!("This *should* be a stmt and parsing should've handled this already!");
    };

    let redex_ind = match stmt_op {
        StmtKind::Pure => 0,
        StmtKind::Assign | StmtKind::Define => 1,
    };

    ast.with_args(stmt, |ast, args| {
        if let Some(new_key) = fold_expr(ast, args[redex_ind]) {
            args[redex_ind] = new_key;
        }
    });
}

fn fold_expr(ast: &mut AST, expr: NodeKey) -> Option<NodeKey> {
    if !ast.nodes[expr].is_leaf() {
        // Reduce children
        ast.with_args(expr, |ast, args| {
            let skip_count = match ast.nodes[expr].ty {
                ASTNodeType::Expr(ExprOp::Func) => 2,
                ASTNodeType::Expr(ExprOp::Call) => 1,
                _ => 0,
            };

            // Blocks contain stmts, not exprs, so...
            if let ASTNodeType::Expr(ExprOp::Block) = ast.nodes[expr].ty {
                for c in args.iter_mut().skip(skip_count) {
                    fold_stmt(ast, *c);
                }
            } else {
                for c in args.iter_mut().skip(skip_count) {
                    if let Some(new_c) = fold_expr(ast, *c) {
                        *c = new_c;
                    }
                }
            }
        });

        let new_lit = match &ast.nodes[expr].ty {
            ASTNodeType::Expr(op) => match op {
                ExprOp::Negate => match &ast.child(expr, 0).ty {
                    ASTNodeType::Literal(val) => match val {
                        Lit::Int(i) => Lit::Int(-i),
                        Lit::Float(i) => Lit::Float(-i),
                        _ => return None,
                    },
                    _ => return None,
                },
                ExprOp::Add => match (&ast.child(expr, 0).ty, &ast.child(expr, 1).ty) {
                    (ASTNodeType::Literal(val1), ASTNodeType::Literal(val2)) => {
                        match (val1, val2) {
                            (Lit::Int(i), Lit::Int(j)) => Lit::Int(i + j),
                            (Lit::Float(i), Lit::Float(j)) => Lit::Float(i + j),
                            (Lit::Int(i), Lit::Float(j)) => Lit::Float((*i as f64) + j),
                            (Lit::Float(i), Lit::Int(j)) => Lit::Float(i + (*j as f64)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                },
                ExprOp::Subtract => match (&ast.child(expr, 0).ty, &ast.child(expr, 1).ty) {
                    (ASTNodeType::Literal(val1), ASTNodeType::Literal(val2)) => {
                        match (val1, val2) {
                            (Lit::Int(i), Lit::Int(j)) => Lit::Int(i - j),
                            (Lit::Float(i), Lit::Float(j)) => Lit::Float(i - j),
                            (Lit::Int(i), Lit::Float(j)) => Lit::Float((*i as f64) - j),
                            (Lit::Float(i), Lit::Int(j)) => Lit::Float(i - (*j as f64)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                },
                ExprOp::Multiply => match (&ast.child(expr, 0).ty, &ast.child(expr, 1).ty) {
                    (ASTNodeType::Literal(val1), ASTNodeType::Literal(val2)) => {
                        match (val1, val2) {
                            (Lit::Int(i), Lit::Int(j)) => Lit::Int(i * j),
                            (Lit::Float(i), Lit::Float(j)) => Lit::Float(i * j),
                            (Lit::Int(i), Lit::Float(j)) => Lit::Float((*i as f64) * j),
                            (Lit::Float(i), Lit::Int(j)) => Lit::Float(i * (*j as f64)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                },
                ExprOp::Divide => match (&ast.child(expr, 0).ty, &ast.child(expr, 1).ty) {
                    (ASTNodeType::Literal(val1), ASTNodeType::Literal(val2)) => {
                        match (val1, val2) {
                            (Lit::Int(i), Lit::Int(j)) => Lit::Int(i / j),
                            (Lit::Float(i), Lit::Float(j)) => Lit::Float(i / j),
                            (Lit::Int(i), Lit::Float(j)) => Lit::Float((*i as f64) / j),
                            (Lit::Float(i), Lit::Int(j)) => Lit::Float(i / (*j as f64)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                },
                ExprOp::Mod => match (&ast.child(expr, 0).ty, &ast.child(expr, 1).ty) {
                    (ASTNodeType::Literal(val1), ASTNodeType::Literal(val2)) => {
                        match (val1, val2) {
                            (Lit::Int(i), Lit::Int(j)) => Lit::Int(i % j),
                            _ => return None,
                        }
                    }
                    _ => return None,
                },
                ExprOp::Group if ast.nodes[expr].args.len() == 1 => {
                    let child = ast.nodes[expr].args[0];
                    if let ASTNodeType::Literal(val) = &mut ast.nodes[child].ty {
                        std::mem::take(val)
                    } else {
                        return None;
                    }
                }
                ExprOp::Func => {
                    ast.with_args(expr, |ast, args| {
                        // Blocks are updated in-place.
                        fold_expr(ast, args[2]);
                    });
                    return None;
                }
                _ => return None,
            },
            _ => panic!(
                "Got unexpected node type {:?}, \
                this should've been caught by the parser.",
                ast.nodes[expr].ty
            ),
        };

        if let Some(dead_node) = ast.nodes.remove(expr) {
            for arg in dead_node.args {
                ast.nodes.remove(arg);
            }
        }

        Some(
            ast.nodes
                .insert(ASTNode::leaf(ASTNodeType::Literal(new_lit))),
        )
    } else {
        None
    }
}
