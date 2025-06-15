use slotmap::{DefaultKey as NodeKey, Key, SlotMap};
use smallvec::SmallVec;
use std::fmt::Display;
use ustr::Ustr;

use crate::{TokenType, Type};

// NOTE: This is not the canonical Rust way of building an AST!
#[derive(Debug, Default)]
pub struct AST<'src> {
    pub nodes: SlotMap<NodeKey, ASTNode<'src>>,
    pub root: NodeKey,
}

type Children = SmallVec<[NodeKey; 3]>;

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
                writeln!(f, "{:?}", node.ty)?;
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
#[derive(Debug)]
pub enum ASTNodeType<'src> {
    Ident(Ustr),
    Literal { val: TokenType<'src>, ty: Type },
    Type(Type),
    Expr(ExprOp),
    Stmt(StmtOp),
    Module,
}

// The order of the enum fields defines operator precedence
// for parsing.
// Appears later -> binds tighter (closer to leaves).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExprOp {
    // Admits a `Group` of `Decl`s (params), a return type or another `Group` of `Decl`s and a `Block`.
    Func,
    // Regions of code delimited by two curly braces.
    Block,
    AndThen,

    // For `()` in arithmetic and procedure literals.
    Group,
    Call,
    SeqSep,

    // Decl is the first colon in `name: tau`, `name := v`, `name :: v`, `name: tau = v`, etc.
    // We need Decl > SeqSep > Group to build procedure parameter lists.
    Decl,

    // Arithmetic
    Subtract,
    Add,
    Divide,
    Multiply,
    Mod,
    Negate,
}

impl ExprOp {
    pub fn arg_count(&self) -> usize {
        match self {
            Self::SeqSep | Self::AndThen => 0,
            Self::Group | Self::Block | Self::Negate => 1,
            Self::Func => 3, // Params, Returns, Block
            _ => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StmtOp {
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
        StmtOp::Pure => 0,
        StmtOp::Assign | StmtOp::Define => 1,
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
                ExprOp::Negate => match &ast.nodes[ast.nodes[expr].args[0]].ty {
                    ASTNodeType::Literal { val, .. } => match val {
                        TokenType::IntLit(i) => TokenType::IntLit(-i),
                        TokenType::FloatLit(i) => TokenType::FloatLit(-i),
                        _ => return None,
                    },
                    _ => return None,
                },
                ExprOp::Add => {
                    match (
                        &ast.nodes[ast.nodes[expr].args[0]].ty,
                        &ast.nodes[ast.nodes[expr].args[1]].ty,
                    ) {
                        (
                            ASTNodeType::Literal { val: val1, .. },
                            ASTNodeType::Literal { val: val2, .. },
                        ) => match (val1, val2) {
                            (TokenType::IntLit(i), TokenType::IntLit(j)) => {
                                TokenType::IntLit(i + j)
                            }
                            (TokenType::FloatLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit(i + j)
                            }
                            (TokenType::IntLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit((*i as f64) + j)
                            }
                            (TokenType::FloatLit(i), TokenType::IntLit(j)) => {
                                TokenType::FloatLit(i + (*j as f64))
                            }
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                ExprOp::Subtract => {
                    match (
                        &ast.nodes[ast.nodes[expr].args[0]].ty,
                        &ast.nodes[ast.nodes[expr].args[1]].ty,
                    ) {
                        (
                            ASTNodeType::Literal { val: val1, .. },
                            ASTNodeType::Literal { val: val2, .. },
                        ) => match (val1, val2) {
                            (TokenType::IntLit(i), TokenType::IntLit(j)) => {
                                TokenType::IntLit(i - j)
                            }
                            (TokenType::FloatLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit(i - j)
                            }
                            (TokenType::IntLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit((*i as f64) - j)
                            }
                            (TokenType::FloatLit(i), TokenType::IntLit(j)) => {
                                TokenType::FloatLit(i - (*j as f64))
                            }
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                ExprOp::Multiply => {
                    match (
                        &ast.nodes[ast.nodes[expr].args[0]].ty,
                        &ast.nodes[ast.nodes[expr].args[1]].ty,
                    ) {
                        (
                            ASTNodeType::Literal { val: val1, .. },
                            ASTNodeType::Literal { val: val2, .. },
                        ) => match (val1, val2) {
                            (TokenType::IntLit(i), TokenType::IntLit(j)) => {
                                TokenType::IntLit(i * j)
                            }
                            (TokenType::FloatLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit(i * j)
                            }
                            (TokenType::IntLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit((*i as f64) * j)
                            }
                            (TokenType::FloatLit(i), TokenType::IntLit(j)) => {
                                TokenType::FloatLit(i * (*j as f64))
                            }
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                ExprOp::Divide => {
                    match (
                        &ast.nodes[ast.nodes[expr].args[0]].ty,
                        &ast.nodes[ast.nodes[expr].args[1]].ty,
                    ) {
                        (
                            ASTNodeType::Literal { val: val1, .. },
                            ASTNodeType::Literal { val: val2, .. },
                        ) => match (val1, val2) {
                            (TokenType::IntLit(i), TokenType::IntLit(j)) => {
                                TokenType::IntLit(i / j)
                            }
                            (TokenType::FloatLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit(i / j)
                            }
                            (TokenType::IntLit(i), TokenType::FloatLit(j)) => {
                                TokenType::FloatLit((*i as f64) / j)
                            }
                            (TokenType::FloatLit(i), TokenType::IntLit(j)) => {
                                TokenType::FloatLit(i / (*j as f64))
                            }
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                ExprOp::Mod => {
                    match (
                        &ast.nodes[ast.nodes[expr].args[0]].ty,
                        &ast.nodes[ast.nodes[expr].args[1]].ty,
                    ) {
                        (
                            ASTNodeType::Literal { val: val1, .. },
                            ASTNodeType::Literal { val: val2, .. },
                        ) => match (val1, val2) {
                            (TokenType::IntLit(i), TokenType::IntLit(j)) => {
                                TokenType::IntLit(i % j)
                            }
                            _ => return None,
                        },
                        _ => return None,
                    }
                }
                ExprOp::Group if ast.nodes[expr].args.len() == 1 => {
                    let child = ast.nodes[expr].args[0];
                    if let ASTNodeType::Literal { val, .. } = &mut ast.nodes[child].ty {
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

        let ty = match new_lit {
            TokenType::IntLit(_) => Type::ComptimeInt,
            TokenType::FloatLit(_) => Type::ComptimeFloat,
            TokenType::StrLit(_) => Type::String,
            _ => unreachable!(),
        };

        if let Some(dead_node) = ast.nodes.remove(expr) {
            for arg in dead_node.args {
                ast.nodes.remove(arg);
            }
        }

        Some(
            ast.nodes
                .insert(ASTNode::leaf(ASTNodeType::Literal { val: new_lit, ty })),
        )
    } else {
        None
    }
}
