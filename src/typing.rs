use slotmap::DefaultKey as NodeKey;
use std::collections::HashMap;
use ustr::{Ustr, UstrMap};

use crate::{AST, ASTNodeType};

// TODO: Report errors nicely instead of panicking at first failure.
// My thought of how I'll do this is to return Type::Hole everywhere checking
// and inference fails after printing the error message to the screen.
// TODO: Consider `Cow` to reduce the amount of cloning we do?

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    // For type inference.
    Hole,
    Unit,
    // Arbitrary precision integers, with a notion of "packed struct."
    // In non-packed environments, APInts are rounded up to their word-aligned
    // size.
    Int(usize),
    Nat(usize),
    // Maybe we should constrain the set of possible float sizes?
    Float(usize),
    String,
    // `Comptime` types assume max(sizeof(their value), use)
    // This is done for convenience -- they're comptime-known, so they're
    // inline or looked-up immediately and constant-propagated.
    // At the end of type inference, if a value is still "comptime...", the value assumes the
    // default type (`int` = Int { size = 64, signed = true }, `float` = Float { size = 64 })
    ComptimeInt,
    ComptimeFloat,
    // I literally just swapped `String` for `Ustr` and it worked transparently
    // This is why I call `.into` on `str` literals that need to behave as `String`s
    // and not `to_string`
    Custom(Ustr),
    Func(Vec<Type>),
}

impl Type {
    pub fn coerces_to_int(&self) -> bool {
        matches!(self, Self::ComptimeInt)
    }

    pub fn coerces_to_float(&self) -> bool {
        matches!(self, Self::ComptimeInt | Self::ComptimeFloat)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Nat(_) | Self::ComptimeInt)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float(_) | Self::ComptimeFloat)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }

    pub fn is_comptime(&self) -> bool {
        matches!(self, Self::ComptimeInt | Self::ComptimeFloat)
    }

    // I guess "congruent" is more accurate but not every programmer is Terrence Tao (I wish I
    // was lol)
    pub fn eq_modulo_comptime(&self, other: &Self) -> bool {
        let mut lhs = self;
        let mut rhs = other;
        if lhs.is_comptime() && !rhs.is_comptime() {
            std::mem::swap(&mut lhs, &mut rhs);
        }

        match (lhs, rhs) {
            (t1, Self::ComptimeInt) if t1.is_int() => true,
            (t1, Self::ComptimeFloat) if t1.is_float() => true,
            (t1, t2) => t1 == t2,
        }
    }
}

impl From<&str> for Type {
    fn from(raw: &str) -> Self {
        match raw {
            "unit" => Type::Unit,
            "int" => Type::Int(64),
            "nat" => Type::Nat(64),
            "f32" => Type::Float(32),
            "f64" => Type::Float(64),
            name => Type::Custom(name.into()),
        }
    }
}

// Oh, this is simpler than I thought.
// We only need a stack of type environments that track all things
// in scope at a particular program point for checking and inference.
//
// TypeEnv : BlockNode -> VariableName -> Type
pub type TypeEnv = HashMap<NodeKey, UstrMap<Type>>;

/// Constructs a `name : type` environment for the top-level module node and blocks.
/// The typing environment relates each AST module and block nodes' keys with a map
/// that contains the definitions they introduce.
pub fn infer_types(ast: &AST) -> TypeEnv {
    // "Root" node is always a "module" node.
    let mut type_env = TypeEnv::default();
    // The list of defintions to search when inferring types.
    let mut ctx = vec![ast.root];
    type_env.insert(ast.root, UstrMap::default());
    for a in &ast.nodes[ast.root].args {
        typeof_stmt(ast, *a, &mut type_env, &mut ctx);
    }

    type_env
}

fn typeof_stmt(ast: &AST, stmt: NodeKey, type_env: &mut TypeEnv, ctx: &mut Vec<NodeKey>) -> Type {
    use crate::ast::StmtOp;
    let ASTNodeType::Stmt(node_kind) = ast.nodes[stmt].ty else {
        panic!("Bad AST -- got {:?} instead of stmt.", ast.nodes[stmt]);
    };

    let expr_node = ast.nodes[stmt].args[match node_kind {
        StmtOp::Pure => 0,
        _ => 1,
    }];

    let expr_ty = typeof_expr(ast, expr_node, type_env, ctx);
    if node_kind != StmtOp::Pure {
        // There are two cases: decl or just assignment.
        // For decl nodes, we introduce the defintion. For assignment, we assert that we have the
        // definition and that its type matches the RHS.
        let lhs_key = ast.nodes[stmt].args[0];
        let ty_lhs = typeof_lhs(ast, lhs_key, type_env, ctx);
        match ty_lhs {
            Type::Hole => {
                let name_key = ast.nodes[lhs_key].args[0];
                let ASTNodeType::Ident(name) = ast.nodes[name_key].ty else {
                    panic!("Expected identifier, found something else...");
                };
                let scope = ctx.last().expect("The root node should never be popped.");
                let block_env = type_env
                    .get_mut(scope)
                    .expect("What, it's on the stack but not in this map?");
                block_env.insert(name, expr_ty);
            }
            // TODO: Coerce `ComptimeInt` and `ComptimeFloat` to concrete types for LHS
            // (We should only do this for `Assign` nodes and not `Define` nodes because they are
            // alive at runtime and not just compile time.)
            ty if ty != &expr_ty => panic!("LHS has type {ty:?} but RHS has type {expr_ty:?}"),
            _ => {}
        }

        // Choosing this type because that's what `;` does to an expr in OCaml.
        // (It seems like a sensible choice)
        Type::Unit
    } else {
        expr_ty
    }
}

fn typeof_lhs<'env>(
    ast: &'env AST,
    lhs: NodeKey,
    type_env: &'env mut TypeEnv,
    ctx: &mut [NodeKey],
) -> &'env Type {
    use crate::ast::ExprOp;
    match &ast.nodes[lhs].ty {
        ASTNodeType::Expr(ExprOp::Decl) => {
            let ASTNodeType::Type(ref ty) = ast.nodes[ast.nodes[lhs].args[1]].ty else {
                panic!("Expected type, found {:?}", ast.nodes[lhs].ty);
            };

            ty
        }
        ASTNodeType::Ident(name) => lookup_ctx(type_env, ctx, *name)
            .unwrap_or_else(|| panic!("Variable referenced but undeclared: {name}")),
        not_a_var => panic!("Expected identifier or decl, found {not_a_var:?}"),
    }
}

fn typeof_expr(ast: &AST, expr: NodeKey, type_env: &mut TypeEnv, ctx: &mut Vec<NodeKey>) -> Type {
    match &ast.nodes[expr].ty {
        ASTNodeType::Expr(_) => typeof_op(ast, expr, type_env, ctx),
        ASTNodeType::Ident(name) => {
            // Function types are the only thing that's messing this up...
            lookup_ctx(type_env, ctx, *name)
                .expect(&format!("Reference to undeclared variable {name}"))
                .clone()
        }
        ASTNodeType::Literal(val) => val.ty(),
        not_expr => panic!("Expected op, identifier, or literal, found {not_expr:?}"),
    }
}

fn typeof_op(ast: &AST, expr: NodeKey, type_env: &mut TypeEnv, ctx: &mut Vec<NodeKey>) -> Type {
    use crate::ast::ExprOp;

    let ASTNodeType::Expr(op) = ast.nodes[expr].ty else {
        panic!("Expected expr node, found {:?}", ast.nodes[expr]);
    };

    // We aren't going just by arg count since we could have keyword "signatures" that depend on
    // the operator in question (e.g. `mod` only admits int as second argument,
    // `if` bool {block : T} `else` {block : T} : T)
    match op {
        ExprOp::Group => typeof_expr(ast, ast.nodes[expr].args[0], type_env, ctx),
        ExprOp::Negate => {
            let child_ty = typeof_expr(ast, ast.nodes[expr].args[0], type_env, ctx);
            if child_ty.is_numeric() {
                child_ty
            } else {
                panic!("Expected numeric type, found {child_ty:?}");
            }
        }
        ExprOp::Add | ExprOp::Subtract | ExprOp::Multiply | ExprOp::Divide => {
            let mut lhs_ty = typeof_expr(ast, ast.nodes[expr].args[0], type_env, ctx);
            let mut rhs_ty = typeof_expr(ast, ast.nodes[expr].args[1], type_env, ctx);

            // "Canonicalize" to int on the left, float on the right.
            if lhs_ty.is_float() && rhs_ty.is_int() {
                std::mem::swap(&mut lhs_ty, &mut rhs_ty);
            }

            // Suppose both are floats or both are ints -- move the concrete type to the left.
            if lhs_ty == rhs_ty && lhs_ty.is_comptime() && !rhs_ty.is_comptime() {
                std::mem::swap(&mut lhs_ty, &mut rhs_ty);
            }

            // TODO: Decide if we want stricter integer size typing within an expr.
            match (lhs_ty, rhs_ty) {
                (Type::Int(sz1), Type::Int(sz2)) => Type::Int(sz1.max(sz2)),
                (Type::Nat(sz1), Type::Nat(sz2)) => Type::Nat(sz1.max(sz2)),
                (Type::Float(sz1), Type::Float(sz2)) => Type::Float(sz1.max(sz2)),
                (Type::ComptimeInt, Type::Float(sz)) => Type::Float(sz),
                (Type::ComptimeInt, Type::ComptimeInt) => Type::ComptimeInt,
                (Type::ComptimeInt | Type::ComptimeFloat, Type::ComptimeFloat) => {
                    Type::ComptimeFloat
                }
                (t1, t2) => panic!("Type mismatch -- lhs: {t1:?}, rhs: {t2:?}"),
            }
        }
        ExprOp::Mod => {
            let lhs_ty = typeof_expr(ast, ast.nodes[expr].args[0], type_env, ctx);
            let rhs_ty = typeof_expr(ast, ast.nodes[expr].args[1], type_env, ctx);
            match (lhs_ty, rhs_ty) {
                (Type::Int(sz1), Type::Int(sz2)) => Type::Int(sz1.max(sz2)),
                (Type::Int(sz1), Type::ComptimeInt) => Type::Int(sz1),
                (Type::ComptimeInt, Type::ComptimeInt) => Type::ComptimeInt,
                (t1, t2) => panic!("Modulo expects integers, got {t1:?} % {t2:?} instead."),
            }
        }
        ExprOp::Func => {
            // Get the parameters into the environment
            let [args_key, ret_key, body_key] = ast.nodes[expr].args[0..3] else {
                panic!("Got a func node without three children, this shouldn't happen!");
            };
            type_env.insert(body_key, UstrMap::default());
            let mut arg_types = Vec::new();

            ctx.push(body_key);
            // I have the arguments wrapped up into a "group" node
            for arg_decl in &ast.nodes[args_key].args {
                // There are stmts, actually.
                typeof_stmt(ast, *arg_decl, type_env, ctx);
                // Kinda messy, but we can pull this out.
                let arg_ty = match &ast.nodes[ast.nodes[ast.nodes[*arg_decl].args[0]].args[1]].ty {
                    ASTNodeType::Type(ty) => ty.clone(),
                    _ => unreachable!(),
                };

                arg_types.push(arg_ty);
            }
            ctx.pop();

            let body_ret_ty = typeof_block(ast, body_key, type_env, ctx);
            let ASTNodeType::Type(sig_ret_ty) = &ast.nodes[ret_key].ty else {
                panic!("Expected a return type, got something else instead...");
            };

            if *sig_ret_ty == body_ret_ty {
                arg_types.push(body_ret_ty.clone());
                Type::Func(arg_types)
            } else {
                panic!(
                    "Type mismatch! Function body returns {body_ret_ty:?} but expected {sig_ret_ty:?}"
                );
            }
        }
        ExprOp::Block => typeof_block(ast, expr, type_env, ctx),
        ExprOp::Decl => {
            let [name_key, type_key] = ast.nodes[expr].args[0..2] else {
                panic!("Expected a name node and a type node.");
            };

            let (ASTNodeType::Ident(name), ASTNodeType::Type(ty)) =
                (&ast.nodes[name_key].ty, &ast.nodes[type_key].ty)
            else {
                panic!(
                    "Expected a name node and a type node, got {:?} and {:?} instead.",
                    ast.nodes[name_key].ty, ast.nodes[type_key].ty
                );
            };

            let scope = ctx.last().expect("How'd we get here without a scope?");
            // NOTE: Something is telling me right now that the AST shouldn't "own" the types of
            // terms, and rather that the nodes were only placed there so I could recover them.
            // Maybe this copy is necessary, depends on what my ambitions are for tooling (that
            // will likely never exist lol, but I like the challenge of designing for that
            // possibility)
            type_env.get_mut(scope).unwrap().insert(*name, ty.clone());
            Type::Unit
        }
        ExprOp::Call => {
            // Look up the type of the function, pass up its return type.
            let ASTNodeType::Ident(func_name) = ast.nodes[ast.nodes[expr].args[0]].ty else {
                panic!("Expected function name, found something else...");
            };

            let call_args: Vec<_> = ast.nodes[expr].args[1..]
                .iter()
                .map(|a| typeof_expr(ast, *a, type_env, ctx))
                .collect();

            let Some(Type::Func(func_args)) = lookup_ctx(type_env, ctx, func_name) else {
                panic!(
                    "Calling function before its definition (or its type is not a function): {func_name}"
                );
            };

            // Check the arity and iterate over the things passed in.
            let call_arity = call_args.len();
            let func_arity = func_args.len() - 1;
            assert_eq!(
                call_arity, func_arity,
                "Arity mismatch: function {func_name} \
                has {func_arity} args but was called with {call_arity} args."
            );

            // Now, let's make sure all the argument types match up.
            for (i, (fa, ca)) in func_args.iter().zip(call_args).enumerate() {
                if !fa.eq_modulo_comptime(&ca) {
                    panic!(
                        "Argument {i} in '{func_name}': Expected argument type {fa:?}, got type {ca:?} instead"
                    );
                }
            }

            func_args
                .last()
                .expect("Functions that return \"nothing\" should return unit!")
                .clone()
        }
        n => todo!("Please implement {n:?}"),
    }
}

// Here, we introduce a new scope.
fn typeof_block(ast: &AST, block: NodeKey, type_env: &mut TypeEnv, ctx: &mut Vec<NodeKey>) -> Type {
    // If we're a function body we just want to use the same env as the parameters.
    if type_env.get(&block).is_none() {
        type_env.insert(block, UstrMap::default());
    }
    ctx.push(block);
    // Empty blocks should have type `unit`
    let mut tail_type = Type::Unit;
    for stmt in &ast.nodes[block].args {
        tail_type = typeof_stmt(ast, *stmt, type_env, ctx);
    }
    ctx.pop();

    tail_type
}

fn lookup_ctx<'env>(
    type_env: &'env TypeEnv,
    ctx: &[NodeKey],
    var_name: Ustr,
) -> Option<&'env Type> {
    ctx.iter()
        .rev()
        .flat_map(|scope| type_env[scope].get(&var_name))
        .next()
}
