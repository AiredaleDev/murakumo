use slotmap::DefaultKey;
use ustr::UstrMap;

use crate::{AST, ASTNodeType, ast::Type};

// This code also builds a name environment... or maybe we should
// do that during parsing...
pub fn unify_types(ast: &mut AST) -> UstrMap<Type> {
    // "Root" node is always a "module" node.

    let mut type_env = UstrMap::default();
    ast.with_args(ast.root, |ast, args| {
        for a in args {
            unify_stmt(ast, *a, &mut type_env);
        }
    });

    type_env
}

fn unify_stmt(ast: &mut AST<'_>, stmt: DefaultKey, type_env: &mut UstrMap<Type>) {
    use crate::ast::StmtOp;
    let expr_node = ast.nodes[stmt].args[match ast.nodes[stmt].ty {
        ASTNodeType::Stmt(StmtOp::Pure) => 0,
        _ => 1,
    }];

    let expr_ty = unify_expr(ast, expr_node, type_env);
    if ast.nodes[stmt].args.len() == 2 {}
}

fn unify_expr(ast: &mut AST, expr: DefaultKey, type_env: &mut UstrMap<Type>) -> Type {
    use crate::ast::ExprOp;

    todo!()
}
