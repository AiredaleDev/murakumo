use murakumo::{
    ast::{ASTNode, ExprOp, Lit, StmtKind},
    *,
};
use slotmap::DefaultKey as NodeKey;
use smallvec::smallvec;
use std::{fs::read_to_string, path::Path};
use ustr::Ustr;

// Compares tree structure (not tags used to build structure)
fn assert_ast_eq(ast1: &AST, ast2: &AST) {
    assert_node_eq(ast1, ast2, ast1.root, ast2.root)
}

fn assert_node_eq(ast1: &AST, ast2: &AST, node1: NodeKey, node2: NodeKey) {
    assert_eq!(
        ast1.nodes[node1].ty, ast2.nodes[node2].ty,
        "Got AST:\n{ast2}\nNodeMap = {:#?}\nLHS Children = {:?}\nRHS Children = {:?}",
        ast2.nodes, ast1.nodes[node1].args, ast2.nodes[node2].args
    );
    assert_eq!(ast1.nodes[node1].args.len(), ast2.nodes[node2].args.len());
    for (n1, n2) in ast1.nodes[node1]
        .args
        .iter()
        .zip(ast2.nodes[node2].args.iter())
    {
        assert_node_eq(ast1, ast2, *n1, *n2);
    }
}

// Maybe I'll make these part of the main codebase.
// They might be nice methods to have on `AST`
fn mk_ident(ast: &mut AST, ident: Ustr) -> NodeKey {
    ast.nodes.insert(ASTNode::leaf(ASTNodeType::Ident(ident)))
}

fn mk_lit<'src>(ast: &mut AST<'src>, lit: Lit<'src>) -> NodeKey {
    ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal(lit)))
}

fn mk_type(ast: &mut AST, ty: Type) -> NodeKey {
    ast.nodes.insert(ASTNode::leaf(ASTNodeType::Type(ty)))
}

fn mk_unop(ast: &mut AST, op: ExprOp, arg: NodeKey) -> NodeKey {
    ast.nodes.insert(ASTNode {
        ty: ASTNodeType::Expr(op),
        args: smallvec![arg],
    })
}

fn mk_binop(ast: &mut AST, op: ExprOp, lhs: NodeKey, rhs: NodeKey) -> NodeKey {
    ast.nodes.insert(ASTNode {
        ty: ASTNodeType::Expr(op),
        args: smallvec![lhs, rhs],
    })
}

fn mk_pure(ast: &mut AST, expr: NodeKey) -> NodeKey {
    ast.nodes.insert(ASTNode {
        ty: ASTNodeType::Stmt(StmtKind::Pure),
        args: smallvec![expr],
    })
}

fn compiler_vs_json(file_name: &Path) {
    let prog_dir = Path::new("tests");
    let src_code = read_to_string(
        prog_dir
            .join("test_progs")
            .join(file_name)
            .with_extension("ku"),
    )
    .expect("Whoops, wrong path!");
    let tokens = lexer::lex(&src_code).expect("I expect this to lex correctly!");
    let parsed_ast = parser::parse(tokens).expect("I expect this to parse correctly!");
    dbg!(
        prog_dir
            .join("parser_json")
            .join(file_name)
            .with_extension("json"),
    );
    let ast_json = read_to_string(
        prog_dir
            .join("parser_json")
            .join(file_name)
            .with_extension("json"),
    )
    .expect("Whoops, wrong path!");
    let expected_ast: AST = serde_json::from_str(&ast_json)
        .expect("Invalid JSON... did you copy it into a file wrong?");

    assert_ast_eq(&parsed_ast, &expected_ast);
}

#[test]
pub fn parse_pemdas() {
    compiler_vs_json(Path::new("eval_order"));
}

#[test]
pub fn parse_simple() {
    compiler_vs_json(Path::new("simple"));
}

#[test]
pub fn parse_branching() {
    let src_code = read_to_string("tests/test_progs/branching.ku").expect("Whoops, wrong path!");
    let tokens = lexer::lex(&src_code).expect("I expect this to lex correctly!");
    let parsed_ast = parser::parse(tokens).expect("I expect this to parse correctly!");

    assert!(false);
}
