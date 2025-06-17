use murakumo::{
    ast::{ASTNode, ExprOp, Lit, StmtOp},
    *,
};
use slotmap::DefaultKey as NodeKey;
use smallvec::smallvec;
use ustr::Ustr;
use std::fs::read_to_string;

// Compares tree structure (not tags used to build structure)
fn ast_deep_eq(ast1: &AST, ast2: &AST) -> bool {
    ast_node_eq(ast1, ast2, ast1.root, ast2.root)
}

fn ast_node_eq(ast1: &AST, ast2: &AST, node1: NodeKey, node2: NodeKey) -> bool {
    ast1.nodes[node1].ty == ast2.nodes[node2].ty
        && ast1.nodes[node1].args.len() == ast2.nodes[node2].args.len()
        && ast1.nodes[node1]
            .args
            .iter()
            .zip(ast2.nodes[node2].args.iter())
            .all(|(n1, n2)| ast_node_eq(ast1, ast2, *n1, *n2))
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

#[test]
pub fn parse_pemdas() {
    let src_code = read_to_string("tests/test_progs/eval_order.ku").expect("Whoops, wrong path!");
    let tokens = lexer::lex(&src_code).expect("I expect this to lex correctly!");
    let parsed_ast = parser::parse(tokens).expect("I expect this to parse correctly!");

    let expected_ast = {
        let mut ast = AST::default();
        let root_key = ast.nodes.insert(ASTNode::leaf(ASTNodeType::Module));

        let four_key = mk_lit(&mut ast, Lit::Int(4));
        let one_key = mk_lit(&mut ast, Lit::Int(1));
        let three_key = mk_lit(&mut ast, Lit::Int(3));

        let fst = {
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, four_key, one_key);
            let add_key = mk_binop(&mut ast, ExprOp::Add, mult_key, three_key);
            ast.nodes.insert(ASTNode { ty: ASTNodeType::Stmt(StmtOp::Pure), args: smallvec![add_key] })
        };
        ast.nodes[root_key].args.push(fst);

        let snd = {
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, one_key, three_key);
            let add_key = mk_binop(&mut ast, ExprOp::Add, four_key, mult_key);
            ast.nodes.insert(ASTNode { ty: ASTNodeType::Stmt(StmtOp::Pure), args: smallvec![add_key] })
        };
        ast.nodes[root_key].args.push(snd);

        let trd = {
            let add_key = mk_binop(&mut ast, ExprOp::Add, one_key, three_key);
            let group_key = mk_unop(&mut ast, ExprOp::Group, add_key);
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, four_key, group_key);
            ast.nodes.insert(ASTNode { ty: ASTNodeType::Stmt(StmtOp::Pure), args: smallvec![mult_key] })
        };
        ast.nodes[root_key].args.push(trd);

        let frt = {
            let add_key = mk_binop(&mut ast, ExprOp::Add, four_key, one_key);
            let group_key = mk_unop(&mut ast, ExprOp::Group, add_key);
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, group_key, three_key);
            ast.nodes.insert(ASTNode { ty: ASTNodeType::Stmt(StmtOp::Pure), args: smallvec![mult_key] })
        };

        ast.nodes[root_key].args.push(frt);

        ast.root = root_key;
        ast
    };

    assert!(ast_deep_eq(&parsed_ast, &expected_ast));
}

// The current largest integration test -- a program that uses all features
// of the language.
#[test]
pub fn parse_simple() {
    // We can then write some simple test progams.
    // How would we test type equality then? Oh, I guess that's easy, just do
    // node equality on each of the key nodes and then check their maps.

    let src_code = read_to_string("tests/test_progs/simple.ku").expect("Whoops, wrong path!");
    let tokens = lexer::lex(&src_code).expect("I expect this to lex correctly!");
    let parsed_ast = parser::parse(tokens).expect("I expect this to parse correctly!");

    // Now, let's compare the ASTs.
    // Building these is going to be a little tedious, but as far as I know there is no better way
    // to express the desired result to your testing suite.
    let expected_ast = {
        let mut ast = AST::default();
        let root_key = ast.nodes.insert(ASTNode::leaf(ASTNodeType::Module));

        let sus_key = {
            let name_key = mk_ident(&mut ast, "sus".into());
            let hole_key = mk_type(&mut ast, Type::Hole);
            let decl_key = mk_binop(&mut ast, ExprOp::Decl, name_key, hole_key);

            let ten_key = mk_lit(&mut ast, Lit::Int(10));
            let twenty_key = mk_lit(&mut ast, Lit::Int(20));
            let add_key = mk_binop(&mut ast, ExprOp::Add, ten_key, twenty_key);

            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Define),
                args: smallvec![decl_key, add_key],
            })
        };
        ast.nodes[root_key].args.push(sus_key);

        let bends_key = {
            let name_key = mk_ident(&mut ast, "bends_but_does_not_break".into());
            let type_key = mk_type(&mut ast, Type::Hole);
            let decl_key = mk_binop(&mut ast, ExprOp::Decl, name_key, type_key);

            let one_key = mk_lit(&mut ast, Lit::Float(1.0));
            let four_key = mk_lit(&mut ast, Lit::Int(4));
            let add_key = mk_binop(&mut ast, ExprOp::Add, one_key, four_key);

            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Define),
                args: smallvec![decl_key, add_key],
            })
        };
        ast.nodes[root_key].args.push(bends_key);

        let add_fn_key = {
            let name_key = mk_ident(&mut ast, "add".into());
            let hole_key = mk_type(&mut ast, Type::Hole);
            let decl_key = mk_binop(&mut ast, ExprOp::Decl, name_key, hole_key);

            let x_name = mk_ident(&mut ast, "x".into());
            let x_type = mk_type(&mut ast, Type::Int(64));
            let x_decl = mk_binop(&mut ast, ExprOp::Decl, x_name, x_type);

            let y_name = mk_ident(&mut ast, "y".into());
            let y_type = mk_type(&mut ast, Type::Int(64));
            let y_decl = mk_binop(&mut ast, ExprOp::Decl, y_name, y_type);

            

            let args = mk_binop(&mut ast, ExprOp::Group, x_decl, y_decl);
            let ret_ty = mk_type(&mut ast, Type::Int(64));
        };

        ast.root = root_key;
        ast
    };

    // This isn't super_descriptive -- I think I know how I could rewrite this tho.
    assert!(ast_deep_eq(&parsed_ast, &expected_ast));
}
