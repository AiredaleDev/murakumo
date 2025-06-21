use murakumo::{
    ast::{ASTNode, ExprOp, Lit, StmtOp},
    *,
};
use slotmap::DefaultKey as NodeKey;
use smallvec::smallvec;
use std::fs::read_to_string;
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
        ty: ASTNodeType::Stmt(StmtOp::Pure),
        args: smallvec![expr],
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
            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Pure),
                args: smallvec![add_key],
            })
        };
        ast.nodes[root_key].args.push(fst);

        let snd = {
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, one_key, three_key);
            let add_key = mk_binop(&mut ast, ExprOp::Add, four_key, mult_key);
            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Pure),
                args: smallvec![add_key],
            })
        };
        ast.nodes[root_key].args.push(snd);

        let trd = {
            let add_key = mk_binop(&mut ast, ExprOp::Add, one_key, three_key);
            let group_key = mk_unop(&mut ast, ExprOp::Group, add_key);
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, four_key, group_key);
            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Pure),
                args: smallvec![mult_key],
            })
        };
        ast.nodes[root_key].args.push(trd);

        let frt = {
            let add_key = mk_binop(&mut ast, ExprOp::Add, four_key, one_key);
            let group_key = mk_unop(&mut ast, ExprOp::Group, add_key);
            let mult_key = mk_binop(&mut ast, ExprOp::Multiply, group_key, three_key);
            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Pure),
                args: smallvec![mult_key],
            })
        };

        ast.nodes[root_key].args.push(frt);

        ast.root = root_key;
        ast
    };

    assert_ast_eq(&parsed_ast, &expected_ast);
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
            let x_decl_stmt = mk_pure(&mut ast, x_decl);

            let y_name = mk_ident(&mut ast, "y".into());
            let y_type = mk_type(&mut ast, Type::Int(64));
            let y_decl = mk_binop(&mut ast, ExprOp::Decl, y_name, y_type);
            let y_decl_stmt = mk_pure(&mut ast, y_decl);

            let add_key = mk_binop(&mut ast, ExprOp::Add, x_name, y_name);
            let return_stmt = ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Pure),
                args: smallvec![add_key],
            });
            let func_body = mk_unop(&mut ast, ExprOp::Block, return_stmt);

            let args = mk_binop(&mut ast, ExprOp::Group, x_decl_stmt, y_decl_stmt);
            let ret_ty = mk_type(&mut ast, Type::Int(64));
            let func_lit = ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Expr(ExprOp::Func),
                args: smallvec![args, ret_ty, func_body],
            });

            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Define),
                args: smallvec![decl_key, func_lit],
            })
        };
        ast.nodes[root_key].args.push(add_fn_key);

        let main_key = {
            let name_key = mk_ident(&mut ast, "main".into());
            let hole_key = mk_type(&mut ast, Type::Hole);
            let decl_key = mk_binop(&mut ast, ExprOp::Decl, name_key, hole_key);

            let args = mk_lit(&mut ast, Lit::Unit);
            let ret_ty = mk_type(&mut ast, Type::Unit);
            let func_body = {
                let empty_block = ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Expr(ExprOp::Block)));
                let empty_stmt = mk_pure(&mut ast, empty_block);

                let zero_pls_name = mk_ident(&mut ast, "zero_me_pls".into());
                let zero_pls_ty = mk_type(&mut ast, Type::Int(64));
                let zero_decl = mk_binop(&mut ast, ExprOp::Decl, zero_pls_name, zero_pls_ty);
                let zero_stmt = mk_pure(&mut ast, zero_decl);

                let subscope = {
                    let scoped_name = mk_ident(&mut ast, "scoped".into());
                    let scoped_ty = mk_type(&mut ast, Type::Int(64));
                    let scoped_decl = mk_binop(&mut ast, ExprOp::Decl, scoped_name, scoped_ty);

                    let init_val = mk_lit(&mut ast, Lit::Int(12));

                    let scoped_stmt = ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Stmt(StmtOp::Assign),
                        args: smallvec![scoped_decl, init_val],
                    });

                    let trailing_unit = mk_lit(&mut ast, Lit::Unit);
                    let tail = mk_pure(&mut ast, trailing_unit);

                    let block_node = ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Expr(ExprOp::Block),
                        args: smallvec![scoped_stmt, tail],
                    });

                    mk_pure(&mut ast, block_node)
                };

                let three_two = mk_lit(&mut ast, Lit::Int(32));
                let six_four = mk_lit(&mut ast, Lit::Int(64));
                let func_name = mk_ident(&mut ast, "add".into());
                let first_summand = mk_binop(&mut ast, ExprOp::Multiply, three_two, three_two);
                let second_summand = mk_binop(&mut ast, ExprOp::Multiply, six_four, six_four);
                let call_add = ast.nodes.insert(ASTNode {
                    ty: ASTNodeType::Expr(ExprOp::Call),
                    args: smallvec![func_name, first_summand, second_summand],
                });

                let res_name = mk_ident(&mut ast, "result".into());
                let res_ty = mk_type(&mut ast, Type::Hole);
                let res_decl = mk_binop(&mut ast, ExprOp::Decl, res_name, res_ty);

                let add_call_stmt = ast.nodes.insert(ASTNode {
                    ty: ASTNodeType::Stmt(StmtOp::Assign),
                    args: smallvec![res_decl, call_add],
                });

                let thirty_e_2 = mk_lit(&mut ast, Lit::Float(30e2));
                let thirty_stmt = mk_pure(&mut ast, thirty_e_2);

                let trailing_unit = mk_lit(&mut ast, Lit::Unit);
                let tail = mk_pure(&mut ast, trailing_unit);

                ast.nodes.insert(ASTNode {
                    ty: ASTNodeType::Expr(ExprOp::Block),
                    args: smallvec![
                        empty_stmt,
                        zero_stmt,
                        subscope,
                        add_call_stmt,
                        thirty_stmt,
                        tail
                    ],
                })
            };

            let func_lit = ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Expr(ExprOp::Func),
                args: smallvec![args, ret_ty, func_body],
            });

            ast.nodes.insert(ASTNode {
                ty: ASTNodeType::Stmt(StmtOp::Define),
                args: smallvec![decl_key, func_lit],
            })
        };
        ast.nodes[root_key].args.push(main_key);

        ast.root = root_key;
        ast
    };

    assert_ast_eq(&parsed_ast, &expected_ast);
}
