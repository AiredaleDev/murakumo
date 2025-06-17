pub mod ast;
pub mod error;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod typing;

pub use ast::{AST, ASTNode, ASTNodeType};
pub use error::{DebugInfo, KumoError, KumoResult};
use lexer::lex;
pub use lexer::{Token, TokenType};
use parser::parse;
pub use typing::Type;
use typing::infer_types;

pub fn compile(input: &str) -> KumoResult<()> {
    let tokens = lex(input)?;
    // println!("{tokens:#?}");
    let mut ast = parse(tokens)?;
    println!("{ast}");
    println!("\nFOLD CONSTANTS:\n");
    ast::fold_constants(&mut ast);
    println!("{ast}");
    let type_env = infer_types(&ast);
    println!("\nINFER TYPES:\n");
    println!("{type_env:#?}");
    Ok(())
}
