use clap::Parser;
use std::{fs::read_to_string, path::PathBuf};

mod error;
mod lexer;
mod parser;

pub use error::{DebugInfo, KumoError, KumoResult};
pub use lexer::Token;
use lexer::lex;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    files: Vec<PathBuf>,
}

fn compile(input: &str) -> KumoResult<()> {
    let tokens = lex(input)?;
    println!("{tokens:#?}");
    Ok(())
}

fn repl() {
    let mut editor = rustyline::DefaultEditor::new().expect("No REPL -> No (No Problems)");
    loop {
        let input = match editor.readline("kumo> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("{e}");
                break;
            }
        };

        if let Err(e) = compile(&input) {
            error::report(&input, e, None);
        }
    }

    println!("See ya.");
}

fn main() {
    // Whether we have a file or an interpreter, we ultimately feed
    // programs into the same code.
    //
    // As this language becomes more complete and we diverge from writing a calculator,
    // how to make a REPL will be less obvious. That said, I still think all languages should have
    // a REPL, even ones like this. It makes scratching out your ideas much easier.
    let cli = Args::parse();

    if cli.files.len() > 0 {
        for file in cli.files {
            let input = read_to_string(&file).expect("Where's that file?");
            if let Err(e) = compile(&input) {
                error::report(&input, e, Some(&file));
            }
        }
    } else {
        repl();
    }
}
