use slotmap::{DefaultKey, SlotMap};

use crate::{Token, lexer::TokenType};

// Currently thinking about how to represent and store this.
// We want this to be amenable to quick updating, hence slotmap.
// It's a little ugly, but I think we should just have everything in one big enum.
// Slotmaps require uniform type.
pub struct AST<'src> {
    nodes: SlotMap<DefaultKey, ASTNode<'src>>,
    // We might want a scratch space for argument lists.
    // That sounds correct, time to depend on Bump or Slab or smth
    // (I'm able to manage memory manually, so why wouldn't I? I'd use some garbage-collected language 
    // if I didn't want that level of control.)
}

struct ASTNode<'src> {
    ty: ASTNodeType<'src>,
    lhs: DefaultKey,
    rhs: DefaultKey,
}

enum Type {
    // Fuck it, what does it cost to support arbitrary precision integers?
    // Copying Zig's homework -- if you don't pack a struct, they just grow to be byte-aligned.
    // Otherwise, they aren't. You don't need "bit fields," just declare the thing's type `u3` or
    // `u27` or whatever weird size your application or protocol demands.
    Int { size: usize, signed: bool },
    Float { size: usize },
    String,
    // `Comptime` types assume max(sizeof(their value), use)
    // This is done for convenience -- they're comptime-known, so they're
    // inline or looked-up immediately and constant-propagated.
    ComptimeInt,
    ComptimeFloat,
}

enum ASTNodeType<'src> {
    Ident { name: TokenType<'src>, ty: Type },
    Literal { ty: Type },
    Decl { ty: Option<Type> },
    Assign,
    Binop { op: Op },
}

enum Op {
    Plus,
    Minus,
    Multiply,
    Divide,
    Mod,
}

// Operator precedence: Tightest to loosest
// ((n DECL t?) ASSIGN) EXPR
// EXPR = PEMDAS | BLOCK
// This language's syntax is basically a bunch of infix operators
// anyway so Pratt parsing is perfect.

// Now we need a grammar.
// TODO: Remember Pratt Parsing.
pub fn parse(tokens: Box<[Token]>) -> AST {
    todo!()
}
