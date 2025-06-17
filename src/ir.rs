use slab::Slab;
use slotmap::DefaultKey as NodeKey;
use smallvec::SmallVec;

use crate::{typing::TypeEnv, ASTNodeType, Type, AST};
// SSA CFG, with some design inspirations from MLIR over LLVM where desired.
//
// Before I can lower an AST to this format, I need to define this format!
//
// A module has:
// 1. Defintions, some constant, some variable (place them into the appropriate region)
// 2. Optional entry point.
// The only difference between a func and a basic block is that a func can be `called`?
// Oh, no, the func can have multiple basic blocks in it.
//
// Recall that in SSA form, you construct a data dependence by referencing the op that creates the
// value needed by a particular op. To steal LLVM's notation, something like:
// %0 = add %arg0, %arg1
// %1 = mul %0, 3
// is implemented like this:
//
// inst_slab = [ add Arg(0) Arg(1) , mul InstResult(0) Lit(3) ]
//
// So we don't need to conceive of a "location" to store the result of something like
// `func(...) * 3;` -- it doesn't have one, so it's trivially dead code up to the call to func.

pub struct MachProgram {
    funcs: Slab<MachFunc>,
    entry_point: Option<usize>,
}

// Just like MLIR, "func" defintions don't introduce arguments, their entry blocks do.
pub struct MachFunc {
    basic_blocks: Slab<MachBasicBlock>,
    control_flow_edges: Vec<usize>,
    entry_block: usize,
}

pub struct MachBasicBlock {
    args: Vec<Type>,
    insts: Slab<MachInst>,
}

pub struct MachInst {
    op: MachOp,
    args: SmallVec<[usize; 3]>,
}

// Since our language does not have control flow constructs yet, every program can be captured
// by these 8 instructions. Wow!
#[derive(Clone, Copy, Debug)]
pub enum MachOp {
    Add,
    Mul,
    Div,
    FAdd,
    FMul,
    FDiv,
    Call,
    Ret,
}

pub enum MachOperand {
    /// Think of `%a` in `^bbx(%a: i64):`
    BlockArg(usize),
    /// Think of `%0` in `%1 = add %0, 3`
    /// `%1 = ...` is represented implicitly.
    InstResult(usize),
    // Breaking up `ast::Lit` into its variants because strings need to be handled
    // distinctly, plus we don't support pointers and slices yet.
    IntLit(i64),
    FloatLit(f64),
}

pub fn lower_ast(ast: AST, type_env: TypeEnv) -> MachProgram {
    

    todo!()
}

fn lower_stmt_node(ast: &AST, expr_node: NodeKey, insts: &mut Slab<MachInst>) {}

// One expr can become many instructions.
// Whenever we have a block (e.g. 1 + { x := 2; x + 2 };)
// we want the contents of the block to always precede the higher nodes...
// ...which is perfectly consistent with the behavior of 
fn lower_expr_node(ast: &AST, expr_node: NodeKey, insts: &mut Slab<MachInst>) {

}

fn lower_lit(ast: &AST, lit_node: NodeKey, insts: &mut Slab<MachInst>) -> MachOperand {
    // Whether or not we should inline the literal.
    // I just realized how silly it is to have a frontend that supports strings
    // but planning on adding support for pointers later lol.
    // So...
    // TODO: Add pointer and slice support lmfao. A "string" in this language is akin to Java's
    // String or Rust's &str: immutable, UTF-8, pointer + length.
    // (Rust lets you construct &mut str, I don't permit this!)
    // If I bother to make a standard library, there will be a StringBuilder (or what Rust calls
    // String and C++ calls std::string)
    let ASTNodeType::Literal(lit) = &ast.nodes[lit_node].ty else {
        panic!("This should always be a literal -- compiler writer's fault!");
    };

    use crate::ast::Lit;
    match *lit {
        Lit::Unit => MachOperand::IntLit(0),
        Lit::Int(i) => MachOperand::IntLit(i),
        Lit::Float(f) => MachOperand::FloatLit(f),
        Lit::String(_) => todo!("String literals have yet to be implemented! \
            The language and the IR presently don't know what a pointer is."),
    }
}
