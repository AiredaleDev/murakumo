use std::borrow::Cow;

use slab::Slab;
use slotmap::DefaultKey as NodeKey;
use smallvec::SmallVec;
use ustr::{Ustr, UstrMap};

use crate::{
    AST, ASTNodeType, Type,
    ast::{ExprOp, Lit, StmtKind},
    typing::TypeEnv,
};
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
// ...where `InstResult(i)` contains an index `i` into `inst_slab`
// So we don't need to conceive of a "location" to store the result of something like
// `func(...) * 3;` -- it doesn't have one, so it's trivially dead code up to the call to func.
//
// Clang and GCC are willing to inline globals that are exclusively READ and marked as
// `static` with -O3.
// Otherwise, they must be accessed via ld/st instructions, because how else will the effects
// of writing to a global occur elsewhere in the program?
//
// This makes me wonder if I should mark things as "pub(crate)" wherever possible to help
// the compiler inline more aggressively.

pub struct MachModule {
    funcs: UstrMap<MachFunc>,
    entry_point: Option<usize>,
}

// Just like MLIR, "func" defintions don't introduce arguments, their entry blocks do.
pub struct MachFunc {
    basic_blocks: Slab<MachBasicBlock>,
    control_flow_edges: Vec<usize>,
}

pub struct MachBasicBlock {
    args: Vec<Type>,
    insts: Slab<MachInst>,
}

pub struct MachInst {
    op: MachOp,
    args: SmallVec<[MachOperand; 3]>,
}

// Since our language does not have control flow constructs yet, every program can be captured
// by these 11 instructions. Wow!
#[derive(Clone, Copy, Debug)]
pub enum MachOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    FAdd,
    FSub,
    FMul,
    FDiv,
    Call,
    Ret,
}

impl MachOp {
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Call | Self::Ret)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MachOperand {
    /// Akin to what LLVM calls "void". My PL nerd background requires that I refuse to call 
    /// anything you can construct "void" -- the whole point of "void" as a type is that you 
    /// can't construct a value that belongs to it, but here's LLVM writing `ret void`.
    Unit,
    // Breaking up `ast::Lit` into its variants because strings need to be handled
    // distinctly, plus we don't support pointers and slices yet.
    IntLit(i64),
    FloatLit(f64),
    Label(Ustr),
    /// Think of `%0` in `%1 = add %0, 3`
    /// `%1 = ...` is represented implicitly.
    InstResult(usize),
    /// Think of `%a` in `^bbx(%a: i64):`
    BlockArg(usize),
    /// Used to index into a `MachFunc`'s list of BBs.
    /// We won't construct this until we introduce branching and looping.
    BasicBlock(usize),
}

struct ASTLoweringContext {
    module: MachModule,
    ssa_map: UstrMap<MachOperand>,
    scope_stack: Vec<NodeKey>,
}

pub fn lower_ast(ast: AST, type_env: TypeEnv) -> MachModule {
    // let mut ssa_map = UstrMap::default();

    todo!()
}

// Block nodes in the AST do not necessarily correspond with basic blocks in the IR.
// Subscopes, for example, are just chained together into one big BB*
//
// We're going to need to
//
// *what about code like this?
// {
//    if x == 1 {  }
// }
//
// Just a normal case, actually. No special logic needed.
fn lower_block(
    ast: &AST,
    block_node: NodeKey,
    type_env: &TypeEnv,
    insts: &mut Slab<MachInst>,
    ssa_map: &mut UstrMap<MachOperand>,
) {
    let stmts = &ast.nodes[block_node].args;
    // Preemptively delete the trailing unit.
    let mut node_count = stmts.len();
    if let Some(nk) = stmts.last() {
        if matches!(ast.nodes[*nk].ty, ASTNodeType::Literal(Lit::Unit)) {
            node_count -= 1;
        }
    }

    for stmt in &stmts[..node_count] {
        lower_stmt_node(ast, *stmt, block_node, type_env, insts, ssa_map);
    }
}

// We need to also pass forward a reference to the current func.
// What if there is a func declared within a func? This is a case I want to support.

fn lower_stmt_node(
    ast: &AST,
    stmt: NodeKey,
    curr_block: NodeKey,
    type_env: &TypeEnv,
    insts: &mut Slab<MachInst>,
    ssa_map: &mut UstrMap<MachOperand>,
) {
    let ASTNodeType::Stmt(stmt_kind) = ast.nodes[stmt].ty else {
        unreachable!("Compiler writer messed up: `lower_stmt_node`");
    };

    let expr = ast.nodes[stmt].args[match stmt_kind {
        StmtKind::Assign | StmtKind::Define => 1,
        StmtKind::Pure => 0,
    }];

    let (tail_inst, _) = lower_expr_node(ast, expr, curr_block, type_env, insts, ssa_map);

    match stmt_kind {
        StmtKind::Assign | StmtKind::Define => {
            let lhs_key = ast.nodes[stmt].args[0];
            let name = match &ast.nodes[lhs_key].ty {
                ASTNodeType::Ident(vn) => *vn,
                ASTNodeType::Expr(ExprOp::Decl) => {
                    let ASTNodeType::Ident(vn) = ast.child(lhs_key, 0).ty else {
                        unreachable!(
                            "Non-ident as first child of decl -- the front-end should've caught this."
                        )
                    };
                    vn
                }
                n => unreachable!(
                    "{n:?} as first child of assign_lhs -- the front-end should've caught this."
                ),
            };

            ssa_map.insert(name, tail_inst);
        }
        StmtKind::Pure => {}
    }
}

// One expr can become many instructions.
// Whenever we have a block (e.g. 1 + { x := 2; x + 2 };)
// we want the contents of the block to always precede the higher nodes...
// ...which is perfectly consistent with the behavior of
fn lower_expr_node<'env>(
    ast: &AST,
    expr: NodeKey,
    curr_block: NodeKey,
    type_env: &'env TypeEnv,
    insts: &mut Slab<MachInst>,
    ssa_map: &mut UstrMap<MachOperand>,
) -> (MachOperand, Cow<'env, Type>) {
    match &ast.nodes[expr].ty {
        ASTNodeType::Literal(lit) => lower_lit(lit),
        ASTNodeType::Ident(name) => (ssa_map[name], Cow::Borrowed(&type_env[&curr_block][name])),
        ASTNodeType::Expr(op) => lower_op_node(ast, curr_block, *op, &ast.nodes[expr].args, type_env, insts, ssa_map),
        e => unreachable!("Bad AST or bad lowering algo: {e:?}"),
    }
}

fn lower_op_node<'env>(
    ast: &AST,
    curr_block: NodeKey,
    node_op: ExprOp,
    node_args: &[NodeKey],
    type_env: &'env TypeEnv,
    insts: &mut Slab<MachInst>,
    ssa_map: &mut UstrMap<MachOperand>,
) -> (MachOperand, Cow<'env, Type>) {
    let (args, types): (SmallVec<[_; 3]>, SmallVec<[_; 3]>) = node_args
        .iter()
        .map(|a| lower_expr_node(ast, *a, curr_block, type_env, insts, ssa_map))
        .unzip();

    let (op, op_ret_ty) = match node_op {
        // NOTE: This clone is free (we're going to return something borrowed or owned and trivial to copy)
        op if op.is_arithmetic() => lower_arith_op(node_op, types[0].clone()),
        // Right now "not arithmetic" means call.
        // ExprOp::Call => ,
        expr => unreachable!("Compiler writer messed up: {expr:?}"),
    };

    let root_inst = insts.insert(MachInst { op, args });
    (MachOperand::InstResult(root_inst), op_ret_ty)
}

fn lower_lit<'l>(lit: &Lit) -> (MachOperand, Cow<'l, Type>) {
    // Whether or not we should inline the literal.
    // I just realized how silly it is to have a frontend that supports strings
    // but planning on adding support for pointers later lol.
    // So...
    // TODO: Add pointer and slice support lmfao. A "string" in this language is akin to Java's
    // String or Rust's &str: immutable, UTF-8, pointer + length.
    // (Rust lets you construct &mut str, I don't permit this!)
    // If I bother to make a standard library, there will be a StringBuilder (or what Rust calls
    // String and C++ calls std::string)
    let mach_op = match *lit {
        Lit::Unit => MachOperand::Unit,
        Lit::Int(i) => MachOperand::IntLit(i),
        Lit::Float(f) => MachOperand::FloatLit(f),
        Lit::String(_) => todo!(
            "String literals have yet to be implemented! \
            The language and the IR presently don't know what a pointer is."
        ),
    };

    (mach_op, Cow::Owned(lit.ty()))
}

fn lower_arith_op(op: ExprOp, operand_ty: Cow<'_, Type>) -> (MachOp, Cow<'_, Type>) {
    if operand_ty.is_int() {
        let mach_op = match op {
            ExprOp::Add => MachOp::Add, 
            ExprOp::Negate | ExprOp::Subtract => MachOp::Sub,
            ExprOp::Multiply => MachOp::Mul,
            ExprOp::Divide => MachOp::Div,
            ExprOp::Mod => MachOp::Rem,
            _ => unreachable!(),
        };

        (mach_op, operand_ty)
    } else if operand_ty.is_float() {
        let mach_op = match op {
            ExprOp::Add => MachOp::FAdd,
            ExprOp::Subtract => MachOp::FSub,
            ExprOp::Multiply => MachOp::FMul,
            ExprOp::Divide => MachOp::FDiv,
            _ => unreachable!(),
        };

        (mach_op, operand_ty)
    } else {
        unreachable!("This should only be called on arithmetic ops (I'm here for debugging purposes)")
    }
}
