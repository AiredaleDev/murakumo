# Murakumo -- A Cyclone-Inspired, Region-Based Systems Language for the curious, motivated undergraduate/self-taught programmer.

I might want to change the name of this, since the first result is a warship even though its literal meaning is "gathering clouds," which fits the weather theme.

I conceived of this language both as a fun exercise in looking at this old paper: <link> and to make a Youtube tutorial series on compiler construction.

Everything except the parser and some useful data structures are hand-written. I want to teach a curious viewer the basics of compiler construction, with a focus on optimization and code-generation, only mentioning enough type theory to make programming in the language nicer. If they want to learn LLVM, they can read Kaleidoscope as a starting point.

For the sake of brevity, I'll work out this whole compiler, then present how to build it piece-by-piece. I understand that the reality of implementing something is debugging it yourself for hours, but 

### Language Features

- C.
- ...with region-based memory management
- ...HM type inference (maybe not for the region parameters though)
- ...and first-class linear algebra support.
- So it's really just Fortran with region-based memory-management and some second-order logic in the type system.
- I kind of hate the way arena allocators in Rust work (or maybe I'm getting skill-issued by them, they don't do that in C) so RBMM it is.
- (scope-creep kickstarter stretch-goal) OpenMP-style parallel-for -- at this point you might as well target MLIR instead of rolling your own backend.

### Pedagogical Goals

- Do something more interesting than just the textbook compiler for some C or Fortran-like language (hence region-based memory management even though it's only as good of an idea as programming in Rust xd)
  - Also towards this end, introduce polyhedral loop optimization which motivated the creation of MLIR
- Be classical (cover all the great optimizations that have been with us since the 70s) but give the viewer a taste of something a little more interesting.
- Highlight some good Rust crates that make programming in it and achieving the performance a non-GC language should have much easier (e.g. slotmap) 
- Make a suite of solid benchmark programs so we can measure if our optmizations are actually doing something.

### The IR

- SSA CFG -- We're iterating on tradition here, not throwing out the baby w/ the bath water.
- Might even copy LLVM's homework and have basic-blocks "take parameters" instead of having phi-nodes.

### Target machine (since it affects backend decisions)

- A home computer. Most simple optimizations happen to make the program smaller, but if we're feeling MOTIVATED we can do a little inlining.
- Targeting x86 will be annoying though, since you get 4 ISA registers... bruh
- Different register allocation strategies might be good.

### Overview of Compiler:

```
                 |--(opt)-||----(opt)---||----(opt)--|
Source ---> AST -|-> TIR -||-> LinIaR --||-> MachIR -|-> ASM with infinite registers ---(reg alloc)--> Machine Code


```
TIR: Typed AST -- We've inferred all types, which includes region parameters. More for correctness.
LinIaR: SSA CFG+Regions -- Thing inspired by MLIR which has "ops with regions" -- we have a notion of a "for loop" in the IR, ladies and gentlemen.
MachIR: SSA CFG -- we've removed all the "for-loop" ops at this point, tried-and-true optimizations start here.

