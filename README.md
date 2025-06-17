# Murakumo -- A Cyclone-Inspired, Region-Based Systems Language

I conceived of this language both as a fun exercise in looking at this [old paper](https://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf) and to make a Youtube tutorial series on compiler construction.

I want to focus on optimization and code-generation, only mentioning enough type theory to make programming in the language nicer. If one wants to learn LLVM, read [Kaleidoscope](https://llvm.org/docs/tutorial/) as a starting point.

In terms of video/blog post content, I think the most valuable thing would be to present certain *algorithms* as opposed to code, as well as my thought process for tricky parts. I implement some parts, like the lexer, parser, and AST in an unorthodox way (at least for Rust) because I _think_ it will yield tigher bounds on performance. The lexer is guaranteed not to allocate anything other than the final `Vec` on the happy path and only allocate the string for the error on the failing path. The AST is built using slotmaps to get the compactness of an arena allocator with the memory re-use of a pool/malloc (AST rewrites often involve replacing nodes wholesale).

### Progress:

- [ ] Front-end:
  - [ ] Lexing (DONE excluding control-flow constructs)
  - [ ] Parsing (DONE excluding control-flow constructs)
  - [ ] Type inference (Simple types)
    - [ ] Region parameters (honestly might just ditch this part of the project)
  - [ ] Nice error-reporting
  - [ ] REPL
- [ ] Middle-"end":
  - [ ] IR
  - [ ] Optimization passes -- need to decide which ones beyond those listed below:
- [ ] Back-end (just x86 for now)
  - [ ] Instruction Selection
  - [ ] Instruction Scheduling
  - [ ] Register Allocation

### Language Features

- C.
- ...with region-based memory management
- ...HM type inference (maybe not for the region parameters though)
- ...and first-class linear algebra support.
- So it's really just Fortran with region-based memory-management and some second-order logic in the type system.
- I kind of hate the way arena allocators in Rust work (or maybe I'm getting skill-issued by them, they don't do that in C) so RBMM it is.
- (scope-creep kickstarter stretch-goal) OpenMP-style parallel-for -- at this point you might as well target MLIR instead of rolling your own backend.
- (ROUND TWO) PTX/ROCm/OpenCL generation with an `acc for` construct -- If we want to tie this in with GPUs then we'll have to tag regions with a "device" and make the runtime a little heaiver. Oh well, the spirit of this project is "anything that is interesting to compile."

### Compiler Goals

- Do something more interesting than just the textbook compiler for some C or Fortran-like language (hence region-based memory management even though it's only as good of an idea as programming in Rust xd)
  - Also towards this end, introduce polyhedral loop optimization which motivated the creation of MLIR
- Be classical (cover all the great optimizations that have been with us since the 70s) but give curious source-code reader a taste of something a little more interesting.
- Highlight some good Rust crates that make programming in it and achieving the performance a non-GC language should have much easier (e.g. slotmap)
- Make a suite of solid benchmark programs so we can measure if our optmizations are actually doing something.
- Be pleasant to use. Efficient and nice error-reporting are rarely mentioned in, let alone the focus of, compiler textbooks. I understand why on a surface level -- it's neither glamorous nor theoretical. The end user, however, is why a programming language exists in the first place.

### The IR

- SSA CFG -- We're iterating on tradition here, not throwing out the baby w/ the bath water.
- Might even copy MLIR's homework and have basic-blocks "take parameters" instead of having phi-nodes.

### Target machine (since it affects backend decisions)

- A home computer. Most simple optimizations happen to make the program smaller, but if we're feeling MOTIVATED we can do a little inlining.
- Targeting x86 will be annoying though, since you get very few ISA registers... bruh
- Different register allocation strategies might be good, are there any others besides graph-coloring/linear scan?

### Overview of Compiler:

```
                 |--(opt)-||----(opt)---||----(opt)--||---------(i-sched passes)-------|
Source ---> AST -|-> TIR -||-> LinIaR --||-> MachIR -||-> ASM with infinite registers -|--(reg alloc)--> Machine Code
```
- TIR: Typed AST -- We've inferred all types, which includes region parameters. More for correctness.
- LinIaR: SSA CFG+Regions -- Thing inspired by MLIR which has "ops with regions" -- we have a notion of a "for loop" in the IR, ladies and gentlemen.
  - For PTX: LinIaR could also denote which loops are "GPU-able" or track GPU-specific details.
  - We will have to modify the backend to generate GPU code too though, so we'll need a "device" tag on each function all the way down.
- MachIR: SSA CFG -- we've removed all the "for-loop" ops at this point, tried-and-true optimizations start here.

