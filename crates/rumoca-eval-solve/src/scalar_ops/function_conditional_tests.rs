//! A function conditional left in a row expands to scalar selects that keep
//! the evaluator's value: the first arm whose condition holds, else the
//! fallback.

use super::*;
use rumoca_ir_solve::{CompareOp, FunctionConditionalProgram, ScalarProgramBlock, UnaryOp};
use std::sync::Arc;

fn span() -> rumoca_core::Span {
    rumoca_core::Span::from_offsets(
        rumoca_core::SourceId::from_source_name("function_conditional.mo"),
        0,
        1,
    )
}

fn run(ops: &[LinearOp], parameters: &[f64]) -> f64 {
    let block = ScalarProgramBlock::with_output_indices(vec![ops.to_vec()], vec![span()], vec![0])
        .expect("a checked scalar program");
    let mut out = [f64::NAN];
    crate::eval_scalar_program_block(&block, &[], parameters, 0.0, None, &mut out).unwrap();
    out[0]
}

/// `p[0] > threshold`, read straight from the parameters as Solve lowering
/// leaves it for `f(g(p))` with an `if` in `g`.
fn above(threshold: f64) -> Vec<LinearOp> {
    vec![
        LinearOp::LoadP { dst: 0, index: 0 },
        LinearOp::Const {
            dst: 1,
            value: threshold,
        },
        LinearOp::Compare {
            dst: 2,
            op: CompareOp::Gt,
            lhs: 0,
            rhs: 1,
        },
        LinearOp::StoreOutput { src: 2 },
    ]
}

fn constant(value: f64) -> Vec<LinearOp> {
    vec![
        LinearOp::Const { dst: 0, value },
        LinearOp::StoreOutput { src: 0 },
    ]
}

/// `capture * p[0]`, so the region reads both the row and its capture.
fn scaled_parameter() -> Vec<LinearOp> {
    vec![
        LinearOp::LoadFunctionConditionalCapture { dst: 0, index: 0 },
        LinearOp::LoadP { dst: 1, index: 0 },
        LinearOp::Binary {
            dst: 2,
            op: BinaryOp::Mul,
            lhs: 0,
            rhs: 1,
        },
        LinearOp::StoreOutputRange {
            start: 2,
            count: 1,
            stride: 1,
        },
    ]
}

fn negated_parameter() -> Vec<LinearOp> {
    vec![
        LinearOp::LoadP { dst: 0, index: 0 },
        LinearOp::Unary {
            dst: 1,
            op: UnaryOp::Neg,
            arg: 0,
        },
        LinearOp::StoreOutputRange {
            start: 1,
            count: 1,
            stride: 1,
        },
    ]
}

/// `2 * (if p > 1 then 10 elseif p > 0 then capture * p else -p) + 1`, with
/// `capture = 3`: the conditional feeds arithmetic after it, as a retained
/// call's argument does.
fn row(program: FunctionConditionalProgram) -> Vec<LinearOp> {
    vec![
        LinearOp::Const { dst: 0, value: 3.0 },
        LinearOp::FunctionConditional {
            dst_start: 1,
            capture_start: 0,
            program: Arc::new(program),
        },
        LinearOp::Const { dst: 2, value: 2.0 },
        LinearOp::Binary {
            dst: 3,
            op: BinaryOp::Mul,
            lhs: 1,
            rhs: 2,
        },
        LinearOp::Const { dst: 4, value: 1.0 },
        LinearOp::Binary {
            dst: 5,
            op: BinaryOp::Add,
            lhs: 3,
            rhs: 4,
        },
        LinearOp::StoreOutput { src: 5 },
    ]
}

fn ordered_arms() -> FunctionConditionalProgram {
    FunctionConditionalProgram::checked(
        1,
        [1],
        [
            (above(1.0), constant(10.0)),
            (above(0.0), scaled_parameter()),
        ],
        negated_parameter(),
    )
    .expect("a checked function conditional")
}

#[test]
fn expanded_conditional_matches_the_conditional_op_and_leaves_none_behind() {
    let canonical = row(ordered_arms());
    let lowered = lower_scalar_ops(&canonical, None).expect("the conditional expands");
    assert!(
        lowered
            .iter()
            .all(|op| !matches!(op, LinearOp::FunctionConditional { .. })),
        "{lowered:?}"
    );
    // The first arm holds for 4 even though the second does too.
    for (parameter, expected) in [(4.0, 21.0), (0.5, 4.0), (-2.0, 5.0)] {
        assert_eq!(run(&canonical, &[parameter]), expected);
        assert_eq!(run(&lowered, &[parameter]), expected);
    }
}

#[test]
fn a_conditional_nested_in_a_region_expands_too() {
    let inner = FunctionConditionalProgram::checked(
        1,
        [1],
        [(above(0.0), scaled_parameter())],
        negated_parameter(),
    )
    .expect("a checked inner conditional");
    let fallback = vec![
        LinearOp::LoadFunctionConditionalCapture { dst: 0, index: 0 },
        LinearOp::FunctionConditional {
            dst_start: 1,
            capture_start: 0,
            program: Arc::new(inner),
        },
        LinearOp::StoreOutput { src: 1 },
    ];
    let outer =
        FunctionConditionalProgram::checked(1, [1], [(above(1.0), constant(10.0))], fallback)
            .expect("a checked outer conditional");
    let canonical = row(outer);
    let lowered = lower_scalar_ops(&canonical, None).expect("nested conditionals expand");
    for (parameter, expected) in [(4.0, 21.0), (0.5, 4.0), (-2.0, 5.0)] {
        assert_eq!(run(&canonical, &[parameter]), expected);
        assert_eq!(run(&lowered, &[parameter]), expected);
    }
}
