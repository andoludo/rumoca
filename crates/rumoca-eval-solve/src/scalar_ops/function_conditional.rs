//! Scalar expansion of a [`LinearOp::FunctionConditional`] left in a row.
//!
//! Solve lowering keeps a conditional function body as a `FunctionConditional`
//! op when the call is the argument of a retained call (`f(g(x))` with an `if`
//! in `g`). Its regions read `y`, `p` and `t` directly, so a scalar template
//! cannot render them as separate closed functions. Expanding every region into
//! the row's register file and choosing the result with `Select` yields plain
//! scalar ops with the evaluator's semantics: the first arm whose condition is
//! nonzero wins, else the fallback. Regions are pure, so evaluating every arm
//! eagerly does not change the selected value.

use rumoca_ir_solve::FunctionConditionalProgram;

use super::{LinearOp, Lowering, Reg};

impl Lowering<'_> {
    pub(super) fn function_conditional(
        &mut self,
        dst_start: Reg,
        capture_start: Reg,
        program: &FunctionConditionalProgram,
    ) -> Option<()> {
        let mut selected = self.region(
            &program.fallback,
            program.fallback_register_count,
            capture_start,
            program.result_count,
        )?;
        for arm in program.arms.iter().rev() {
            let [condition] = <[Reg; 1]>::try_from(self.region(
                &arm.condition,
                arm.condition_register_count,
                capture_start,
                1,
            )?)
            .ok()?;
            let taken = self.region(
                &arm.result,
                arm.result_register_count,
                capture_start,
                program.result_count,
            )?;
            selected = taken
                .into_iter()
                .zip(selected)
                .map(|(if_true, if_false)| {
                    let dst = self.fresh()?;
                    self.out.push(LinearOp::Select {
                        dst,
                        cond: condition,
                        if_true,
                        if_false,
                    });
                    Some(dst)
                })
                .collect::<Option<Vec<_>>>()?;
        }
        // Every region is read before the first destination is written, so a
        // destination overlapping a capture keeps the conditional's semantics.
        for (offset, src) in selected.into_iter().enumerate() {
            let dst = Self::reg(dst_start, offset)?;
            self.out.push(LinearOp::Move { dst, src });
        }
        Some(())
    }

    /// Copy one region into fresh registers; return the registers holding its
    /// stored outputs, in store order. Fails closed on any op it cannot place.
    fn region(
        &mut self,
        ops: &[LinearOp],
        register_count: usize,
        capture_start: Reg,
        output_count: usize,
    ) -> Option<Vec<Reg>> {
        let frame = RegionFrame {
            base: self.next,
            register_count,
            capture_start,
        };
        self.next = Self::reg(frame.base, register_count)?;
        let mut outputs = Vec::with_capacity(output_count);
        for op in ops {
            self.region_op(op, &frame, &mut outputs)?;
        }
        (outputs.len() == output_count).then_some(outputs)
    }

    fn region_op(
        &mut self,
        op: &LinearOp,
        frame: &RegionFrame,
        outputs: &mut Vec<Reg>,
    ) -> Option<()> {
        match op {
            LinearOp::StoreOutput { src } => outputs.push(self.snapshot(frame.local(*src)?)?),
            LinearOp::StoreOutputRange {
                start,
                count,
                stride,
            } => {
                for index in 0..*count {
                    let offset = Reg::try_from(index.checked_mul(*stride)?).ok()?;
                    outputs.push(self.snapshot(frame.local(start.checked_add(offset)?)?)?);
                }
            }
            LinearOp::LoadFunctionConditionalCapture { dst, index } => {
                self.out.push(LinearOp::Move {
                    dst: frame.local(*dst)?,
                    src: frame.capture(*index)?,
                });
            }
            LinearOp::LoadFunctionConditionalCaptureRange {
                dst_start,
                index_start,
                count,
            } => {
                for index in 0..*count {
                    self.out.push(LinearOp::Move {
                        dst: frame.local(Self::reg(*dst_start, index)?)?,
                        src: frame.capture(index_start.checked_add(index)?)?,
                    });
                }
            }
            LinearOp::FunctionConditional {
                dst_start,
                capture_start,
                program,
            } => self.function_conditional(
                frame.local(*dst_start)?,
                frame.local(*capture_start)?,
                program,
            )?,
            _ => self
                .out
                .push(relocated(op, &|register| frame.local(register))?),
        }
        Some(())
    }

    /// The value of `register` now: a region may assign a register again after
    /// storing it.
    fn snapshot(&mut self, register: Reg) -> Option<Reg> {
        let dst = self.fresh()?;
        self.out.push(LinearOp::Move { dst, src: register });
        Some(dst)
    }
}

/// A region's private register file inside the row, and the row registers
/// holding the conditional's captures.
struct RegionFrame {
    base: Reg,
    register_count: usize,
    capture_start: Reg,
}

impl RegionFrame {
    fn local(&self, register: Reg) -> Option<Reg> {
        if usize::try_from(register).ok()? >= self.register_count {
            return None;
        }
        self.base.checked_add(register)
    }

    fn capture(&self, index: usize) -> Option<Reg> {
        self.capture_start.checked_add(Reg::try_from(index).ok()?)
    }
}

/// `op` with every register moved into the region's frame; `None` for any op
/// outside the scalar subset a conditional region holds.
fn relocated(op: &LinearOp, local: &impl Fn(Reg) -> Option<Reg>) -> Option<LinearOp> {
    Some(match *op {
        LinearOp::Const { dst, value } => LinearOp::Const {
            dst: local(dst)?,
            value,
        },
        LinearOp::LoadTime { dst } => LinearOp::LoadTime { dst: local(dst)? },
        LinearOp::LoadY { dst, index } => LinearOp::LoadY {
            dst: local(dst)?,
            index,
        },
        LinearOp::LoadP { dst, index } => LinearOp::LoadP {
            dst: local(dst)?,
            index,
        },
        LinearOp::LoadIndexedP {
            dst,
            base,
            count,
            index,
        } => LinearOp::LoadIndexedP {
            dst: local(dst)?,
            base,
            count,
            index: local(index)?,
        },
        LinearOp::Move { dst, src } => LinearOp::Move {
            dst: local(dst)?,
            src: local(src)?,
        },
        LinearOp::Unary { dst, op, arg } => LinearOp::Unary {
            dst: local(dst)?,
            op,
            arg: local(arg)?,
        },
        LinearOp::Binary { dst, op, lhs, rhs } => LinearOp::Binary {
            dst: local(dst)?,
            op,
            lhs: local(lhs)?,
            rhs: local(rhs)?,
        },
        LinearOp::Compare { dst, op, lhs, rhs } => LinearOp::Compare {
            dst: local(dst)?,
            op,
            lhs: local(lhs)?,
            rhs: local(rhs)?,
        },
        LinearOp::Select {
            dst,
            cond,
            if_true,
            if_false,
        } => LinearOp::Select {
            dst: local(dst)?,
            cond: local(cond)?,
            if_true: local(if_true)?,
            if_false: local(if_false)?,
        },
        _ => return None,
    })
}

/// Scalars the expansion emits: one per region op, plus a capture move, a
/// select and a snapshot per result slot of every region.
pub(super) fn cost(program: &FunctionConditionalProgram) -> Option<usize> {
    let regions = program
        .arms
        .iter()
        .flat_map(|arm| [&arm.condition, &arm.result])
        .chain([&program.fallback]);
    let per_slot = program.arms.len().checked_add(1)?.checked_mul(3)?;
    regions
        .flatten()
        .try_fold(program.result_count.checked_mul(per_slot)?, |total, op| {
            total.checked_add(match op {
                LinearOp::FunctionConditional { program, .. } => cost(program)?,
                LinearOp::StoreOutputRange { count, .. }
                | LinearOp::LoadFunctionConditionalCaptureRange { count, .. } => {
                    count.checked_mul(2)?
                }
                _ => 2,
            })
        })
}
