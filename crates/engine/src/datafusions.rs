use std::{lazy::SyncLazy, sync::Arc, sync::Mutex};

use arrow::{
    array::{
        ArrayData, ArrayRef, Date16Array, DecimalArray, FixedSizeBinaryArray,
        Float32Array, Float64Array, GenericStringArray, Int16Array, Int32Array,
        Int64Array, Int8Array, Timestamp32Array, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    },
    buffer::Buffer,
    datatypes::{DataType, Field, Schema},
    ffi::FFI_ArrowArray,
    record_batch::RecordBatch,
};
use base::contract;
use datafusion::{
    datasource::MemTable, error::Result, logical_plan::LogicalPlan,
    prelude::ExecutionContext,
};
use lang::parse::{parse_where, TablesContext};
use meta::{
    store::{
        parts::{CoPaInfo, PartStore},
        sys::MetaStore,
    },
    types::{BqlType, Id},
};
use tokio::runtime::{self, Runtime};

use crate::{
    errs::{EngineError, EngineResult},
    types::QueryState,
};
use meta::types::IntoRef;

static TOKIO_RT: SyncLazy<Runtime> =
    SyncLazy::new(|| runtime::Builder::new_multi_thread().build().unwrap());

fn btype_to_arrow_type(typ: BqlType) -> EngineResult<DataType> {
    match typ {
        BqlType::UInt(bits) if bits == 8 => Ok(DataType::UInt8),
        BqlType::UInt(bits) if bits == 16 => Ok(DataType::UInt16),
        BqlType::UInt(bits) if bits == 32 => Ok(DataType::UInt32),
        BqlType::UInt(bits) if bits == 64 => Ok(DataType::UInt64),
        BqlType::Int(bits) if bits == 8 => Ok(DataType::Int8),
        BqlType::Int(bits) if bits == 16 => Ok(DataType::Int16),
        BqlType::Int(bits) if bits == 32 => Ok(DataType::Int32),
        BqlType::Int(bits) if bits == 64 => Ok(DataType::Int64),
        BqlType::Float(bits) if bits == 16 => Ok(DataType::Float16),
        BqlType::Float(bits) if bits == 32 => Ok(DataType::Float32),
        BqlType::Float(bits) if bits == 64 => Ok(DataType::Float64),
        BqlType::DateTime => Ok(DataType::Timestamp32(None)),
        BqlType::DateTimeTz(tz) => Ok(DataType::Timestamp32(Some(tz))),
        BqlType::Date => Ok(DataType::Date16),
        BqlType::Decimal(p, s) => Ok(DataType::Decimal(p as usize, s as usize)),
        BqlType::String => Ok(DataType::LargeUtf8),
        BqlType::LowCardinalityString => Ok(DataType::UInt32),
        BqlType::LowCardinalityTinyText => Ok(DataType::UInt8),
        BqlType::FixedString(len) => Ok(DataType::FixedSizeBinary(len as i32)),
        _ => Err(EngineError::UnsupportedBqlType),
    }
}

pub(crate) fn run(
    ms: &MetaStore,
    ps: &PartStore,
    current_db: &str,
    raw_query: &str,
    is_explain: bool,
    _query_id: &str,
    tctx: TablesContext,
    qs: &mut QueryState,
) -> EngineResult<Vec<RecordBatch>> {
    // let t = Instant::now();
    let mut ctx = ExecutionContext::new();
    let tabs = tctx.tabs;
    let cols = tctx.cols;
    for tab in tabs {
        let qn1 = [current_db, &tab].join(".");
        let qtn = if tab.contains('.') { tab } else { &qn1 };
        let tid = ms.tid_by_qname(&qtn).ok_or(EngineError::TableNotExist)?;
        //parts pruning
        let ptk_range = match ms.get_table_info_partition_cols(tid)? {
            Some(iv) => {
                let ptc = unsafe { std::str::from_utf8_unchecked(&*iv) };
                let pc = if ptc.ends_with(",") {
                    &ptc[..ptc.len() - 1]
                } else {
                    &ptc
                };
                log::debug!("--- pc: {:?}", pc);
                if (&cols).contains(pc) {
                    match ms.get_table_info_partition_keys_expr(tid)? {
                        Some(iv) => {
                            let ptk_expr = unsafe { std::str::from_utf8_unchecked(&*iv) };
                            log::debug!("ptk_expr: {:?}", ptk_expr);
                            parse_where(tctx.where_str, ptk_expr)?
                        }
                        None => {
                            vec![0..=u64::MAX]
                        }
                    }
                } else {
                    vec![0..=u64::MAX]
                }
            }
            _ => vec![0..=u64::MAX],
        };
        log::debug!("ptk_range: {:?}", ptk_range);
        if ptk_range.len() == 0 {
            return Ok(Vec::<RecordBatch>::new());
        }

        // *cid, ci.data_type
        let mut cis = Vec::new();
        let mut fields = Vec::new();
        if cols.len() != 0 && !tctx.has_select_all {
            for cn in cols.iter() {
                let qcn = if cn.contains('.') {
                    // ms.cid_by_qname(&cn).ok_or(EngineError::ColumnNotExist)?
                    //FIXME for t.c, not work for db.t.c
                    [current_db, cn].join(".")
                } else {
                    [qtn, cn].join(".")
                };
                if qcn.contains(qtn) {
                    if let Some(cid) = ms.cid_by_qname(&qcn) {
                        if let Some(ci) = ms.get_column_info(cid)? {
                            cis.push((cid, ci.data_type));
                            fields.push(Field::new(
                                cn,
                                btype_to_arrow_type(ci.data_type)?,
                                false,
                            ));
                        } else {
                            return Err(EngineError::ColumnInfoNotExist);
                        }
                    }
                }
            }
        } else {
            if tctx.has_count_all {
                log::debug!("current_db: {}, cn: {}", current_db, &tab);
                let cs = ms.get_columns_by_qtn(qtn)?;
                let (cn, cid, ci) = &cs[0];
                cis.push((*cid, ci.data_type));
                fields.push(Field::new(
                    cn.as_str(),
                    btype_to_arrow_type(ci.data_type)?,
                    false,
                ));
            } else if tctx.has_select_all {
                let cs = ms.get_columns_by_qtn(qtn)?;
                for (cn, cid, ci) in cs {
                    cis.push((cid, ci.data_type));
                    fields.push(Field::new(
                        cn.as_str(),
                        btype_to_arrow_type(ci.data_type)?,
                        false,
                    ));
                }
            } else {
                return Err(EngineError::UnsupportedQuery);
            }
        }
        //log::debug!("[df][Schema] - fields: {:?}", fields);
        let schema = Arc::new(Schema::new(fields));
        let copasss = &mut qs.copasss;
        let mut copass = Vec::new();
        ps.fill_copainfos_int_by_ptk_range(&mut copass, tid, &cis, ptk_range)?;
        if copass.len() > 0 {
            log::debug!(
                "got {} copas, with {} copa per copas for {}",
                copass.len(),
                copass[0].len(),
                tid,
            );
            setup_tables(tab, schema, &mut ctx, &cis, &copass, is_explain)?;
            copasss.push(copass);
        }
    }
    if qs.copasss.len() == 0 {
        let res: Vec<RecordBatch> = Vec::new();
        return Ok(res);
    }
    // log::info!("query setup runtime(ms): {}", t.elapsed().as_millis());

    //FIXME copa prunning

    let df = ctx.sql(raw_query)?;
    if is_explain {
        run_explain(&mut ctx, &df.to_logical_plan())
    } else {
        let res: Result<Vec<RecordBatch>> = TOKIO_RT.block_on(async move {
            let result = df.collect().await?;
            // arrow::util::pretty::print_batches(&result)?;
            Ok(result)
        });
        Ok(res?)
    }
}

// FIXME: workaround for explain output physical plan,
// datafusion raw explain output logical plan
fn run_explain(
    ctx: &mut ExecutionContext,
    plan: &LogicalPlan,
) -> EngineResult<Vec<RecordBatch>> {
    let state = ctx.state.lock().unwrap().clone();
    let ctx = ExecutionContext::from(Arc::new(Mutex::new(state)));
    // if we use BallistaContext, Optimization has been done
    let plan = ctx.optimize(plan)?;
    let plan = ctx.create_physical_plan(&plan)?;
    let displayable_plan = datafusion::physical_plan::displayable(plan.as_ref());
    let mut builder = arrow::array::LargeStringBuilder::new(1);
    builder.append_value(displayable_plan.indent().to_string().as_str())?;
    let explain_data = builder.finish();

    let mut explain_str = displayable_plan.indent().to_string();


    // let array = arrow::array::LargeStringArray::from(
    //     vec![explain_str]);

    let mut buf = Vec::with_capacity(explain_str.len());
    base::codec::encode_ascii_bytes_vec_short(explain_str.as_bytes(), &mut buf)
        .unwrap_or_default();

    let dummy = Arc::new(FFI_ArrowArray::empty());
    // let buf = unsafe {
    //     let ptr = std::ptr::NonNull::new(explain_str.as_mut_ptr())
    //         .ok_or(EngineError::UnwrapOptionError)?;
    //     Buffer::from_unowned(ptr, explain_str.len(), dummy)
    // };
    let buf = unsafe {
        let ptr = std::ptr::NonNull::new(explain_str.as_mut_ptr())
            .ok_or(EngineError::UnwrapOptionError)?;
        Buffer::from_unowned(ptr, explain_str.len(), dummy)
    };

    let data = ArrayData::builder(DataType::LargeUtf8)
        .len(2)
        .add_buffer(Buffer::from_slice_ref(&[1]))
        .add_buffer(buf)
        .build();
    let array = Arc::new(GenericStringArray::<i64>::from(data));

    let schema = Arc::new(Schema::new(vec![Field::new(
        "Explain Physical Plan",
        DataType::LargeUtf8,
        false,
    )]));
    let record_batch = RecordBatch::try_new(schema, vec![array])?;

    Ok(vec![record_batch])
}

fn setup_tables(
    tabname: &str,
    schema: Arc<Schema>,
    ctx: &mut ExecutionContext,
    cis: &Vec<(Id, BqlType)>,
    copass: &Vec<Vec<CoPaInfo>>,
    is_explain: bool,
) -> EngineResult<()> {
    contract!(copass.len() > 0, "copass should not be empty");
    contract!(
        copass.iter().all(|cps| cps.len() == copass[0].len()),
        "copas length should be equal"
    );
    let nc = copass.len();
    let np = copass[0].len();
    let mut batches = Vec::with_capacity(np);
    if !is_explain {
        fill_batches(schema.clone(), cis, copass, np, nc, &mut batches).unwrap();
    }
    // println!("batches.len: {}", batches.len());
    ctx.register_table(
        tabname,
        Arc::new(MemTable::try_new(schema.clone(), vec![batches])?),
    )?;

    Ok(())
}

fn fill_batches(
    schema: Arc<Schema>,
    cis: &Vec<(Id, BqlType)>,
    copass: &Vec<Vec<CoPaInfo>>,
    np: usize,
    nc: usize,
    batches: &mut Vec<RecordBatch>,
) -> EngineResult<()> {
    for i in 0..np {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(nc);
        for j in 0..nc {
            let copa = &copass[j];
            let cpi = &copa[i];
            let typ = btype_to_arrow_type(cis[j].1)?;
            let data = gen_arrow_arraydata(cpi, &typ)?;
            match typ {
                DataType::Int8 => {
                    cols.push(Arc::new(Int8Array::from(data)));
                }
                DataType::Int16 => {
                    cols.push(Arc::new(Int16Array::from(data)));
                }
                DataType::Int32 => {
                    cols.push(Arc::new(Int32Array::from(data)));
                }
                DataType::Int64 => {
                    cols.push(Arc::new(Int64Array::from(data)));
                }
                DataType::UInt8 => {
                    cols.push(Arc::new(UInt8Array::from(data)));
                }
                DataType::UInt16 => {
                    cols.push(Arc::new(UInt16Array::from(data)));
                }
                DataType::UInt32 => {
                    cols.push(Arc::new(UInt32Array::from(data)));
                }
                DataType::UInt64 => {
                    cols.push(Arc::new(UInt64Array::from(data)));
                }
                // TODO: arrow::array does not have Float16Array
                // DataType::Float16 => {
                //     cols.push(Arc::new(Float16Array::from(data)));
                // }
                DataType::Float32 => {
                    cols.push(Arc::new(Float32Array::from(data)));
                }
                DataType::Float64 => {
                    cols.push(Arc::new(Float64Array::from(data)));
                }
                DataType::Timestamp32(_) => {
                    cols.push(Arc::new(Timestamp32Array::from(data)));
                }
                DataType::Date16 => {
                    cols.push(Arc::new(Date16Array::from(data)));
                }
                DataType::Decimal(_, _) => {
                    cols.push(Arc::new(DecimalArray::from(data)));
                }
                DataType::LargeUtf8 => {
                    cols.push(Arc::new(GenericStringArray::<i64>::from(data)));
                }
                DataType::FixedSizeBinary(_) => {
                    cols.push(Arc::new(FixedSizeBinaryArray::from(data)));
                }
                // DataType::Null => {}
                // DataType::Boolean => {}
                // DataType::Timestamp(_, _) => {}
                // DataType::Date64 => {}
                // DataType::Time32(_) => {}
                // DataType::Time64(_) => {}
                // DataType::Duration(_) => {}
                // DataType::Interval(_) => {}
                // DataType::Binary => {}
                // DataType::LargeBinary => {}
                // DataType::Utf8 => {}
                // DataType::List(_) => {}
                // DataType::FixedSizeList(_, _) => {}
                // DataType::LargeList(_) => {}
                // DataType::Struct(_) => {}
                // DataType::Union(_) => {}
                // DataType::Dictionary(_, _) => {}
                _ => return Err(EngineError::UnsupportedBqlType),
            }
        }
        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        batches.push(batch);
    }
    Ok(())
}

fn gen_arrow_arraydata(cpi: &CoPaInfo, typ: &DataType) -> EngineResult<ArrayData> {
    let dummy = Arc::new(FFI_ArrowArray::empty());
    let buf = unsafe {
        let ptr = std::ptr::NonNull::new(cpi.addr as *mut u8)
            .ok_or(EngineError::UnwrapOptionError)?;
        Buffer::from_unowned(ptr, cpi.len_in_bytes, dummy)
    };
    let data = if matches!(typ, DataType::LargeUtf8) {
        let dummy_om = Arc::new(FFI_ArrowArray::empty());
        let buf_om = unsafe {
            let ptr = std::ptr::NonNull::new(cpi.addr_om as *mut u8)
                .ok_or(EngineError::UnwrapOptionError)?;
            Buffer::from_unowned(ptr, CoPaInfo::len_in_bytes_om(cpi.size), dummy_om)
        };
        ArrayData::builder(typ.clone())
            .len(cpi.size)
            .add_buffer(buf_om)
            .add_buffer(buf)
            .build()
    } else {
        ArrayData::builder(typ.clone())
            .len(cpi.size)
            .add_buffer(buf)
            .build()
    };
    Ok(data)
}

#[cfg(test)]
mod unit_tests {
    #[test]
    fn test() {}
}
