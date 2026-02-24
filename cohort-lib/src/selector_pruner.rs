use crate::bitmap_index::hash_key;
use crate::models::Selector;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;

/// Returns `true` if the file can definitely be skipped — no row can match `predicate`.
///
/// Conservative: returns `false` (don't skip) when uncertain.
pub fn can_skip_file(selectors: &[Selector], logic_op: &str, predicate: &Expr) -> bool {
    match logic_op.to_uppercase().as_str() {
        "AND" => {
            // With AND logic every selector constrains ALL rows.
            // If any selector contradicts the predicate, the file is irrelevant.
            predicate_impossible(selectors, predicate)
        }
        "OR" => {
            // With OR logic rows satisfy at least one selector.
            // We can't make strong per-column guarantees → be conservative.
            false
        }
        _ => false,
    }
}

// ── Recursive predicate evaluator ──────────────────────────────

fn predicate_impossible(selectors: &[Selector], expr: &Expr) -> bool {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
            Operator::And => {
                predicate_impossible(selectors, left)
                    || predicate_impossible(selectors, right)
            }
            Operator::Or => {
                predicate_impossible(selectors, left)
                    && predicate_impossible(selectors, right)
            }
            Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq
            | Operator::Gt | Operator::GtEq => {
                check_comparison(selectors, left, *op, right)
            }
            _ => false,
        },
        Expr::InList(il) => check_in_list(selectors, &il.expr, &il.list, il.negated),
        _ => false,
    }
}

// ── Leaf evaluators ────────────────────────────────────────────

fn check_comparison(
    selectors: &[Selector],
    left: &Expr,
    op: Operator,
    right: &Expr,
) -> bool {
    let (col_name, scalar, op) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (c.name(), v.clone(), op),
        (Expr::Literal(v, _), Expr::Column(c)) => (c.name(), v.clone(), flip(op)),
        _ => return false,
    };

    selectors
        .iter()
        .filter(|s| s.field == col_name && !s.is_exclusion)
        .any(|s| comparison_contradicts(s, op, &scalar))
}

fn check_in_list(
    selectors: &[Selector],
    expr: &Expr,
    list: &[Expr],
    negated: bool,
) -> bool {
    if negated {
        return false;
    }
    let col_name = match expr {
        Expr::Column(c) => c.name(),
        _ => return false,
    };
    let query_vals: Vec<ScalarValue> = list
        .iter()
        .filter_map(|e| match e {
            Expr::Literal(v, _) => Some(v.clone()),
            _ => None,
        })
        .collect();
    if query_vals.len() != list.len() {
        return false;
    }

    selectors
        .iter()
        .filter(|s| s.field == col_name && !s.is_exclusion)
        .any(|s| in_list_contradicts(s, &query_vals))
}

// ── Selector vs predicate contradiction checks ────────────────

fn comparison_contradicts(sel: &Selector, op: Operator, val: &ScalarValue) -> bool {
    match sel.operator.as_str() {
        "eq" | "in" => {
            let sv = extract_values(&sel.values);
            match op {
                Operator::Eq => {
                    let q = scalar_to_string(val);
                    !sv.iter().any(|v| *v == q)
                }
                Operator::NotEq => {
                    if sv.len() == 1 {
                        sv[0] == scalar_to_string(val)
                    } else {
                        false
                    }
                }
                Operator::Gt => all_leq(&sv, val),
                Operator::GtEq => all_lt(&sv, val),
                Operator::Lt => all_geq(&sv, val),
                Operator::LtEq => all_gt(&sv, val),
                _ => false,
            }
        }
        "between" => {
            let (lo, hi) = extract_bounds(&sel.values);
            match (lo, hi, scalar_to_f64(val)) {
                (Some(lo), Some(hi), Some(q)) => match op {
                    Operator::Eq => q < lo || q > hi,
                    Operator::Gt => hi <= q,
                    Operator::GtEq => hi < q,
                    Operator::Lt => lo >= q,
                    Operator::LtEq => lo > q,
                    _ => false,
                },
                _ => false,
            }
        }
        "gt" | "gte" | "lt" | "lte" => {
            match (extract_single_num(&sel.values), scalar_to_f64(val)) {
                (Some(sv), Some(qv)) => range_contradicts(sel.operator.as_str(), sv, op, qv),
                _ => false,
            }
        }
        _ => false,
    }
}

fn in_list_contradicts(sel: &Selector, query_vals: &[ScalarValue]) -> bool {
    match sel.operator.as_str() {
        "eq" | "in" => {
            let sv = extract_values(&sel.values);
            let qs: Vec<String> = query_vals.iter().map(scalar_to_string).collect();
            !qs.iter().any(|q| sv.contains(q))
        }
        "between" => {
            let (lo, hi) = extract_bounds(&sel.values);
            match (lo, hi) {
                (Some(lo), Some(hi)) => query_vals
                    .iter()
                    .all(|v| scalar_to_f64(v).map(|f| f < lo || f > hi).unwrap_or(false)),
                _ => false,
            }
        }
        _ => false,
    }
}

/// Single-comparison selector (gt/gte/lt/lte) vs query comparison.
fn range_contradicts(sel_op: &str, sv: f64, query_op: Operator, qv: f64) -> bool {
    match sel_op {
        // All rows have field > sv
        "gt" => matches!(query_op,
            Operator::Eq | Operator::Lt | Operator::LtEq if qv <= sv),
        // All rows have field >= sv
        "gte" => match query_op {
            Operator::Eq | Operator::Lt => qv < sv,
            Operator::LtEq => qv < sv,
            _ => false,
        },
        // All rows have field < sv
        "lt" => matches!(query_op,
            Operator::Eq | Operator::Gt | Operator::GtEq if qv >= sv),
        // All rows have field <= sv
        "lte" => match query_op {
            Operator::Eq | Operator::Gt => qv > sv,
            Operator::GtEq => qv > sv,
            _ => false,
        },
        _ => false,
    }
}

// ── Helpers ────────────────────────────────────────────────────

fn flip(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

fn extract_values(v: &serde_json::Value) -> Vec<String> {
    match v {
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|x| match x {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect(),
        serde_json::Value::String(s) => vec![s.clone()],
        other => vec![other.to_string()],
    }
}

fn extract_bounds(v: &serde_json::Value) -> (Option<f64>, Option<f64>) {
    if let serde_json::Value::Array(arr) = v {
        if arr.len() == 2 {
            return (json_f64(&arr[0]), json_f64(&arr[1]));
        }
    }
    (None, None)
}

fn extract_single_num(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Array(a) if !a.is_empty() => json_f64(&a[0]),
        _ => json_f64(v),
    }
}

fn json_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn scalar_to_string(s: &ScalarValue) -> String {
    match s {
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => v.clone(),
        ScalarValue::Utf8View(Some(v)) => v.clone(),
        ScalarValue::Int8(Some(v)) => v.to_string(),
        ScalarValue::Int16(Some(v)) => v.to_string(),
        ScalarValue::Int32(Some(v)) => v.to_string(),
        ScalarValue::Int64(Some(v)) => v.to_string(),
        ScalarValue::UInt8(Some(v)) => v.to_string(),
        ScalarValue::UInt16(Some(v)) => v.to_string(),
        ScalarValue::UInt32(Some(v)) => v.to_string(),
        ScalarValue::UInt64(Some(v)) => v.to_string(),
        ScalarValue::Float32(Some(v)) => v.to_string(),
        ScalarValue::Float64(Some(v)) => v.to_string(),
        other => format!("{other}"),
    }
}

fn scalar_to_f64(s: &ScalarValue) -> Option<f64> {
    match s {
        ScalarValue::Int8(Some(v)) => Some(*v as f64),
        ScalarValue::Int16(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::UInt8(Some(v)) => Some(*v as f64),
        ScalarValue::UInt16(Some(v)) => Some(*v as f64),
        ScalarValue::UInt32(Some(v)) => Some(*v as f64),
        ScalarValue::UInt64(Some(v)) => Some(*v as f64),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Float64(Some(v)) => Some(*v as f64),
        ScalarValue::Utf8(Some(v)) => v.parse().ok(),
        _ => None,
    }
}

/// True if every value in `sv` is ≤ the scalar (so `field > scalar` is impossible).
fn all_leq(sv: &[String], scalar: &ScalarValue) -> bool {
    let q = scalar_to_f64(scalar);
    q.map(|qv| {
        sv.iter()
            .all(|v| v.parse::<f64>().map(|f| f <= qv).unwrap_or(false))
    })
    .unwrap_or(false)
}

fn all_lt(sv: &[String], scalar: &ScalarValue) -> bool {
    let q = scalar_to_f64(scalar);
    q.map(|qv| {
        sv.iter()
            .all(|v| v.parse::<f64>().map(|f| f < qv).unwrap_or(false))
    })
    .unwrap_or(false)
}

fn all_geq(sv: &[String], scalar: &ScalarValue) -> bool {
    let q = scalar_to_f64(scalar);
    q.map(|qv| {
        sv.iter()
            .all(|v| v.parse::<f64>().map(|f| f >= qv).unwrap_or(false))
    })
    .unwrap_or(false)
}

fn all_gt(sv: &[String], scalar: &ScalarValue) -> bool {
    let q = scalar_to_f64(scalar);
    q.map(|qv| {
        sv.iter()
            .all(|v| v.parse::<f64>().map(|f| f > qv).unwrap_or(false))
    })
    .unwrap_or(false)
}

// ── Bitmap key-value extraction ────────────────────────────────

/// Extract literal key values from a DataFusion predicate that filters on `key_column`.
///
/// Returns `Some(vec![...])` with u64 bitmap keys when the predicate is a simple
/// equality, IN-list, or AND combination targeting the key column.
/// Returns `None` for anything else (conservative: no pruning).
///
/// When `is_string_key` is true, string literals are hashed via `hash_key()`.
/// Integer literals are used directly as u64.
pub fn extract_key_filter_values(
    expr: &Expr,
    key_column: &str,
    is_string_key: bool,
) -> Option<Vec<u64>> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
            Operator::Eq => {
                extract_eq_value(left, right, key_column, is_string_key)
                    .or_else(|| extract_eq_value(right, left, key_column, is_string_key))
            }
            Operator::And => {
                let l = extract_key_filter_values(left, key_column, is_string_key);
                let r = extract_key_filter_values(right, key_column, is_string_key);
                match (l, r) {
                    (Some(mut a), Some(b)) => {
                        a.extend(b);
                        Some(a)
                    }
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            }
            Operator::Or => {
                // OR: patient_id = 1 OR patient_id = 2
                // Can only prune when BOTH sides extract values (if either side
                // is non-extractable, it might match any file → can't skip).
                let l = extract_key_filter_values(left, key_column, is_string_key);
                let r = extract_key_filter_values(right, key_column, is_string_key);
                match (l, r) {
                    (Some(mut a), Some(b)) => {
                        a.extend(b);
                        Some(a)
                    }
                    _ => None,
                }
            }
            _ => None,
        },
        Expr::InList(il) => {
            let col_name = match il.expr.as_ref() {
                Expr::Column(c) => c.name(),
                _ => return None,
            };
            if col_name.to_lowercase() != key_column.to_lowercase() {
                return None;
            }
            if il.negated {
                return None;
            }
            let vals: Option<Vec<u64>> = il
                .list
                .iter()
                .map(|e| scalar_to_bitmap_key(e, is_string_key))
                .collect();
            vals
        }
        _ => None,
    }
}

fn extract_eq_value(
    col_side: &Expr,
    lit_side: &Expr,
    key_column: &str,
    is_string_key: bool,
) -> Option<Vec<u64>> {
    let col_name = match col_side {
        Expr::Column(c) => c.name(),
        _ => return None,
    };
    if col_name.to_lowercase() != key_column.to_lowercase() {
        return None;
    }
    scalar_to_bitmap_key(lit_side, is_string_key).map(|v| vec![v])
}

fn scalar_to_bitmap_key(expr: &Expr, is_string_key: bool) -> Option<u64> {
    match expr {
        Expr::Literal(scalar, _) => {
            if is_string_key {
                match scalar {
                    ScalarValue::Utf8(Some(s))
                    | ScalarValue::LargeUtf8(Some(s))
                    | ScalarValue::Utf8View(Some(s)) => Some(hash_key(s)),
                    _ => None,
                }
            } else {
                match scalar {
                    ScalarValue::Int8(Some(v)) => Some(*v as u64),
                    ScalarValue::Int16(Some(v)) => Some(*v as u64),
                    ScalarValue::Int32(Some(v)) => Some(*v as u64),
                    ScalarValue::Int64(Some(v)) => Some(*v as u64),
                    ScalarValue::UInt8(Some(v)) => Some(*v as u64),
                    ScalarValue::UInt16(Some(v)) => Some(*v as u64),
                    ScalarValue::UInt32(Some(v)) => Some(*v as u64),
                    ScalarValue::UInt64(Some(v)) => Some(*v),
                    _ => None,
                }
            }
        }
        _ => None,
    }
}
