use super::BaselineRun;
use crate::engine::OUT_COLS;
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: u32 = 2400;
const CUMULATIVE_HEIGHT: u32 = 1000;
const DRAWDOWN_HEIGHT: u32 = 600;
const HEATMAP_WIDTH: i32 = 2400;
const CELL_WIDTH: i32 = 170;
const CELL_HEIGHT: i32 = 66;
const LEFT_MARGIN: i32 = 190;
const TOP_MARGIN: i32 = 132;

#[derive(Debug, Clone)]
struct ReturnSeries {
    name: String,
    returns: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug, Clone, Copy)]
struct PlotArea {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

impl PlotArea {
    fn right(self) -> f64 {
        self.left + self.width
    }

    fn bottom(self) -> f64 {
        self.top + self.height
    }

    fn x(self, idx: usize, len: usize) -> f64 {
        if len <= 1 {
            return self.left;
        }
        self.left + self.width * idx as f64 / (len - 1) as f64
    }

    fn y(self, value: f64, y_min: f64, y_max: f64) -> f64 {
        if (y_max - y_min).abs() < 1e-12 {
            return self.top + self.height / 2.0;
        }
        self.bottom() - self.height * (value - y_min) / (y_max - y_min)
    }
}

pub(super) fn write_backtest_plots(
    output_dir: &Path,
    dates_ns: &[i64],
    out: &[f64],
    baseline_runs: &[BaselineRun],
) -> Result<Vec<PathBuf>, String> {
    let dates = dates_ns
        .iter()
        .map(|value| datetime_ns_to_date(*value))
        .collect::<Result<Vec<_>, _>>()?;
    let series = build_return_series(dates_ns, out, baseline_runs);
    let cumulative_path = output_dir.join("native_cumulative_return.svg");
    let drawdown_path = output_dir.join("native_drawdown.svg");
    let heatmap_path = output_dir.join("native_monthly_heatmap.svg");
    plot_cumulative_return(&cumulative_path, &dates, &series)?;
    plot_drawdown(&drawdown_path, &dates, &strategy_returns(out))?;
    plot_monthly_heatmap(&heatmap_path, &dates, &strategy_returns(out))?;
    Ok(vec![cumulative_path, drawdown_path, heatmap_path])
}

fn build_return_series(
    dates_ns: &[i64],
    out: &[f64],
    baseline_runs: &[BaselineRun],
) -> Vec<ReturnSeries> {
    let mut series = vec![
        ReturnSeries {
            name: "Strategy".to_owned(),
            returns: strategy_returns(out),
        },
        ReturnSeries {
            name: "Benchmark".to_owned(),
            returns: column_values(out, 4),
        },
    ];
    for baseline in baseline_runs {
        series.push(ReturnSeries {
            name: plot_display_name(&baseline.display_name),
            returns: column_values(&baseline.out, 1),
        });
    }
    let fixed_by_date = baseline_runs
        .iter()
        .map(|baseline| {
            baseline
                .fixed_risk_dates_ns
                .iter()
                .enumerate()
                .map(|(row, date_ns)| (*date_ns, baseline.fixed_risk_out[row * OUT_COLS + 1]))
                .collect::<BTreeMap<_, _>>()
        })
        .collect::<Vec<_>>();
    for (baseline, fixed_returns) in baseline_runs.iter().zip(fixed_by_date.iter()) {
        series.push(ReturnSeries {
            name: plot_display_name(&format!("Fixed-Risk {}", baseline.display_name)),
            returns: dates_ns
                .iter()
                .map(|date_ns| fixed_returns.get(date_ns).copied().unwrap_or(0.0))
                .collect(),
        });
    }
    series
}

fn plot_display_name(name: &str) -> String {
    name.replace("Fixed-Risk", "Fixed Risk")
        .replace("Sign-Aligned", "Sign Aligned")
        .replace("Rank-ZScore", "Rank Z-Score")
        .replace("RankIC-Weighted", "Rank IC Weighted")
}

fn strategy_returns(out: &[f64]) -> Vec<f64> {
    column_values(out, 1)
}

fn column_values(out: &[f64], col: usize) -> Vec<f64> {
    (0..out.len() / OUT_COLS)
        .map(|row| out[row * OUT_COLS + col])
        .collect()
}

fn cumulative_values(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = 1.0_f64;
    let mut out = Vec::with_capacity(returns.len());
    for value in returns {
        if value.is_finite() {
            cumulative *= 1.0 + value;
        }
        out.push(cumulative);
    }
    out
}

fn drawdown_values(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut out = Vec::with_capacity(returns.len());
    for value in returns {
        if value.is_finite() {
            cumulative *= 1.0 + value;
        }
        peak = peak.max(cumulative);
        out.push(if peak > 0.0 {
            cumulative / peak - 1.0
        } else {
            0.0
        });
    }
    out
}

fn plot_cumulative_return(
    path: &Path,
    dates: &[NaiveDate],
    series: &[ReturnSeries],
) -> Result<(), String> {
    if dates.is_empty() {
        return Ok(());
    }
    let cumulative = series
        .iter()
        .map(|item| (item.name.clone(), cumulative_values(&item.returns)))
        .collect::<Vec<_>>();
    let (y_min, y_max) = expand_range(
        cumulative
            .iter()
            .flat_map(|(_, values)| values.iter().copied())
            .filter(|value| value.is_finite()),
        0.08,
    );
    let area = PlotArea {
        left: 110.0,
        top: 88.0,
        width: WIDTH as f64 - 150.0,
        height: CUMULATIVE_HEIGHT as f64 - 178.0,
    };
    let mut svg = svg_open(WIDTH, CUMULATIVE_HEIGHT);
    svg.push_str(&svg_text(
        WIDTH as f64 / 2.0,
        54.0,
        "Cumulative Return",
        42,
        "middle",
        None,
    ));
    draw_grid(&mut svg, area, dates, y_min, y_max, 9, |value| {
        format!("{value:.2}")
    });
    draw_legend_box(
        &mut svg,
        area,
        cumulative.iter().map(|(name, _)| name.as_str()),
    );
    for (idx, (name, values)) in cumulative.iter().enumerate() {
        let color = series_color(idx);
        draw_polyline(
            &mut svg,
            area,
            dates.len(),
            values,
            y_min,
            y_max,
            color,
            2.3,
        );
        draw_legend_entry(&mut svg, area, idx, name, color);
    }
    svg.push_str("</svg>\n");
    fs::write(path, svg).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn plot_drawdown(path: &Path, dates: &[NaiveDate], returns: &[f64]) -> Result<(), String> {
    if dates.is_empty() {
        return Ok(());
    }
    let drawdown = drawdown_values(returns);
    let min_drawdown = drawdown
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::min);
    let y_min = (min_drawdown * 1.08).min(-0.01);
    let area = PlotArea {
        left: 110.0,
        top: 82.0,
        width: WIDTH as f64 - 150.0,
        height: DRAWDOWN_HEIGHT as f64 - 162.0,
    };
    let mut svg = svg_open(WIDTH, DRAWDOWN_HEIGHT);
    svg.push_str(&svg_text(
        WIDTH as f64 / 2.0,
        52.0,
        &format!("Drawdown (max={:.2}%)", min_drawdown * 100.0),
        40,
        "middle",
        None,
    ));
    draw_grid(&mut svg, area, dates, y_min, 0.0, 7, |value| {
        format!("{:.0}%", value * 100.0)
    });
    draw_area_series(
        &mut svg,
        area,
        dates.len(),
        &drawdown,
        y_min,
        0.0,
        Color {
            r: 215,
            g: 48,
            b: 39,
        },
        0.45,
    );
    svg.push_str("</svg>\n");
    fs::write(path, svg).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn plot_monthly_heatmap(path: &Path, dates: &[NaiveDate], returns: &[f64]) -> Result<(), String> {
    let monthly = monthly_compound_returns(dates, returns);
    let years = monthly
        .keys()
        .map(|(year, _)| *year)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let height = (TOP_MARGIN + CELL_HEIGHT * years.len() as i32 + 58).max(260) as u32;
    let mut svg = svg_open(HEATMAP_WIDTH as u32, height);
    svg.push_str(&svg_text(
        HEATMAP_WIDTH as f64 / 2.0,
        56.0,
        "Monthly Returns Heatmap",
        42,
        "middle",
        None,
    ));
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    for (month_idx, month) in months.iter().enumerate() {
        let x = LEFT_MARGIN + month_idx as i32 * CELL_WIDTH + CELL_WIDTH / 2;
        svg.push_str(&svg_text(
            x as f64,
            (TOP_MARGIN - 24) as f64,
            month,
            26,
            "middle",
            None,
        ));
    }
    for (row, year) in years.iter().enumerate() {
        let y0 = TOP_MARGIN + row as i32 * CELL_HEIGHT;
        svg.push_str(&svg_text(
            42.0,
            (y0 + CELL_HEIGHT / 2 + 1) as f64,
            &year.to_string(),
            28,
            "start",
            Some("middle"),
        ));
        for month in 1..=12u32 {
            let x0 = LEFT_MARGIN + (month - 1) as i32 * CELL_WIDTH;
            let x1 = x0 + CELL_WIDTH;
            let y1 = y0 + CELL_HEIGHT;
            let value = monthly.get(&(*year, month)).copied();
            let color = value.map(heatmap_color).unwrap_or(Color {
                r: 245,
                g: 245,
                b: 245,
            });
            svg.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" />\n",
                x0,
                y0,
                CELL_WIDTH,
                CELL_HEIGHT,
                color_hex(color)
            ));
            svg.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#ebebeb\" stroke-width=\"1\" />\n",
                x0,
                y0,
                x1 - x0,
                y1 - y0
            ));
            if let Some(ret) = value {
                svg.push_str(&svg_text(
                    (x0 + CELL_WIDTH / 2) as f64,
                    (y0 + CELL_HEIGHT / 2 + 1) as f64,
                    &format!("{:.2}%", ret * 100.0),
                    24,
                    "middle",
                    Some("middle"),
                ));
            }
        }
    }
    svg.push_str("</svg>\n");
    fs::write(path, svg).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn draw_grid(
    svg: &mut String,
    area: PlotArea,
    dates: &[NaiveDate],
    y_min: f64,
    y_max: f64,
    y_ticks: usize,
    y_label: impl Fn(f64) -> String,
) {
    svg.push_str(&format!(
        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"#ffffff\" stroke=\"#111111\" stroke-width=\"1\" />\n",
        area.left, area.top, area.width, area.height
    ));
    let ticks = linear_ticks(y_min, y_max, y_ticks);
    for (idx, value) in ticks.iter().enumerate() {
        let y = area.y(*value, y_min, y_max);
        let stroke = if idx == 0 || idx + 1 == ticks.len() {
            "#b6b6b6"
        } else {
            "#d2d2d2"
        };
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"1\" />\n",
            area.left,
            y,
            area.right(),
            y,
            stroke
        ));
        svg.push_str(&svg_text(
            area.left - 14.0,
            y + 1.0,
            &y_label(*value),
            24,
            "end",
            Some("middle"),
        ));
        if idx + 1 < ticks.len() {
            let next = ticks[idx + 1];
            for minor in 1..5 {
                let minor_value = value + (next - value) * minor as f64 / 5.0;
                let minor_y = area.y(minor_value, y_min, y_max);
                svg.push_str(&format!(
                    "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#eeeeee\" stroke-width=\"1\" />\n",
                    area.left,
                    minor_y,
                    area.right(),
                    minor_y
                ));
            }
        }
    }
    for idx in x_tick_indices(dates.len(), 10) {
        let x = area.x(idx, dates.len());
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#c8c8c8\" stroke-width=\"1\" />\n",
            x,
            area.top,
            x,
            area.bottom()
        ));
        svg.push_str(&svg_text(
            x,
            area.bottom() + 34.0,
            &date_label(dates, idx),
            24,
            "middle",
            None,
        ));
    }
}

fn draw_polyline(
    svg: &mut String,
    area: PlotArea,
    len: usize,
    values: &[f64],
    y_min: f64,
    y_max: f64,
    color: Color,
    stroke_width: f64,
) {
    let points = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            value.is_finite().then(|| {
                format!(
                    "{:.2},{:.2}",
                    area.x(idx, len),
                    area.y(*value, y_min, y_max)
                )
            })
        })
        .collect::<Vec<_>>()
        .join(" ");
    if points.is_empty() {
        return;
    }
    svg.push_str(&format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"round\" />\n",
        points,
        color_hex(color),
        stroke_width
    ));
}

fn draw_area_series(
    svg: &mut String,
    area: PlotArea,
    len: usize,
    values: &[f64],
    y_min: f64,
    y_max: f64,
    color: Color,
    opacity: f64,
) {
    let points = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            value
                .is_finite()
                .then(|| (area.x(idx, len), area.y(*value, y_min, y_max), idx, *value))
        })
        .collect::<Vec<_>>();
    let Some((first_x, _, _, _)) = points.first().copied() else {
        return;
    };
    let Some((last_x, _, _, _)) = points.last().copied() else {
        return;
    };
    let zero_y = area.y(0.0, y_min, y_max);
    let mut path = format!("M {:.2} {:.2}", first_x, zero_y);
    for (x, y, _, _) in &points {
        path.push_str(&format!(" L {:.2} {:.2}", x, y));
    }
    path.push_str(&format!(" L {:.2} {:.2} Z", last_x, zero_y));
    svg.push_str(&format!(
        "<path d=\"{}\" fill=\"{}\" fill-opacity=\"{:.3}\" stroke=\"none\" />\n",
        path,
        color_hex(color),
        opacity
    ));
    draw_polyline(svg, area, len, values, y_min, y_max, color, 1.8);
}

fn draw_legend_box<'a>(svg: &mut String, area: PlotArea, names: impl Iterator<Item = &'a str>) {
    let names = names.collect::<Vec<_>>();
    let max_chars = names
        .iter()
        .map(|name| name.chars().count())
        .max()
        .unwrap_or(0);
    let width = (90.0 + max_chars as f64 * 16.0).max(240.0);
    let height = 18.0 + names.len() as f64 * 32.0;
    svg.push_str(&format!(
        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"#ffffff\" fill-opacity=\"0.88\" stroke=\"#111111\" stroke-width=\"1\" />\n",
        area.left + 8.0,
        area.top + 8.0,
        width,
        height
    ));
}

fn draw_legend_entry(svg: &mut String, area: PlotArea, idx: usize, name: &str, color: Color) {
    let x = area.left + 22.0;
    let y = area.top + 28.0 + idx as f64 * 32.0;
    svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"2.8\" />\n",
        x,
        y,
        x + 22.0,
        y,
        color_hex(color)
    ));
    svg.push_str(&svg_text(
        x + 34.0,
        y + 1.0,
        name,
        24,
        "start",
        Some("middle"),
    ));
}

fn svg_open(width: u32, height: u32) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n\
<style>\n\
svg {{ background: #ffffff; }}\n\
text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", \"Adwaita Mono\", monospace; fill: #111111; font-kerning: normal; text-rendering: optimizeLegibility; }}\n\
polyline, path, line, rect {{ shape-rendering: geometricPrecision; }}\n\
</style>\n",
        width, height, width, height
    )
}

fn svg_text(
    x: f64,
    y: f64,
    text: &str,
    font_size: u32,
    anchor: &str,
    baseline: Option<&str>,
) -> String {
    let baseline_attr = baseline
        .map(|value| format!(" dominant-baseline=\"{}\"", value))
        .unwrap_or_default();
    format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{}\" text-anchor=\"{}\"{}>{}</text>\n",
        x,
        y,
        font_size,
        anchor,
        baseline_attr,
        escape_xml(text)
    )
}

fn escape_xml(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn x_tick_indices(len: usize, labels: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 || labels <= 1 {
        return vec![0];
    }
    let mut out = Vec::new();
    for idx in 0..labels {
        let value = ((len - 1) as f64 * idx as f64 / (labels - 1) as f64).round() as usize;
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

fn linear_ticks(min_value: f64, max_value: f64, count: usize) -> Vec<f64> {
    if count <= 1 || (max_value - min_value).abs() < 1e-12 {
        return vec![min_value];
    }
    (0..count)
        .map(|idx| min_value + (max_value - min_value) * idx as f64 / (count - 1) as f64)
        .collect()
}

fn monthly_compound_returns(dates: &[NaiveDate], returns: &[f64]) -> BTreeMap<(i32, u32), f64> {
    let mut products = BTreeMap::<(i32, u32), f64>::new();
    for (date, value) in dates.iter().zip(returns.iter()) {
        if !value.is_finite() {
            continue;
        }
        let key = (date.year(), date.month());
        let product = products.entry(key).or_insert(1.0);
        *product *= 1.0 + value;
    }
    products
        .into_iter()
        .map(|(key, product)| (key, product - 1.0))
        .collect()
}

fn heatmap_color(value: f64) -> Color {
    let clamped = value.clamp(-0.5, 0.5);
    if clamped < 0.0 {
        interpolate_color(
            Color {
                r: 26,
                g: 152,
                b: 80,
            },
            Color {
                r: 255,
                g: 255,
                b: 255,
            },
            (clamped + 0.5) / 0.5,
        )
    } else {
        interpolate_color(
            Color {
                r: 255,
                g: 255,
                b: 255,
            },
            Color {
                r: 215,
                g: 48,
                b: 39,
            },
            clamped / 0.5,
        )
    }
}

fn interpolate_color(start: Color, end: Color, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    Color {
        r: (start.r as f64 + (end.r as f64 - start.r as f64) * t).round() as u8,
        g: (start.g as f64 + (end.g as f64 - start.g as f64) * t).round() as u8,
        b: (start.b as f64 + (end.b as f64 - start.b as f64) * t).round() as u8,
    }
}

fn series_color(idx: usize) -> Color {
    const COLORS: [Color; 10] = [
        Color {
            r: 31,
            g: 119,
            b: 180,
        },
        Color {
            r: 255,
            g: 127,
            b: 14,
        },
        Color {
            r: 44,
            g: 160,
            b: 44,
        },
        Color {
            r: 214,
            g: 39,
            b: 40,
        },
        Color {
            r: 148,
            g: 103,
            b: 189,
        },
        Color {
            r: 140,
            g: 86,
            b: 75,
        },
        Color {
            r: 227,
            g: 119,
            b: 194,
        },
        Color {
            r: 127,
            g: 127,
            b: 127,
        },
        Color {
            r: 188,
            g: 189,
            b: 34,
        },
        Color {
            r: 23,
            g: 190,
            b: 207,
        },
    ];
    COLORS[idx % COLORS.len()]
}

fn color_hex(color: Color) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
}

fn expand_range(values: impl Iterator<Item = f64>, padding_ratio: f64) -> (f64, f64) {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for value in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }
    if !min_value.is_finite() || !max_value.is_finite() {
        return (0.0, 1.0);
    }
    if (max_value - min_value).abs() < 1e-12 {
        return (min_value - 0.01, max_value + 0.01);
    }
    let padding = (max_value - min_value) * padding_ratio;
    (min_value - padding, max_value + padding)
}

fn date_label(dates: &[NaiveDate], idx: usize) -> String {
    dates
        .get(idx.min(dates.len().saturating_sub(1)))
        .map(|date| date.format("%Y-%m").to_string())
        .unwrap_or_default()
}

fn datetime_ns_to_date(value: i64) -> Result<NaiveDate, String> {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|datetime| datetime.date_naive())
        .ok_or_else(|| format!("invalid timestamp ns: {value}"))
}
