library(ggplot2)

all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", all_args[grep(file_arg, all_args)])
script_dir <- if (length(script_path) > 0) {
  dirname(normalizePath(script_path))
} else {
  getwd()
}

args <- commandArgs(trailingOnly = TRUE)
lit_file <- if (length(args) >= 1) args[1] else "classification_literature_range.csv"
cur_file <- if (length(args) >= 2) args[2] else "classification_current_work.csv"
out_stem <- if (length(args) >= 3) args[3] else "classification_benchmark_reviz"
y_label <- if (length(args) >= 4) args[4] else "ROC-AUC"
plot_title <- if (length(args) >= 5) args[5] else "Biological Activity Benchmarks with Literature Ranges"
rescale_midpoint <- if (length(args) >= 6) tolower(args[6]) %in% c("1", "true", "t", "yes", "y") else FALSE
plot_subtitle <- if (length(args) >= 7 && nzchar(args[7])) args[7] else NULL
plot_width_cm <- if (length(args) >= 8) as.numeric(args[8]) else 9
plot_height_cm <- if (length(args) >= 9) as.numeric(args[9]) else 10

lit_path <- file.path(script_dir, lit_file)
cur_path <- file.path(script_dir, cur_file)
out_png <- file.path(script_dir, paste0(out_stem, ".png"))
out_pdf <- file.path(script_dir, paste0(out_stem, ".pdf"))

literature <- read.csv(lit_path, stringsAsFactors = FALSE)
current <- read.csv(cur_path, stringsAsFactors = FALSE)

if ("n_points" %in% names(literature)) {
  literature$n_points <- as.numeric(literature$n_points)
  literature <- literature[order(literature$n_points, literature$benchmark), ]
  benchmark_labels <- paste0(
    literature$benchmark,
    " (",
    format(literature$n_points, big.mark = ",", scientific = FALSE, trim = TRUE),
    ")"
  )
} else {
  benchmark_labels <- literature$benchmark
}

benchmark_levels <- literature$benchmark
literature$benchmark <- factor(literature$benchmark, levels = benchmark_levels)
current$benchmark <- factor(current$benchmark, levels = benchmark_levels)
current$setting <- factor(current$setting, levels = c("Unfrozen", "Frozen"))
literature$bench_idx <- as.numeric(literature$benchmark)
current$bench_idx <- as.numeric(current$benchmark)

if (rescale_midpoint) {
  lit_mid <- (literature$lit_min + literature$lit_max) / 2
  names(lit_mid) <- as.character(literature$benchmark)

  literature$lit_min <- literature$lit_min / lit_mid[as.character(literature$benchmark)]
  literature$lit_max <- literature$lit_max / lit_mid[as.character(literature$benchmark)]
  current$roc_auc <- current$roc_auc / lit_mid[as.character(current$benchmark)]
  current$sd <- current$sd / lit_mid[as.character(current$benchmark)]

  if (length(args) < 4) {
    y_label <- "Relative to literature midpoint (=1)"
  }
}

current$plot_model <- NA_character_
current$plot_model[current$model == "Uni-Mol replicate" & current$setting == "Frozen"] <- "Uni-Mol replicate (frozen)"
current$plot_model[current$model == "ConforFormer--UniMol" & current$setting == "Frozen"] <- "ConforFormer-UniMol"
current$plot_model[current$model == "ConforFormer--OMol" & current$setting == "Frozen"] <- "ConforFormer-OMol"
current$plot_model[current$model == "CatBoost FP4 baseline"] <- "CatBoost FP4 baseline"
current$plot_model[current$model == "XGBoost ECFP4_1024 baseline"] <- "XGBoost ECFP4_1024 baseline"
current <- current[!is.na(current$plot_model), ]

model_levels <- c(
  "Uni-Mol replicate (frozen)",
  "ConforFormer-UniMol",
  "ConforFormer-OMol",
  "CatBoost FP4 baseline",
  "XGBoost ECFP4_1024 baseline"
)
current$plot_model <- factor(current$plot_model, levels = model_levels)
current <- current[order(current$plot_model, current$bench_idx), ]
literature <- literature[order(literature$bench_idx), ]

y_min <- min(literature$lit_min, current$roc_auc - current$sd, na.rm = TRUE) - 0.02
y_max <- max(literature$lit_max, current$roc_auc + current$sd, na.rm = TRUE) + 0.02
y_breaks <- pretty(c(y_min, y_max), n = 8)

plot_obj <- ggplot() +
  geom_ribbon(
    data = literature,
    aes(x = bench_idx, ymin = lit_min, ymax = lit_max, group = 1),
    inherit.aes = FALSE,
    alpha = 0.45,
    fill = "#9FB0CB"
  ) +
  geom_line(
    data = current,
    aes(x = bench_idx, y = roc_auc, group = plot_model, color = plot_model, linetype = plot_model),
    linewidth = 0.45,
    alpha = 0.85
  ) +
  geom_errorbar(
    data = current,
    aes(
      ymin = roc_auc - sd,
      ymax = roc_auc + sd,
      x = bench_idx,
      color = plot_model
    ),
    width = 0.12,
    linewidth = 0.35,
    alpha = 0.9
  ) +
  geom_point(
    data = current,
    aes(x = bench_idx, y = roc_auc, color = plot_model),
    size = 1.9,
    alpha = 0.95
  ) +
  scale_x_continuous(
    breaks = seq_along(benchmark_levels),
    labels = benchmark_labels
  ) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = y_breaks) +
  scale_color_manual(
    name = "Model",
    values = c(
      "Uni-Mol replicate (frozen)" = "#1B9E77",
      "ConforFormer-UniMol" = "#D95F02",
      "ConforFormer-OMol" = "#7570B3",
      "CatBoost FP4 baseline" = "#4D4D4D",
      "XGBoost ECFP4_1024 baseline" = "#355C7D"
    )
  ) +
  scale_linetype_manual(
    name = "Model",
    values = c(
      "Uni-Mol replicate (frozen)" = "solid",
      "ConforFormer-UniMol" = "solid",
      "ConforFormer-OMol" = "solid",
      "CatBoost FP4 baseline" = "dotdash",
      "XGBoost ECFP4_1024 baseline" = "dashed"
    )
  ) +
  labs(
    x = "Benchmark (and point count)",
    y = y_label,
    title = plot_title,
    subtitle = plot_subtitle
  ) +
  {
    if (rescale_midpoint) geom_hline(yintercept = 1, linewidth = 0.35, color = "grey45", linetype = "dotted")
  } +
  coord_flip() +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.major.y = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.key.height = unit(3.8, "mm"),
    legend.key.width = unit(5, "mm")
  ) +
  guides(color = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1))

ggsave(out_png, plot = plot_obj, width = plot_width_cm, height = plot_height_cm, units = "cm", dpi = 600)
ggsave(out_pdf, plot = plot_obj, width = plot_width_cm, height = plot_height_cm, units = "cm", dpi = 600)

cat("Wrote:\n")
cat(" - ", out_png, "\n", sep = "")
cat(" - ", out_pdf, "\n", sep = "")
