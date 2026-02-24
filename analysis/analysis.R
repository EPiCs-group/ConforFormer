library(RSQLite)
library(tidyverse)
library(dbplyr)
library(stringr)
unimol_con <- dbConnect(RSQLite::SQLite(), "~/Downloads/UniMol_sim.sqlite3")
dbListTables(unimol_con)

unimol_sim_scores <- tbl(unimol_con, "sim_scores")

unimol_sim_scores %>%
  summarize(across(c(tgt, is_enantiomer, is_stereoisomer, is_diastereomer), function (x) 100*mean(x))) %>%
  mutate(backbone = 100 - tgt - is_stereoisomer)

strucs <- tbl(unimol_con, "smiles")
strucs %>%
  summarize(n_distinct(smi))
contrast_con <- dbConnect(RSQLite::SQLite(), "~/Downloads/Seed3_contrast_sim.sqlite3")
contrast_sim_scores <- tbl(contrast_con, "sim_scores")
contrast_sim_scores

contrast_sim_scores %>%
  collect() %>%
  summarize(n(), n_distinct(smi_1_id, smi_2_id))

contrast_overview <- contrast_sim_scores %>%
  group_by(sim = ceiling(cos_sim * 1000)) %>%
  filter(sim >= 0 & sim <= 1000) %>%
  summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
  mutate(ds="Seed3 contrast") %>%
  collect()

contrast_stereo_overview <- contrast_sim_scores %>%
  filter(is_stereoisomer == 1) %>%
  group_by(sim = ceiling(cos_sim * 1000)) %>%
  filter(sim >= 0 & sim <= 1000) %>%
  summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
  mutate(ds = "Seed3_contrast stereo") %>%
  collect()

make_overview <- function(path, ds_name, npoints = 1000) {
  con <- dbConnect(RSQLite::SQLite(), path)
  union_all(
    tbl(con, "sim_scores") %>%
      group_by(sim = ceiling(cos_sim * npoints)) %>%
      filter(sim >= 0 & sim <= npoints) %>%
      summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
      mutate(ds=ds_name, isomer_type="All") %>%
      collect(),
    tbl(con, "sim_scores") %>%
      mutate(isomer_type=if_else(is_stereoisomer == 1, "Stereo", "Backbone")) %>%
      group_by(sim = ceiling(cos_sim * npoints), isomer_type) %>%
      filter(sim >= 0 & sim <= npoints) %>%
      summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
      ungroup() %>%
      mutate(ds=ds_name) %>%
      collect()
  )
} 

tanimoto_overview <- union_all(
  contrast_sim_scores %>%
    group_by(sim = ceiling((1 - tanimoto_dist) * 10000)) %>%
    filter(sim >= 0 & sim <= 10000) %>%
    summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
    mutate(ds="Tanimoto FP2", isomer_type="All") %>%
    collect(),
  contrast_sim_scores %>%
    mutate(isomer_type=if_else(is_stereoisomer == 1, "Stereo", "Backbone")) %>%
    group_by(sim = ceiling((1 - tanimoto_dist) * 10000), isomer_type) %>%
    filter(sim >= 0 & sim <= 10000) %>%
    summarize(tgt_frac = avg((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
    ungroup() %>%
    mutate(ds="Tanimoto FP2") %>%
    collect()
)


contrast_overview <- make_overview( "~/Downloads/Seed3_contrast_sim.sqlite3", "Contrast Head", 10000)
omol_overview <- filldataset_contrast_overview <- make_overview(
  "~/Downloads/OMol_sim.sqlite3",
  "Omol",
  10000
)

omol_contrast_overview <- filldataset_contrast_overview <- make_overview(
  "~/Downloads/OMol_contrast_sim.sqlite3",
  "Omol Contrast",
  10000
)
omol2_contrast_overview <- filldataset_contrast_overview <- make_overview(
  "~/Downloads/OMol_contrast_moretrain_sim.sqlite3",
  "Omol Contrast 2",
  10000
)

filldataset_contrast_overview <- make_overview(
  "~/Downloads/FullDatasetContrast_sim.sqlite3",
  "Unimol Contrast",
  10000)
unimol_overview <- make_overview( "~/Downloads/UniMol_sim.sqlite3", "Unimol", 10000)
replicate_overview <- make_overview("~/Downloads/Replicate_sim.sqlite3", "Replicate", 10000)
onlycontrast_overview <- make_overview("~/Downloads/Seed3_onlyContrast_sim.sqlite3", 
                                       "Only Contrast", 10000)
contrast_overview %>%
  arrange(desc(sim)) %>%
  mutate(run_total = cumsum(total), run_tgt = cumsum(targets)) %>%
  mutate(recall = run_tgt/sum(targets), precision = run_tgt/run_total) %>% 
  View()

unimol_overview %>%
  arrange(desc(sim)) %>%
  mutate(run_total = cumsum(total), run_tgt = cumsum(targets)) %>%
  mutate(recall = run_tgt/sum(targets), precision = run_tgt/run_total)

randomguess_overview <- contrast_sim_scores %>%
  filter(is_stereoisomer == 1) %>%
  collect() %>%
  mutate(cos_sim = runif(n())) %>%
  group_by(sim = ceiling(cos_sim * 1000)) %>%
  filter(sim >= 0 & sim <= 1000) %>%
  summarize(tgt_frac = mean((tgt |  is_enantiomer)*1.0), targets = sum((tgt |  is_enantiomer)), total = n()) %>%
  mutate(ds = "randomguess stereo")

contrast_plot <- 
  # contrast_overview %>% mutate(ds="1/8 Unimol Contrast") %>%
  omol_overview %>% mutate(ds="Uni-Mol, OMol data") %>%
  union_all(tanimoto_overview) %>%
  union_all(omol_contrast_overview %>% mutate(ds="ConforFormer-Omol")) %>%
  # union_all(omol2_contrast_overview) %>% 
  union_all(filldataset_contrast_overview %>% mutate(ds="ConforFormer-UniMol")) %>%
  # union_all(onlycontrast_overview %>% mutate(ds="PharmIsomer posttrain")) %>%
  union_all(replicate_overview  %>% mutate(ds="UniMol replicate")) %>%
  group_by(ds, isomer_type) %>%
  arrange(desc(sim)) %>%
  mutate(run_total = cumsum(total), run_tgt = cumsum(targets)) %>%
  mutate(recall = run_tgt/sum(targets), precision = run_tgt/run_total) %>%
  ggplot(aes(recall, precision, colour=ds)) + geom_line(linewidth=0.6, alpha=0.75) +
  theme_bw(base_size = 8) + scale_color_brewer(palette = "Dark2", name="Training setup") + ylim(c(0,1)) + 
  facet_grid(isomer_type ~ .) + labs(x = "Recall", y="Precision") +
  theme(
    legend.key.height = unit(4, "mm"),
    legend.key.width  = unit(5, "mm")
    # or: legend.key.size = unit(4, "mm")  # square keys
  ) 
contrast_plot
ggsave("training_overview.pdf", plot=contrast_plot, width = 14, height=6, units="cm", dpi = 600)
# remove(sims)
isomer_sep <- unimol_overview %>%
  union_all(omol_contrast_overview) %>%
  union_all(filldataset_contrast_overview) %>%
  filter(isomer_type == "All") %>%
  mutate(non_isomers= total - targets) %>%
  rename(isomers=targets) %>%
  pivot_longer(c(isomers, non_isomers), names_to="pair_type", values_to="val") %>%
  mutate(ds=factor(ds, levels=c("Unimol", "Unimol Contrast", "Omol Contrast"), labels=c("Uni-Mol", "ConforFormer-UniMol", "ConforFormer-OMol"))) %>%
  group_by(ds, pair_type) %>%
  mutate(sim = sim/10000, density = val/sum(val)) %>%
  ggplot(aes(sim, density, colour=pair_type, fill=pair_type)) + geom_area(alpha=0.4, position="identity", linewidth=0.2) + theme_bw(base_size = 8) +
  facet_wrap(~ds) +scale_x_continuous(limits = c(0.7,1), name = "cosine similarity") +
  theme(
    legend.key.height = unit(3, "mm"),
    legend.key.width  = unit(5, "mm")
    # or: legend.key.size = unit(4, "mm")  # square keys
  ) 
isomer_sep

ggsave("isomer_separation.pdf", plot=isomer_sep, width = 14, height=4, units="cm", dpi = 600)
ggsave("isomer_separation.png", plot=isomer_sep, width = 20, height=8, units="cm", dpi = 600)

contrast_smiles <- tbl(contrast_con, "smiles")
contrast_sim_scores
contrast_smiles
bad_similar_2 <- contrast_sim_scores %>%
  filter(cos_sim > 0.980 & cos_sim < 0.985 & tgt == 0) %>%
  inner_join(contrast_smiles, by = join_by(smi_1_id == id)) %>%
  inner_join(contrast_smiles, by = join_by(smi_2_id == id), suffix = c("_1", "_2")) %>%
  collect()

good_unsimilar <- contrast_sim_scores %>%
  filter(cos_sim < 0.950 & tgt == 1) %>%
  inner_join(contrast_smiles, by = join_by(smi_1_id == id)) %>%
  inner_join(contrast_smiles, by = join_by(smi_2_id == id), suffix = c("_1", "_2")) %>%
  collect()

cis_trans <- contrast_smiles %>%
  filter(smi %like% '%/C=C\\%') %>%
  inner_join(contrast_sim_scores, by = join_by(id == smi_1_id)) %>%
  inner_join(contrast_smiles, by = join_by(smi_2_id == id), suffix = c("_1", "_2")) %>%
  filter(smi_1 %like% '%/C=C\\%' & smi_2 %like% '%/C=C/%') %>%
  collect()

cis_trans %>%
  filter(tgt == 0) %>%
  ggplot(aes(cos_sim)) + geom_histogram()

cis_trans %>%
  filter(tgt == 0) %>%
  group_by(sim = round(cos_sim, 3)) %>%
  summarize(totals = n()) %>%
  arrange(desc(sim)) %>%
  mutate(running = cumsum(totals)) %>%
  mutate(frac = running/sum(totals)) %>%
  View()

onlycontrast_con <- dbConnect(
  RSQLite::SQLite(), 
  "~/Downloads/Seed3_onlyContrast_sim.sqlite3"
)
onlycontrast_sim_scores <- tbl(onlycontrast_con, "sim_scores")
onlycontrast_smiles <- tbl(onlycontrast_con, "smiles")

bad_similar_onlycontrast <- onlycontrast_sim_scores %>%
  filter(cos_sim > 0.980 & tgt == 0 & is_enantiomer == 0) %>%
  inner_join(onlycontrast_smiles, by = join_by(smi_1_id == id)) %>%
  inner_join(onlycontrast_smiles, by = join_by(smi_2_id == id), suffix = c("_1", "_2")) %>%
  collect() 

bad_similar_onlycontrast %>%
  select(!starts_with("inchi")) %>%
  View()



contrast_overview %>% mutate(ds="1/8 Unimol Contrast") %>%
  union_all(omol_overview) %>%
  union_all(tanimoto_overview) %>%
  union_all(omol_contrast_overview) %>%
  # union_all(omol2_contrast_overview) %>% 
  union_all(filldataset_contrast_overview %>% mutate(ds="Unimol Contrast")) %>%
  union_all(onlycontrast_overview %>% mutate(ds="PharmIsomer posttrain")) %>%
  union_all(replicate_overview  %>% mutate(ds="Unimol")) %>%
  group_by(ds, isomer_type) %>%
  arrange(desc(sim)) %>%
  mutate(run_total = cumsum(total), run_tgt = cumsum(targets)) %>%
  mutate(recall = run_tgt/sum(targets), precision = run_tgt/run_total) %>%
  filter(abs(recall - 0.5) < 0.01) %>%
  group_by(isomer_type, ds) %>%
  summarize(across(c(precision, recall), mean))
