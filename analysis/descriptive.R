setwd("C:/.Soyeon/GameAgent/LLMGA")

suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
  library(readr)
})

# 폴더 준비
dir.create("processed_output", showWarnings = FALSE, recursive = TRUE)
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# 1) 데이터 로드 (wide format)
# -----------------------------
df <- read_csv("data/questionnaire/final_scores_with_immersion.csv",
               show_col_types = FALSE)

ID_COL    <- "ID"
MODEL_COL <- "Model"
stopifnot(ID_COL %in% names(df), MODEL_COL %in% names(df))
df[[MODEL_COL]] <- factor(df[[MODEL_COL]])

# -------------------------------------------
# 2) Hamlet_/Venice_ 페어 탐지
# -------------------------------------------
scenario_prefixes <- c("Hamlet", "Venice")
pattern <- "^(Hamlet|Venice)_(.+)$"

pairs_tbl <- tibble(col = names(df)) %>%
  mutate(Scenario = str_match(col, pattern)[, 2],
         Measure  = str_match(col, pattern)[, 3]) %>%
  filter(!is.na(Scenario), !is.na(Measure))

scenario_measures <- pairs_tbl %>%
  count(Measure, Scenario) %>%
  complete(Measure, Scenario = scenario_prefixes, fill = list(n = 0)) %>%
  group_by(Measure) %>%
  summarize(has_both = all(n > 0), .groups = "drop") %>%
  filter(has_both) %>%
  pull(Measure) %>%
  sort()

# -------------------------------------------
# 3) long_df 구성
# -------------------------------------------
long_df <- tibble()
if (length(scenario_measures) > 0) {
  long_df <- df %>%
    select(all_of(c(ID_COL, MODEL_COL)), matches(pattern)) %>%
    pivot_longer(cols = matches(pattern),
                 names_to = c("Scenario", "Measure"),
                 names_pattern = pattern,
                 values_to = "Value") %>%
    filter(Measure %in% scenario_measures) %>%
    mutate(
      Scenario = factor(Scenario, levels = scenario_prefixes),
      # facet 라벨을 "score" → "Score" 로 교체
      Measure  = recode(Measure, "score" = "Score"),
      Measure  = factor(Measure)
    )
  write_csv(long_df, "processed_output/long_scenario_data.csv")
  message("✅ LONG format saved → processed_output/long_scenario_data.csv")
}

# -------------------------------------------
# 4) 전역(비-시나리오) 수치형 지표
# -------------------------------------------
all_numeric <- df %>% select(where(is.numeric)) %>% names()
paired_numeric_cols <- pairs_tbl %>%
  filter(Measure %in% scenario_measures) %>%
  pull(col) %>% unique()
global_numeric <- setdiff(all_numeric, c(paired_numeric_cols, ID_COL))

message("📌 Scenario measures (paired): ",
        ifelse(length(scenario_measures) > 0, paste(scenario_measures, collapse = ", "), "(none)"))
message("📌 Global numeric outcomes: ",
        ifelse(length(global_numeric) > 0, paste(global_numeric, collapse = ", "), "(none)"))

# -------------------------------------------
# 5) 공통 팔레트(전역 버전과 동일): A/B 색 통일
# -------------------------------------------
model_levels <- levels(df[[MODEL_COL]])
base_palette <- c("A" = "#A8D5BA",  # pastel green
                  "B" = "#A7C7E7")  # pastel blue
model_palette <- setNames(rep("#CFCFCF", length(model_levels)), model_levels)
for (nm in names(base_palette)) if (nm %in% model_levels) model_palette[nm] <- base_palette[[nm]]

fill_scale_model <- scale_fill_manual(
  values = model_palette, limits = model_levels, drop = FALSE, name = "Model"
)

# 공통 테마: 축/패널 테두리
panel_border_theme <- theme(
  panel.border   = element_rect(colour = "black", fill = NA, linewidth = 0.7),
  plot.background = element_rect(colour = NA, fill = "white")
)

# -------------------------------------------
# 6) Visualization with ggplot2
# -------------------------------------------

# 6a) 시나리오 분할: x=Model, 시나리오별 도징, 점 없음
if (nrow(long_df) > 0) {
  g1 <- ggplot(long_df, aes(x = .data[[MODEL_COL]], y = Value, fill = Scenario)) +
    geom_boxplot(
      position = position_dodge(width = 0.75),
      outlier.shape = NA, width = 0.4, color = "grey30"
    ) +
    facet_wrap(~ Measure, ncol = 2, scales = "free_y") +
    scale_fill_brewer(palette = "Pastel1", name = "Scenario") +
    labs(title = "Scenario-split outcomes by Model", x = NULL, y = "GEQ Score") +  # y축 라벨만 유지
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.text      = element_blank(),  # ← facet 제목 숨김
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      panel.border    = element_rect(colour = "black", fill = NA, linewidth = 0.7)
    )
  
  ggsave("figures/scenario_measures_boxstrip_2.png", g1,
         width = 5, height = ceiling(nlevels(long_df$Measure) / 2) * 4.0,
         dpi = 300, bg = "white")
  message("📦 Saved → figures/scenario_measures_boxstrip_2.png")
}

# 6b) 전역 지표: immersion_ 접두어 제거해서 facet 이름 표시
# 6b) 전역 지표: 가로 한 줄(6칸) + y축 라벨 "GEQ Score"
if (length(global_numeric) > 0) {
  df_global_long <- df %>%
    select(all_of(c(MODEL_COL, global_numeric))) %>%
    pivot_longer(cols = all_of(global_numeric),
                 names_to = "Measure", values_to = "Value") %>%
    mutate(
      Measure = str_remove(Measure, "^immersion_"),
      Measure = str_replace_all(Measure, "_", " "),
      Measure = str_to_sentence(Measure),   # 첫 글자 대문자 (time loss 등)
      Measure = factor(Measure)
    )
  
  g2 <- ggplot(df_global_long,
               aes(x = .data[[MODEL_COL]], y = Value, fill = .data[[MODEL_COL]])) +
    geom_boxplot(outlier.shape = NA, width = 0.4, color = "grey30") +
    facet_wrap(~ Measure, nrow = 1, scales = "free_y") +   # ← 한 줄로
    fill_scale_model +
    labs(title = "Global outcomes by Model", x = NULL, y = "GEQ Score") +  # ← y축 라벨
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.text      = element_text(face = "bold"),
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      panel.border    = element_rect(colour = "black", fill = NA, linewidth = 0.7)
    )
  
  ggsave("figures/global_measures_boxstrip_2.png", g2,
         width = 12, height = 3.5,   # ← 가로 1행에 맞게 사이즈 조정
         dpi = 300, bg = "white")
  message("📦 Saved → figures/global_measures_boxstrip_2.png")
}

message("✅ Done (grey box borders, no scenario points, immersion_ removed from global facets).")