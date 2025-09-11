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

global_numeric <- setdiff(global_numeric, c("total", "immersion_total"))

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
  # --- 파라미터(원하는 간격/두께로 조절) ---
  bw      <- 0.3  # 각 박스 자체 너비
  dodge   <- 0.50  # 같은 Model 내 시나리오 센터 간 거리 (도징 폭)
  capw    <- 0.25  # 캡(가로선) 길이
  
  # --- 좌표 준비: Model의 기본 x 위치 + 시나리오 도징 오프셋 ---
  model_lvls    <- levels(df[[MODEL_COL]])
  scenario_lvls <- levels(long_df$Scenario)
  x_base_map <- setNames(seq(1, by = 1.5, length.out = length(model_lvls)), model_lvls)
  
  # 시나리오 수(k)에 따라 가운데 정렬 오프셋 계산 (예: 2개면 -dodge/2, +dodge/2)
  k <- length(scenario_lvls)
  scen_offsets <- setNames(seq(-(k-1)/2, (k-1)/2, length.out = k) * dodge, scenario_lvls)
  
  # --- 상자그림 요약통계 계산 ---
  summary_df <- long_df %>%
    group_by(Measure, Scenario, Model = .data[[MODEL_COL]]) %>%
    summarise(stats = list(boxplot.stats(Value[!is.na(Value)])$stats), .groups = "drop") %>%
    mutate(
      ymin   = purrr::map_dbl(stats, ~ .x[1]),
      lower  = purrr::map_dbl(stats, ~ .x[2]),
      middle = purrr::map_dbl(stats, ~ .x[3]),
      upper  = purrr::map_dbl(stats, ~ .x[4]),
      ymax   = purrr::map_dbl(stats, ~ .x[5]),
      x_base = unname(x_base_map[as.character(Model)]),
      x_pos  = x_base + unname(scen_offsets[as.character(Scenario)])
    ) %>% select(-stats)
  
  # --- 플롯(박스 내부 세로줄기 없음 + 캡 추가, 시나리오 도징 반영) ---
  g1 <- ggplot(summary_df, aes(x = x_pos, fill = Scenario)) +
    # 수염(박스 밖만)
    geom_segment(aes(xend = x_pos, y = lower, yend = ymin),
                 linewidth = 0.6, color = "grey20") +
    geom_segment(aes(xend = x_pos, y = upper, yend = ymax),
                 linewidth = 0.6, color = "grey20") +
    # 캡(가로선)
    geom_segment(aes(x = x_pos - capw/2, xend = x_pos + capw/2, y = ymin, yend = ymin),
                 linewidth = 0.6, color = "grey20") +
    geom_segment(aes(x = x_pos - capw/2, xend = x_pos + capw/2, y = ymax, yend = ymax),
                 linewidth = 0.6, color = "grey20") +
    # 박스(Q1~Q3)
    geom_rect(aes(xmin = x_pos - bw/2, xmax = x_pos + bw/2, ymin = lower, ymax = upper),
              color = "grey30", linewidth = 0.7) +
    # 중앙값
    geom_segment(aes(x = x_pos - bw/2, xend = x_pos + bw/2, y = middle, yend = middle),
                 linewidth = 0.7, color = "grey30") +
    facet_wrap(~ Measure, ncol = 2, scales = "free_y") +
    scale_fill_brewer(palette = "Pastel1", name = "Scenario") +
    # x축: Model 기준 눈금으로 표시 (시나리오 도징은 내부 좌표만 영향)
    scale_x_continuous(
      breaks = seq_along(model_lvls),
      labels = model_lvls,
      expand = c(0.2, 0.2)
    ) +
    labs(title = "Scenario-split outcomes by Model", x = NULL, y = "Consistency Score") +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.text      = element_blank(),
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      panel.border    = element_rect(colour = "black", fill = NA, linewidth = 0.7)
    )
  
  ggsave("figures/scenario_measures_boxstrip_2.png", g1,
         width = 5, height = ceiling(nlevels(long_df$Measure) / 2) * 4.0,
         dpi = 300, bg = "white")
  message("📦 Saved → figures/scenario_measures_boxstrip_2.png")
}

# 6b) 전역 지표: immersion_ 접두어 제거해서 facet 이름 표시
# 6b) 전역 지표: 박스 내부 세로선 제거 + 캡 추가 (오류 없는 버전)
if (length(global_numeric) > 0) {
  df_global_long <- df %>%
    select(all_of(c(MODEL_COL, global_numeric))) %>%
    pivot_longer(cols = all_of(global_numeric),
                 names_to = "Measure", values_to = "Value") %>%
    mutate(
      Measure = str_remove(Measure, "^immersion_"),
      Measure = str_replace_all(Measure, "_", " "),
      Measure = str_to_sentence(Measure),
      Measure = factor(Measure)
    )
  
  
  bw   <- 0.40   # 박스 너비(데이터 좌표 단위)
  capw <- 0.30   # 캡 길이
  xpad <- 1.0    # 좌우 패딩(크게 줄수록 A-B가 더 붙어 보임)
  
  model_lvls <- levels(df[[MODEL_COL]])
  x_map <- setNames(c(1, 2), model_lvls)   # ← 고정 좌표
  
  summary_df <- df_global_long %>%
    group_by(Measure, Model = .data[[MODEL_COL]]) %>%
    summarise(stats = list(boxplot.stats(Value[!is.na(Value)])$stats), .groups = "drop") %>%
    mutate(
      ymin   = purrr::map_dbl(stats, ~ .x[1]),
      lower  = purrr::map_dbl(stats, ~ .x[2]),
      middle = purrr::map_dbl(stats, ~ .x[3]),
      upper  = purrr::map_dbl(stats, ~ .x[4]),
      ymax   = purrr::map_dbl(stats, ~ .x[5]),
      x_pos  = unname(x_map[as.character(Model)])
    ) %>% select(-stats)
  
  g2 <- ggplot(summary_df, aes(x = x_pos, fill = Model)) +
    # whisker stem (밖만)
    geom_segment(aes(xend = x_pos, y = lower, yend = ymin), linewidth = 0.6, color = "grey20") +
    geom_segment(aes(xend = x_pos, y = upper, yend = ymax), linewidth = 0.6, color = "grey20") +
    # caps
    geom_segment(aes(x = x_pos - capw/2, xend = x_pos + capw/2, y = ymin, yend = ymin),
                 linewidth = 0.6, color = "grey20") +
    geom_segment(aes(x = x_pos - capw/2, xend = x_pos + capw/2, y = ymax, yend = ymax),
                 linewidth = 0.6, color = "grey20") +
    # box
    geom_rect(aes(xmin = x_pos - bw/2, xmax = x_pos + bw/2, ymin = lower, ymax = upper),
              color = "grey30", linewidth = 0.7) +
    # median
    geom_segment(aes(x = x_pos - bw/2, xend = x_pos + bw/2, y = middle, yend = middle),
                 linewidth = 0.7, color = "grey30") +
    facet_wrap(~ Measure, nrow = 1, scales = "fixed") +
    fill_scale_model +
    scale_x_continuous(
      breaks = unname(x_map), labels = names(x_map),
      limits = c(min(x_map) - xpad, max(x_map) + xpad),  # ← 여기로 간격 조절
      expand = c(0, 0)
    ) +
    labs(x = NULL, y = "GEQ Score") +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.text      = element_text(face = "bold"),
      plot.title      = element_blank(),
      panel.border    = element_rect(colour = "black", fill = NA, linewidth = 0.7),
      panel.spacing.x = unit(0.15, "lines")
    )
  
  ggsave("figures/global_measures_boxstrip_2.png", g2,
         width = 8, height = 3.5, dpi = 300, bg = "white")
  message("📦 Saved → figures/global_measures_boxstrip_2.png")
}

message("✅ Done (grey box borders, no scenario points, immersion_ removed from global facets).")