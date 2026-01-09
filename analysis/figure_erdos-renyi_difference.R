library(ggplot2)
library(dplyr)

source("R/helpers.R")

fitter <- "smac"
tbl_true <- read.csv("output_data/target_params/erdos-renyi.csv")
tbl_fitted <- read.csv(file.path("output_data", "fitted_features", fitter, "erdos-renyi.csv")
)
# tbl_fitted <- tbl_fitted %>%
# mutate(graph = sub("_seed=[0-9]*$", "", graph))
tbl_fitting_process <- read.csv(file.path("output_data", "fitted_params", fitter, "erdos-renyi.csv"))

tbl_fitted <- tbl_fitted %>%
    group_by(graph) %>%
    summarise(across(where(is.numeric), mean)) %>%
    ungroup()

tbl <- left_join(tbl_true, tbl_fitted, by = c("graph"), suffix = c("_true", "_fitted"))
tbl_generation_params <- read.csv("output_data/attributes/erdos-renyi.csv")
tbl <- left_join(tbl_generation_params, tbl, by = c("graph"))
tbl <- left_join(tbl, tbl_fitting_process, by = c("graph"), suffix = c("", "_fitting"))


summary(tbl)


tbl$n_diff <- abs(tbl$n_true - tbl$n_fitted) # / tbl$n_true
cat("n Pearson correlation:", cor(tbl$n_true, tbl$n_fitted, method = "pearson"), "\n")
cat("n mean absolute difference:", mean(tbl$n_diff), "\n")
p <- ggplot(tbl, aes(x = n_true, y = n_fitted, color = total_iterations)) +
    geom_point(size = 1, stroke = 0, shape = 19) +
    geom_abline(linewidth = 0.1) +
    scale_color_gradient(low = blue, high = red) +
    labs(
        x = "Target vertex count", y = "Mean actual vertex count", color = "Iterations"
    ) +
    theme(legend.position = c(0.2, 0.68))

create_pdf(file.path("output_data", "figures", fitter, "figure_erdos-renyi_difference_n.pdf"), p, width = 0.45, height = 0.5)

tbl$d_diff <- abs(tbl$d_true - tbl$d_fitted) # / tbl$d_true
cat("d Pearson correlation:", cor(tbl$d_true, tbl$d_fitted, method = "pearson"), "\n")
cat("d mean absolute difference:", mean(tbl$d_diff), "\n")
p <- ggplot(tbl, aes(x = d_true, y = d_fitted, color = total_iterations)) +
    geom_point(size = 1, stroke = 0, shape = 19) +
    geom_abline(linewidth = 0.1) +
    scale_color_gradient(low = blue, high = red) +
    labs(
        x = "Target average degree", y = "Mean actual average degree", color = "Iterations"
    ) +
    theme(legend.position = "none")

create_pdf(file.path("output_data", "figures", fitter, "figure_erdos-renyi_difference_d.pdf", p, width = 0.45, height = 0.5)

# Mean iterations
cat("Mean iterations:", mean(tbl$total_iterations), "\n")
