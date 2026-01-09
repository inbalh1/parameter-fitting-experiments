## general theme
library(showtext)
theme_set(theme_bw())

## latex font
font_add("Linux Libertine O", "R/LinLibertine_R.ttf")
showtext_auto()
theme_update(text = element_text(family = "Linux Libertine O", size = 8))

## some colors
blue <- "#5EA6E5"
green <- "#6EC620"
yellow <- "#EEC200"
red <- "#E62711"
violet <- "#c6468d"
purple <- "#613872"
gray <- "#666666"

## pdf output; width and height relative to ACM text width
create_pdf <- function(file, plot, width = 1, height = 0.43) {
    textwidth <- 15 / 2.54
    pdf(file, width = width * textwidth, height = height * textwidth)
    print(plot)
    no_output <- dev.off()
}


sanitize_number <- function(accuracy) {
    function(num) {
        paste0("\\num{", scales::number(num, accuracy = accuracy, big.mark = ""), "}")
    }
}

sanitize_percent <- function(accuracy) {
    function(num) {
        paste0("\\SI{", scales::number(num * 100, accuracy = accuracy, big.mark = ""), "}{\\%}")
    }
}

sanitize_text <- xtable::sanitize

sanitize_table <- function(tbl, functions) {
    for (i in seq_len(ncol(tbl))) {
        tbl[, i] <- functions[[i]](tbl[, i])
    }
    return(tbl)
}
