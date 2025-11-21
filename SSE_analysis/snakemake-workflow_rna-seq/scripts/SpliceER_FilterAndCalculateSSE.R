#!/usr/bin/env Rscript

######################################################################
# @author      : Your Name (your@email.com)
# @file        : SpliceER_FilterAndCalculateSSE.R
# @created     : 2025-06-09 11:11
#
# @description : 
######################################################################

# Use hard coded arguments in interactive R session, else use command line args
if(interactive()){
    args <- scan(text=
                 "scratch/test.SSE.bed.gz rna-seq/SplicingAnalysis/SplisER_Quantifications/GRCh38_GencodeRelease44Comprehensive/Donors.Alpha.sorted.bed.gz rna-seq/SplicingAnalysis/SplisER_Quantifications/GRCh38_GencodeRelease44Comprehensive/Donors.Beta1.sorted.bed.gz rna-seq/SplicingAnalysis/SplisER_Quantifications/GRCh38_GencodeRelease44Comprehensive/Donors.Beta2.sorted.bed.gz ", what='character')
} else{
    args <- commandArgs(trailingOnly=TRUE)
}

f_out <- args[1]          # Output file for SSE values
f_details <- args[5]      # Output file for detailed information
f_donor_alpha <- args[2]
f_donor_beta1 <- args[3]
f_donor_beta2 <- args[4]

library(tidyverse)
library(data.table)

# Read input files
bed6 <- fread(f_donor_alpha, select=c(1:6))
alpha <- fread(f_donor_alpha, drop=c(1,2,3,5,6)) %>%
    column_to_rownames("name") %>% as.matrix()
beta1 <- fread(f_donor_beta1, drop=c(1,2,3,5,6)) %>%
    column_to_rownames("name") %>% as.matrix()
beta2 <- fread(f_donor_beta2, drop=c(1,2,3,5,6)) %>%
    column_to_rownames("name") %>% as.matrix()

sum.counts <- alpha + beta1 + beta2
SSE <- alpha / sum.counts

# Write out SSE matrix as bed6+, with 4 sigfigs
SSE_out <- cbind(bed6, round(SSE, 4))
write_tsv(SSE_out, file = f_out)

# Create detailed output with alpha, beta1, beta2, and SSE for each sample
detailed_output <- bed6

# Get sample names (assuming all matrices have same columns)
sample_names <- colnames(alpha)

for (sample in sample_names) {
    # Create a data frame for each sample's metrics
    sample_df <- data.frame(
        alpha = alpha[, sample],
        beta1 = beta1[, sample],
        beta2 = beta2[, sample],
        SSE = SSE[, sample]
    )
    
    # Add sample suffix to column names
    colnames(sample_df) <- paste(colnames(sample_df), sample, sep = "_")
    
    # Add to detailed output
    detailed_output <- cbind(detailed_output, sample_df)
}

# Round numeric columns to 4 decimal places
detailed_output <- detailed_output %>%
    mutate(across(where(is.numeric), ~ round(., 4)))

# Write detailed output
write_tsv(detailed_output, file = f_details)