import glob
import os

wildcard_constraints:
    DonorsOrAcceptors = "Donors|Acceptors",

GENOME_NAME = "GRCh38"
GENOME_PATH = "/project2/yangili1/yhsz5566/LCL_chRNA-seq_SJ/GRCh38_GencodeRelease44Comprehensive"

# Path to new STAR_Align directory
SAMPLES_PATH = "/project2/yangili1/bjf79/ChromatinSplicingQTLs/code/Alignments/STAR_Align/chRNA.Expression.Splicing"

# Grab sample IDs based on folder names before "/1/Filtered.bam"
SAMPLES = [
    os.path.basename(os.path.dirname(os.path.dirname(path)))
    for path in glob.glob(f"{SAMPLES_PATH}/*/1/Filtered.bam")
]

print("Found samples:", SAMPLES)

rule all:
    input:
        "SplicingAnalysis/ObservedJuncsAnnotations/GRCh38.uniq.annotated.with_ss_scores.tsv.gz",
        "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.csv",
        "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_matrix.tsv",
        "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions.bed",
        "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_rawCounts.bed",
        "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_CPM_Normalized.bed",
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.bed.gz",
               genome=[GENOME_NAME],
               type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.saf",
               genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.Alpha.bed.gz",
                genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.Beta1.bed.gz",
                genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.Beta2.bed.gz",
                genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.SSE.bed.gz",
                genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        expand("SplicingAnalysis/SplisER_Quantifications/{genome}/{type}.SSE_details.bed.gz",
                genome=[GENOME_NAME], type=["Donors", "Acceptors"]),
        f"{GENOME_PATH}/Reference.Introns.bed.gz",
        f"{GENOME_PATH}/Reference.Introns.saf",

        
def get_bam_file(wildcards):
    return f"{SAMPLES_PATH}/{wildcards.sample}/1/Filtered.bam"

def get_bam_index(wildcards):
    return f"{SAMPLES_PATH}/{wildcards.sample}/1/Filtered.bam.bai"

def get_SJ_out_file(wildcards):
    return f"{SAMPLES_PATH}/{wildcards.sample}/SJ.out.tab"

def get_junc_file(wildcards):
    return f"SplicingAnalysis/juncfiles/{wildcards.sample}.junc"

rule ExtractJuncs:
    input:
        bam = get_bam_file,
        index = get_bam_index,
    output:
        "SplicingAnalysis/juncfiles/{sample}.junc",
    params:
        strand = "XS"
    log:
        "logs/ExtractJuncs/{sample}.log"
    shell:
        """
        (regtools junctions extract -m 20 -s {params.strand} {input.bam} > {output}) &> {log}
        """

rule annotate_juncfiles:
    input:
        gtf = f"{GENOME_PATH}/Reference.gtf",
        fa = f"{GENOME_PATH}/Reference.fa",
        fai = f"{GENOME_PATH}/Reference.fa.fai",
        juncs = get_junc_file,
    output:
        counts = "SplicingAnalysis/annotated_juncfiles/{sample}.junccounts.tsv.gz"
    log:
        "logs/annotate_juncfiles/{sample}.log"
    shell:
        """
        awk -F'\\t' 'BEGIN {{OFS=FS}} $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ {{print}}' {input.juncs} | \
        regtools junctions annotate - {input.fa} {input.gtf} | \
        awk -F'\\t' -v OFS='\\t' 'NR>1 {{$4=$1"_"$2"_"$3"_"$6; print $4, $5}}' | \
        gzip -c > {output.counts} 2> {log}
        """

rule build_junctions_matrix:
    input:
        juncs = expand("{path}/{sample}/1/SJ.out.tab", 
                       path=SAMPLES_PATH, 
                       sample=SAMPLES)
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.csv"
    log:
        "logs/build_junctions_matrix.log"
    script:
        "snakemake-workflow_rna-seq/scripts/merge_junctions.py"

rule build_junctions_matrix_bed:
    input:
        "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.csv"
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.bed"
    shell:
        """
        {{ echo -n '#'; head -n1 {input} | tr ',' '\t'; tail -n +2 {input} | tr ',' '\t'; }} > {output}
        """

rule splicemap_filter_junctions:
    input:
        matrix = "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.csv"
    output:
        filtered_matrix = "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_matrix.tsv",
        bed = "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions.bed"
    log:
        "logs/splicemap_filter_junctions.log"
    shell:
        """
        source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh
        conda activate splicemap
        python snakemake-workflow_rna-seq/scripts/splicemap_filter_junctions.py \
            --input {input.matrix} \
            --output_matrix {output.filtered_matrix} \
            --output_bed {output.bed} \
            > {log} 2>&1
        """

rule intersect_filtered_with_rawcounts:
    input:
        filtered_bed = "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions.bed",
        raw_bed = "SplicingAnalysis/ObservedJuncsAnnotations/merged_junctions_matrix.bed"
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_rawCounts.bed"
    log:
        "logs/intersect_filtered_with_rawcounts.log"
    shell:
        """
        (
            head -n1 {input.raw_bed} ;
            bedtools intersect -a {input.filtered_bed} -b {input.raw_bed} -wa -wb -f 1.0 -r
        ) > {output} 2> {log}
        """
 
rule normalize_filtered_junctions:
    input:
        raw_counts_bed = "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_rawCounts.bed"
    output:
        normalized_tsv = "SplicingAnalysis/ObservedJuncsAnnotations/splicemap_filtered_junctions_CPM_Normalized.bed"
    log:
        "logs/normalize_filtered_junctions.log"
    script:
        "snakemake-workflow_rna-seq/scripts/normalize_junctions_cpm.py"

rule ConcatJuncFilesAndKeepUniq:
    input:
        expand("SplicingAnalysis/juncfiles/{sample}.junc", sample=SAMPLES),
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.junc.gz"
    log:
        "logs/ConcatJuncFilesAndKeepUniq/{GenomeName}.log"
    shell:
        """
        # First process all junctions as before
        awk -v OFS='\\t' '{{ split($11, blockSizes, ","); JuncStart=$2+blockSizes[1]; JuncEnd=$3-blockSizes[2]; print $0, JuncStart, JuncEnd }}' {input} | \
        sort -k1,1 -k6,6 -k13,13n -k14,14n | \
        python snakemake-workflow_rna-seq/scripts/AggregateSortedCattedJuncBeds.py | \
        bedtools sort -i - | \
        # Now filter for standard chromosomes only
        awk -F'\\t' 'BEGIN {{OFS=FS}} $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ {{print}}' | \
        bgzip -c > {output} 2> {log}
        """

rule AnnotateConcatedUniqJuncFile_basic:
    input:
        junc = "SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.junc.gz",
        gtf = f"{GENOME_PATH}/Reference.gtf",
        fa = f"{GENOME_PATH}/Reference.fa",
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.annotated.tsv.gz"
    log:
        "logs/AnnotateConcatedUniqJuncFile_hg38Basic.{GenomeName}.log"
    shell:
        """
        (regtools junctions annotate {input.junc} {input.fa} {input.gtf} | gzip - > {output}) &> {log}
        """

rule Extract_introns:
    """
    Extract introns from gtf as bed file with 5'ss and 3'ss sequences
    """
    input:
        fa = f"{GENOME_PATH}/Reference.fa",
        gtf = f"{GENOME_PATH}/Reference.gtf",
    output:
        bed = f"{GENOME_PATH}/Reference.Introns.bed.gz", 
        index  = touch(f"{GENOME_PATH}/Reference.Introns.bed.gz.indexing_done"),
    log:
        "logs/Extract_introns.log"
    params:
        tabixParams = ''
    shell:
        """
        (python snakemake-workflow_rna-seq/scripts/ExtractIntronsFromGtf.py --gtf {input.gtf} --reference {input.fa} --output /dev/stdout |  awk -F'\\t' -v OFS='\\t' '{{split($4, a, "|"); print $1, $2, $3, a[2]"|"a[3], $5, $6}}' | sort | uniq | bedtools sort -i - | bgzip -c /dev/stdin > {output.bed} ) &> {log}
        tabix -f {params.tabixParams} -p bed {output.bed} && touch {output.index}
        """

rule Annotated_Introns_to_SAF:
    input:
        AnnotatedIntronsWithSS = f"{GENOME_PATH}/Reference.Introns.bed.gz", 
    output:
        SAF = f"{GENOME_PATH}/Reference.Introns.saf",
    shell:
        r"""
        zcat {input.AnnotatedIntronsWithSS} | awk 'BEGIN {{
            print "GeneID\tChr\tStart\tEnd\tStrand"
        }}
        $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ {{
            # Append chr:start-end to the splice site ID to make it unique
            print $1 ":" $2 "-" $3 "\t" $1 "\t" $2 "\t" $3 "\t" $6
        }}' > {output.SAF}
        """

rule Count_Annotated_Introns_featureCounts:
    input:
        bam = expand(f"{SAMPLES_PATH}/{{sample}}/Aligned.sortedByCoord.out.bam", sample=SAMPLES),
        index = expand(f"{SAMPLES_PATH}/{{sample}}/Aligned.sortedByCoord.out.bam.bai", sample=SAMPLES),
        gtf = f"{GENOME_PATH}/Reference.Introns.saf",
    output:
        counts = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/AnnotatedIntrons.Counts.txt",
        summary = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/AnnotatedIntrons.Counts.txt.summary",
    log:
        "logs/featureCounts/{GenomeName}.AnnotatedIntrons.log"
    threads: 16
    params:
        strand = "-s 0"  # Default to unstranded, adjust as needed
    shell:
        """
        featureCounts {params.strand} -T {threads} --ignoreDup --primary -F SAF --fracOverlapFeature 0.5 -O -a {input.gtf} -o {output.counts} {input.bam} --maxMOp 100 &> {log}
        """

rule Add_splice_site_scores_to_regtools_annotate: 
    input:
        annotated_junctions="SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.annotated.tsv.gz",
        reference_fasta = f"{GENOME_PATH}/Reference.fa",
        fai = f"{GENOME_PATH}/Reference.fa.fai",
        AnnotatedIntronsWithSS = f"{GENOME_PATH}/Reference.Introns.bed.gz", 
    output:
        "SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.annotated.with_ss_scores.tsv.gz"
    log:
        "logs/Add_splice_site_scores_to_regtools_annotate.{GenomeName}.log"
    shell:
        """
        python snakemake-workflow_rna-seq/scripts/Add_SS_To_RegtoolsAnnotate.py \
            --input {input.annotated_junctions} \
            --reference {input.reference_fasta} \
            --introns {input.AnnotatedIntronsWithSS} \
            --output {output} &> {log}
        """

rule SpliSER_IdentifySpliceSites:
    input:
        juncs = "SplicingAnalysis/ObservedJuncsAnnotations/{GenomeName}.uniq.annotated.with_ss_scores.tsv.gz",
    output:
        donors = ("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/Donors.bed"),
        acceptors = ("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/Acceptors.bed"),
    log:
        "logs/SpliSER_IdentifySpliceSites/{GenomeName}.log"
    params:
        threshold = 10
    shell:
        """
        Rscript snakemake-workflow_rna-seq/scripts/Collapse_SpliceSites.R {input.juncs} {output.donors} {output.acceptors} {params.threshold} &> {log}
        """ 

rule SpliSER_index_SpliceSites:
    input:
        bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.bed",
    output:
        bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.bed.gz",
        index = touch("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.bed.gz.indexing_done"),
    log:
        "logs/index_SpliceSites/{GenomeName}.{DonorsOrAcceptors}.log"
    shell:
        """
        (bgzip {input.bed} -c > {output.bed}) &> {log}
        (tabix -f -p bed {output.bed}) &>> {log} && touch {output.index}
        """

rule SpliSER_Count_Alpha_and_Beta2_ForAllSpliceSites:
    input:
        SpliceSites = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.bed.gz",
        juncs = expand("SplicingAnalysis/juncfiles/{sample}.junc", sample=SAMPLES)
    output:
        Alpha = ("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Alpha.bed.gz"),
        Beta2 = ("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta2.bed.gz"),
    log:
        "logs/SpliSER_Count_Alpha_and_Beta2_ForAllSpliceSites/{GenomeName}.{DonorsOrAcceptors}.log"
    shell:
        """
        python -u snakemake-workflow_rna-seq/scripts/SplisER_Count_Alpha_and_Beta2.py --SpliceSites {input.SpliceSites} --site_type {wildcards.DonorsOrAcceptors} --InputJuncs {input.juncs} --AlphaOut {output.Alpha} --Beta2Out {output.Beta2} &> {log}
        """

rule SpliSER_Making_Beta1_SAF:
    input:
        SpliceSite = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.bed.gz",
        fai = f"{GENOME_PATH}/Reference.fa.fai"
    output:
        SAF = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.saf",
    shell:
        """
        zcat {input.SpliceSite} | awk -v OFS='\\t' -F'\\t' 'NR>1 {{print $1, $2-1, $2, $4, $5, $6}}' | bedtools sort -i - | bedtools slop -s -l 3 -r 2 -i - -g {input.fai} | awk -v OFS='\\t' -F'\\t' '{{print $4, $1, $2+1, $3, $6, $5}}' > {output.SAF}
        """

rule SpliSER_Count_Beta1_ForAllSpliceSites_featureCounts:
    input:
        bam = expand(f"{SAMPLES_PATH}/{{sample}}/1/Filtered.bam", sample=SAMPLES),
        index = expand(f"{SAMPLES_PATH}/{{sample}}/1/Filtered.bam.bai", sample=SAMPLES),
        gtf = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.saf",
    output:
        counts = ("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta1_{Strandedness}.Counts.txt"),
        summary = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta1_{Strandedness}.Counts.txt.summary",
    log:
        "logs/featureCounts/{GenomeName}.{DonorsOrAcceptors}.{Strandedness}.log"
    threads: 8
    params:
        strand = "-s 0"  # Default to unstranded, adjust as needed
    shell:
        """
        featureCounts -p {params.strand} -T {threads} --ignoreDup --primary -F SAF --fracOverlapFeature 1 -O -a {input.gtf} -o {output.counts} {input.bam} --maxMOp 100 &> {log}
        """

rule SplisER_Beta1CountsToBed:
    input:
        counts = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta1_0.Counts.txt",
        Example_bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Alpha.bed.gz",
    output:
        bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta1.bed.gz",
    log:
        "logs/SplisER_Beta1CountsToBed/{GenomeName}.{DonorsOrAcceptors}.log"
    shell:
        """
        Rscript snakemake-workflow_rna-seq/scripts/SplisER_AggregateFeatureCountsBeta1Output_ToBed.R {output.bed} {input.Example_bed} {input.counts} &> {log}
        """

rule SplisER_Count_SSE:
    input:
        Alpha = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Alpha.bed.gz",
        Beta1 = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta1.bed.gz",
        Beta2 = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.Beta2.bed.gz",
    output:
        SSE = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.SSE.bed.gz",
        details = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.SSE_details.bed.gz",
    log:
        "logs/SplisER_Count_SSE/{GenomeName}.{DonorsOrAcceptors}.log"
    shell:
        """
        Rscript snakemake-workflow_rna-seq/scripts/SpliceER_FilterAndCalculateSSE.R {output.SSE} {input.Alpha} {input.Beta1} {input.Beta2} {output.details} &> {log}
        """

rule SplisER_SortAndBgzip:
    input:
        bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.{CountType}.bed.gz",
    output:
        bed = "SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.{CountType}.sorted.bed.gz",
        index = touch("SplicingAnalysis/SplisER_Quantifications/{GenomeName}/{DonorsOrAcceptors}.{CountType}.sorted.bed.gz.indexing_done"),
    log:
        "logs/SplisER_SortAndBgzip/{GenomeName}.{CountType}.{DonorsOrAcceptors}.log"
    shell:
        """
        zcat {input.bed} | sort -k1,1 -k2,2n | bgzip -c > {output.bed}
        tabix -f -p bed {output.bed} && touch {output.index}
        """

if False:
    rule BamToBigWig:
        input:
            bam=get_bam_file,
            bai=get_bam_index
        output:
            bigwig="SplicingAnalysis/bigwig/{sample}.bw"
        params:
            genome_sizes=f"{GENOME_PATH}/STARIndex/chrLength.txt"
        threads: 16
        log:
            "logs/BamToBigWig/{sample}.log"
        shell:
            """
            bamCoverage -b {input.bam} -o {output.bigwig} \
                --binSize 1 \
                --normalizeUsing CPM \
                --effectiveGenomeSize $(head -n 24 {params.genome_sizes} | awk '{{s+=$1}} END {{print s}}') \
                --extendReads \
                --ignoreDuplicates \
                --numberOfProcessors {threads} &> {log}
            """

    rule MergeBigWigs:
        input:
            expand("SplicingAnalysis/bigwig/{sample}.bw", sample=SAMPLES)
        output:
            "SplicingAnalysis/bigwig/all_sample_pooled.bw"
        log:
            "logs/MergeBigWigs/pooled.log"
        shell:
            """
            bigWigMerge {input} pooled.bedGraph
            bedGraphToBigWig pooled.bedGraph {GENOME_PATH}/chrom.sizes {output}
            rm pooled.bedGraph
            """