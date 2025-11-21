import numpy as np
import pandas as pd
from pybedtools import BedTool
# Define constants
SPECIES = "Human"
GTF = "gencode.v44.annotation.gtf"

def get_feature(fname, feature = "exon"):
    ss2gene = {}
    if ".gz" in fname:
        F = gzip.open(fname)
    else:
        F = open(fname)
    for ln in F:
        if ln[0] == "#": continue
        ln = ln.split('\t')
        gID = ln[-1].split('gene_id "')[1].split('"')[0]
        if ln[2] != feature:
            continue
        strand = ln[6]
        ss2gene[(ln[0], int(ln[3]), strand)] = gID
        ss2gene[(ln[0], int(ln[4]), strand)] = gID
    return ss2gene

rule all:
    input:
        "%s.merged.usage.bed" % SPECIES,
        "splice_table_%s.txt" % SPECIES,
        "%s.donor.usage.bed" % SPECIES,
        "%s.acceptor.usage.bed" % SPECIES,
        #"splice_table_%s.test.txt" % SPECIES

rule process_sse_files:
    input:
        donor="SplicingAnalysis/SplisER_Quantifications/GRCh38/Donors.SSE.bed.gz",
        acceptor="SplicingAnalysis/SplisER_Quantifications/GRCh38/Acceptors.SSE.bed.gz"
    output:
        donor="%s.donor.usage.bed" % SPECIES,
        acceptor="%s.acceptor.usage.bed" % SPECIES,
        merged="%s.merged.usage.bed" % SPECIES
    run:
        import pandas as pd
        import gzip
 
        def sse_to_color(val, site_type):
            """
            Map SSE mean [0,1] to a gradient from white → pure color.
            Donor: white → blue
            Acceptor: white → red
            """
            val = max(0.0, min(1.0, val))  # clamp
            
            if site_type == "donor":
                # White → Blue
                r = int(255 * (1 - val))
                g = int(255 * (1 - val))
                b = 255
            else:
                # White → Red
                r = 255
                g = int(255 * (1 - val))
                b = int(255 * (1 - val))
            
            return f"{r},{g},{b}"

        def process_sse_file(filepath, site_type):
            # Pick opener
            open_func = gzip.open if filepath.endswith(".gz") else open
            
            # Extract header
            with open_func(filepath, "rt") as f:
                for line in f:
                    if line.startswith('#'):
                        header = line.lstrip('#').strip().split('\t')
                        break
            
            # Load dataframe
            df = pd.read_csv(filepath, sep='\t', comment='#', header=None, names=header)
            
            # Standard BED columns
            bed_cols = ['chrom', 'start', 'end', 'name', 'score', 'strand']
            df.columns = bed_cols + list(df.columns[6:])
            
            # Compute mean SSE across samples
            df["LCL_mean"] = df.iloc[:, 6:].fillna(0).mean(axis=1)
            
            # Map mean SSE to RGB gradient
            df["itemRgb"] = df["LCL_mean"].apply(lambda v: sse_to_color(v, site_type))
            
            # BED with color → 9 fields
            df["thickStart"] = df["start"]
            df["thickEnd"] = df["end"]
            
            output_cols = ['chrom', 'start', 'end', 'name', 'score', 'strand',
                           'thickStart', 'thickEnd', 'itemRgb', 'LCL_mean']
            return df[output_cols]
        
        # Process separately
        donor_df = process_sse_file(input.donor, "donor")
        acceptor_df = process_sse_file(input.acceptor, "acceptor")
        
        # Write out donor and acceptor
        donor_df.to_csv(output.donor, sep='\t', index=False, header=False)
        acceptor_df.to_csv(output.acceptor, sep='\t', index=False, header=False)

        
        # Combine and save
        merged_df = pd.concat([donor_df, acceptor_df])
        merged_df.to_csv(output.merged, sep='\t', index=False, header=False)
        
rule splice_table:
    input:
        sites="%s.merged.usage.bed" % SPECIES,
        gtf=GTF,
    output:
        "splice_table_%s.txt" % SPECIES
    run:
        snps = BedTool(input.sites)
        genes = BedTool(input.gtf)
        intersection = genes.intersect(snps, wa=True, wb=True)
        gene_dict, snps_list = {}, []
        for snp in intersection:
            if snp.fields[2] == "gene":
                lcl = snp.fields[-1]
                parts = snp.fields[-7].split(':')
                chrom, pos, ss_strand, ss_type = parts
                pos = (int(float(pos)))  # convert scientific notation to integer string
                # fix shift
                if (ss_strand == "+" and ss_type == "D") or (ss_strand == "-" and ss_type == "A"):
                    pos -= 1
                elif (ss_strand == "+" and ss_type == "A") or (ss_strand == "-" and ss_type == "D"):
                    pos += 1
                
                ss_id = f"{chrom}|{pos}|{ss_strand}|{ss_type}"      
                gene_strand = snp.fields[6]  # strand information
                if gene_strand != ss_strand:
                    continue  # Skip if strand does not match
                gene = snp["gene_id"]
                snps_list.append(ss_id)
                entry = "%s:%s" % (ss_id, lcl)
                try:
                    gene_dict["%s,%s,%s,%s,%s" % (snp.chrom,gene,snp.start,snp.end,ss_strand)].append(entry)
                except KeyError:
                    gene_dict["%s,%s,%s,%s,%s" % (snp.chrom,gene,snp.start,snp.end,ss_strand)] = [entry]
        fout = open(output[0], 'w')
        for gene, snps in gene_dict.items():
            gene = gene.split(',')
            fout.write("%s\t0\t%s\t%s\t%s\t%s\t%s\n" % (gene[1],gene[0],gene[-1],gene[2],gene[3],';'.join(snps)+';'))
