import os
import sys
import h5py
import numpy as np
import mappy as mp
import tempfile
import logging

import os
import h5py
import numpy as np
import mappy as mp
import tempfile
import logging

# -------------------------------------------------
# FUNCTION: remove_paralogous_sequences_on_the_fly
# -------------------------------------------------
def remove_paralogous_sequences_on_the_fly(
    train_data, test_h5_path, min_identity, min_coverage, output_dir, exp
):
    """
    Remove paralogous sequences between train and test datasets using mappy.
    Writes filtered sequences directly into a new HDF5 file (same structure).
    """
    print(f"\n=== Starting paralogy removal process ({exp}) ===")
    print(f"Initial train set size: {len(train_data[0])}")

    train_seqs = train_data[5]  # SEQ is at index 5

    # Create a temporary FASTA file with training sequences
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        for i, seq in enumerate(train_seqs):
            temp_file.write(f">seq{i}\n{seq}\n")
        temp_filename = temp_file.name

    # Build mappy index
    print("Creating mappy index from training sequences...")
    try:
        aligner = mp.Aligner(temp_filename, preset="map-ont")
        if not aligner:
            raise Exception("Failed to load/build mappy index")
    except Exception as e:
        logging.error(f"Error creating mappy aligner: {str(e)}")
        os.unlink(temp_filename)
        return

    # Open HDF5 files
    with h5py.File(test_h5_path, "r") as src, h5py.File(
        f"{output_dir}/filtered_test_{exp}.h5", "w"
    ) as dst, open(f"{output_dir}/removed_paralogs_{exp}.txt", "w") as fw:

        X_keys = sorted([k for k in src.keys() if k.startswith("X")], key=lambda x: int(x[1:]))
        Z_keys = sorted([k for k in src.keys() if k.startswith("Z")], key=lambda x: int(x[1:]))
        total = len(X_keys)
        print(f"Initial test set size: {total}")

        kept = 0
        removed = 0

        print("Processing test sequences and writing filtered entries...")

        for idx, x_key in enumerate(X_keys):
            seq_idx = int(x_key[1:])
            z_data = src[Z_keys[seq_idx]][:]
            name = f"seq{seq_idx}"

            # Decode sequence from one-hot to perform mapping
            x_data = src[x_key][:]
            seq = []
            for base in x_data:
                if np.all(base == 0):
                    continue
                idx_b = np.argmax(base) + 1
                if idx_b == 1:
                    seq.append("A")
                elif idx_b == 2:
                    seq.append("C")
                elif idx_b == 3:
                    seq.append("G")
                elif idx_b == 4:
                    seq.append("T")
            seq = "".join(seq)

            # Check paralogy
            is_paralogous = False
            for hit in aligner.map(seq):
                identity = hit.mlen / hit.blen
                coverage = hit.blen / len(seq)
                fw.write(f"{name}\t{identity:.4f}\t{coverage:.4f}\n")
                if identity >= min_identity and coverage >= min_coverage:
                    is_paralogous = True
                    removed += 1
                    break

            if not is_paralogous:
                # Copy all relevant datasets directly
                for prefix in ["X", "Y", "M", "Z"]:
                    dst.create_dataset(f"{prefix}{kept}", data=src[f"{prefix}{seq_idx}"][:])
                kept += 1

            if (idx + 1) % 500 == 0:
                print(f"\tProcessed {idx + 1}/{total} sequences...")

        print("\n=== Paralogy removal complete ===")
        print(f"Removed {removed} paralogous sequences.")
        print(f"Kept {kept} sequences.")
        print(f"Filtered dataset saved to filtered_test_{exp}.h5")

    os.unlink(temp_filename)


# -------------------------------------------------
# FUNCTION: load_h5_dataset (train set only)
# -------------------------------------------------
def load_h5_dataset(h5_path):
    """
    Load dataset from HDF5 file and reconstruct DNA sequences from one-hot encoding.
    Returns a list of lists: [NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL]
    """
    print(f"Loading training dataset from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        X_keys = sorted([k for k in f.keys() if k.startswith("X")], key=lambda x: int(x[1:]))
        Y_keys = sorted([k for k in f.keys() if k.startswith("Y")], key=lambda x: int(x[1:]))
        Z_keys = sorted([k for k in f.keys() if k.startswith("Z")], key=lambda x: int(x[1:]))

        X = [f[k][:] for k in X_keys]
        Y = [f[k][:] for k in Y_keys]
        Z = [f[k][:] for k in Z_keys]

    mapping = {1: "A", 2: "C", 3: "G", 4: "T"}
    seqs = []
    for arr in X:
        seq = []
        for base in arr:
            if np.all(base == 0):
                continue
            idx = np.argmax(base) + 1
            seq.append(mapping.get(idx, "N"))
        seqs.append("".join(seq))

    chroms = [z[0].decode() for z in Z]
    starts = [z[1].decode() for z in Z]
    ends = [z[2].decode() for z in Z]
    strands = [z[3].decode() for z in Z]
    names = [f"seq{i}" for i in range(len(seqs))]
    labels = [int(np.argmax(y.mean(axis=0))) for y in Y]

    return [names, chroms, strands, starts, ends, seqs, labels]


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    train_path = "dataset_train_all.h5"
    test_path = "dataset_test_1.h5"
    output_dir = "./"
    exp = "1"
    min_identity = 0.8
    min_coverage = 0.7

    train_data = load_h5_dataset(train_path)

    remove_paralogous_sequences_on_the_fly(
        train_data=train_data,
        test_h5_path=test_path,
        min_identity=min_identity,
        min_coverage=min_coverage,
        output_dir=output_dir,
        exp=exp,
    )
