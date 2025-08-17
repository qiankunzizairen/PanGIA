#!/usr/bin/env python3
import sys, os, re, gzip

def opengz(path, mode='rt'):
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    return open(path, mode)

def flush_record(h, seqbuf, fa_out, stat_out, totals):
    if h is None:
        return
    seq_raw = ''.join(seqbuf)
    seq = re.sub(r'\s+', '', seq_raw).upper()

    u_replaced = seq.count('U')
    seq_t = seq.replace('U', 'T')

    n_count = seq_t.count('N')
    L = len(seq_t)
    n_frac = (n_count / L) if L > 0 else 0.0

    fa_out.write(f">{h}\n{seq_t}\n")

    stat_out.write(f"{h}\t{L}\t{u_replaced}\t{n_count}\t{n_frac:.6f}\n")

    totals['records'] += 1
    totals['bases'] += L
    totals['U_replaced'] += u_replaced
    totals['N_count'] += n_count

def main():
    if len(sys.argv) != 4:
        print("Usage: python fasta_u2t_countN.py <input.fasta[.gz]> <output.fasta[.gz]> <stats.tsv>", file=sys.stderr)
        sys.exit(1)

    in_fa, out_fa, out_stat = sys.argv[1], sys.argv[2], sys.argv[3]
    totals = dict(records=0, bases=0, U_replaced=0, N_count=0)

    with opengz(in_fa, 'rt') as fin, opengz(out_fa, 'wt') as fout, open(out_stat, 'w') as fs:
        fs.write("id\tlength\tU_replaced\tN_count\tN_fraction\n")

        header, buf = None, []
        for line in fin:
            if line.startswith('>'):
                flush_record(header, buf, fout, fs, totals)
                header = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        flush_record(header, buf, fout, fs, totals)


    print(f"[DONE] records={totals['records']}, bases={totals['bases']}, "
          f"U_replaced={totals['U_replaced']}, N_total={totals['N_count']}",
          file=sys.stderr)

if __name__ == "__main__":
    main()