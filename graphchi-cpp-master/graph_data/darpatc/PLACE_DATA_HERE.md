# Data goes here

This branch keeps the path convention inherited from the original ThreaTrace
repo layout: parsed CADETS provenance TSVs are read from this directory by
default. The path is not touched in this branch (see the root `README.md`,
"Known rough edges" — replacing it with a plain `data/` directory is future
cleanup, not done here).

Before running anything, place these two files here (copy from another
machine that already has them, or produce them by running
`scripts/parse_darpatc.py` — see the root `README.md`):

- `cadets_train.txt`
- `cadets_test.txt`

Both files are large (~580 MB each) and are deliberately **not** committed to
git — see `.gitignore`. Nothing else needs to go in this directory.
