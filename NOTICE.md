# Source and license notices

Task, configuration and tooling code was extracted from UniLab (Apache-2.0).
The HIM-PPO modules were extracted from unilab_rl and retain their BSD-3-Clause
SPDX headers and HIMLoco attribution. `MIGRATION_MANIFEST.json` records source
commits and pre-migration file hashes. See
[the extraction record](docs/ARCHITECTURE.md) for the ownership boundary.

Upstream HIMLoco includes a BSD-3-Clause license in its `rsl_rl/LICENSE`
(ETH Zurich / Nikita Rudin and NVIDIA) and a CC-BY-NC-SA-4.0 top-level license
(Junfeng Long and Zirui Wang). Both original notices are included in
[`LICENSES/HIMLoco.txt`](LICENSES/HIMLoco.txt) and
[`LICENSES/RSL-RL-BSD-3-Clause.txt`](LICENSES/RSL-RL-BSD-3-Clause.txt);
the repository's Apache license does not replace these upstream notices.
See https://github.com/InternRobotics/HIMLoco for upstream attribution.

Robot meshes and the floor texture are bundled from the pinned
`unilabsim/unilab-robots` dataset.
[`src/legged_manipulation_unilab/assets/manifest.json`](src/legged_manipulation_unilab/assets/manifest.json)
records the source revision and file hashes; this package does not relicense
that data. These assets ship with the repository and package; no dataset fetch
is needed at runtime.
