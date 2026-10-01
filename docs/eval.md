# Evaluation

The policy is resolved from the checkpoint's `config.yaml`, so any policy implementing
the API contract is evaluated with the featurization it was trained with.

## Drive one route

Evaluation needs a live simulator, so start CARLA in a second terminal:

```console
user@host:~/lead$ scripts/cli/start_carla
```

Then point `python -m lead` at a checkpoint directory and a route:

```console
user@host:~/lead$ python -m lead --checkpoint checkpoints/transfuser --routes src/lead/routes/benchmark_routes/longest6/00.xml
```

Routes ship under `src/lead/routes/benchmark_routes/`, one XML per route:

| benchmark   | routes                               | flag            |
| :---------- | :----------------------------------- | :-------------- |
| Longest6    | `longest6/00.xml` …                  | none            |
| Town13      | `Town13/0.xml` …                     | none            |
| Bench2Drive | `bench2drive/23687.xml` …            | `--bench2drive` |
| Fail2Drive  | `fail2drive/Base_Animals_0075.xml` … | `--fail2drive`  |

Fail2Drive needs its own simulator build under `3rd_party/CARLA/fail2drive_0915`. The
`scripts/common/eval_*.sh` scripts show single-route runs for every benchmark.

## What a run writes

Results land in `<LEAD_OUTPUT_DIR_ROOT>/local_evaluation/<route_id>/`; the directory is
deleted before the run unless `--resume` is given. `checkpoint_endpoint.json` holds the
route's scores and infractions.

Videos are config, not CLI flags; enable them through the `LEAD_CONFIG` dotlist:

```console
user@host:~/lead$ LEAD_CONFIG="evaluation.produce_demo_video=true" python -m lead --checkpoint checkpoints/transfuser --routes src/lead/routes/benchmark_routes/longest6/00.xml
```

| key                                  | what it renders                                             |
| :----------------------------------- | :---------------------------------------------------------- |
| `evaluation.produce_demo_video`      | the drive as a demo video                                   |
| `evaluation.produce_input_video`     | the sensor inputs the policy sees                           |
| `evaluation.produce_debug_video`     | predictions and controller state; on with `debug_mode=true` |
| `evaluation.produce_grid_video`      | demo and input stacked vertically                           |
| `evaluation.produce_frame_frequency` | render every n-th step, at `20 / n` fps                     |

Each of the four video keys has an `_image` counterpart that writes stills instead.

## A whole benchmark, one route at a time, with video

`run_bench2drive_per_route_v2.sh` drives one route per `python -m lead` call with the
demo and debug videos on, so every route can be watched back. It owns the CARLA
lifecycle, retries a route that crashes, and skips routes that are already done.

```console
user@host:~/lead$ bash scripts/common/run_bench2drive_per_route_v2.sh 0-19
```

Routes are picked by **index** (0-219, the order in the official `bench2drive220.xml`),
by **range**, or by the route's **XML file**:

| argument         | selects                                               |
| :--------------- | :---------------------------------------------------- |
| `0 5 219`        | those indices                                         |
| `0-19`, `0,5-10` | index ranges, and comma-separated lists of them       |
| `1711`           | route ID 1711 (any integer above 219 is an ID)        |
| `1711.xml`       | the route XML by name; a full path works too          |
| *(no argument)*  | all 220 routes; `SUBSET="0 5-10"` has the same effect |

An index in `0-219` always means an index — write `1711.xml` to name a route ID that
could be mistaken for one. Selection is resolved and validated *before* anything runs,
so a bad argument (`0 5-10 221`) exits without starting CARLA; `LIST_ONLY=1` prints the
resolved index/ID/path list without evaluating anything.

Results land in `<LEAD_OUTPUT_DIR_ROOT>/local_evaluation/Bench2Drive_220/<route_id>/`
with `checkpoint_endpoint.json`, `infractions.json`, `metric_info.json` and
`<route_id>_demo.mp4` / `<route_id>_debug.mp4`. That is the layout
`3rd_party/infraction_dashboard` reads, so it browses the whole set:

```console
user@host:~/lead$ python 3rd_party/infraction_dashboard/app.py
```

The full-benchmark run (`run_bench2drive_fast_v2.sh`) writes its single
`checkpoint_endpoint.json` into the same directory; the per-route folders sit alongside
it and are ignored by that script.

Two knobs matter on a long run. `SKIP_DONE=0` re-runs routes that already have a result.
`REQUIRE_VIDEO=1` treats a route that scored but produced no video as a failed attempt
and retries it — off by default, because a wrong `LEAD_CONFIG` would otherwise burn
every attempt of every route.

A route's `status` is what says whether it really ran: `Completed` is a full route, while
`Failed - TickRuntime` and `Failed - Agent got blocked` also reach completion and leave
their videos stopping where the route did. The script prints the status per route and
summarises the not-`Completed` ones at the end.

## A whole benchmark on SLURM

```console
user@host:~/lead$ python scripts/slurm/evaluate_policy.py \
      --checkpoint checkpoints/transfuser \
      --partition <partition> \
      --route-dir src/lead/routes/benchmark_routes/bench2drive \
      --leaderboard bench2drive
```

Run this on a login node; it stays in the foreground until the benchmark is done. Each
route XML becomes one SLURM job with a private CARLA instance. Crashed routes (`Failed`
status or zero route score) are resubmitted; routes with a good result are skipped, so
re-running only evaluates what is missing. Results land under
`<LEAD_OUTPUT_DIR_ROOT>/slurm_evaluation/<route dir>/`.

## Aggregate into benchmark scores

```console
user@host:~/lead$ python scripts/common/result_parser.py \
      --xml src/lead/routes/benchmark_routes/longest6.xml \
      --results outputs/slurm_evaluation/longest6/routes
```

This writes `results.csv` next to the result JSONs. `--xml` is the benchmark's combined
route file, shipped next to the route folders. Fail2Drive has its own parser:

```console
user@host:~/lead$ python scripts/common/f2d_result_parser.py outputs/slurm_evaluation/fail2drive/routes \
      --route-dir src/lead/routes/benchmark_routes/fail2drive
```
